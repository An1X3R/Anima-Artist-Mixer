"""Temporary, removable end-to-end Anima conditioning probes.

The probes wrap public ComfyUI Python call boundaries at runtime; no ComfyUI
source file is modified.  They are strictly observational: tensors are never
retained or changed, original return values pass through unchanged, and a
diagnostic failure is reduced to a warning.

Set ``ANIMA_MIXER_CONDITION_DIAG=0`` (or the existing master switch
``ANIMA_MIXER_SEMANTIC_DIAG=0``) before starting ComfyUI to skip installation.
Call :func:`uninstall` to restore every wrapped callable in a running process.
Deleting this file and the ``TEMP_CONDITIONING_DIAG_HOOK`` block in the plugin
entrypoint removes the probes without touching Mixer behavior.
"""

from collections import OrderedDict
import contextvars
import functools
import hashlib
import inspect
import itertools
import logging
import os
import threading
import time
import weakref

import torch


logger = logging.getLogger(__name__)

MASTER_ENV_NAME = "ANIMA_MIXER_SEMANTIC_DIAG"
ENV_NAME = "ANIMA_MIXER_CONDITION_DIAG"
_FALSE_VALUES = {"0", "false", "off", "no", "disable", "disabled"}
_HISTORY_LIMIT = 256
_WRAPPER_MARKER = "_anima_condition_diag_wrapper"
_ORIGINAL_MARKER = "_anima_condition_diag_original"

_install_lock = threading.RLock()
_state_lock = threading.RLock()
_error_lock = threading.Lock()
_installed_patches = []
_reported_errors = OrderedDict()
_encode_lineage = OrderedDict()
_encode_history = OrderedDict()
_model_cond_history = OrderedDict()
_uuid_lineage = OrderedDict()
_guider_samples = weakref.WeakKeyDictionary()
_guider_samples_by_id = OrderedDict()
_current_sample = contextvars.ContextVar(
    "anima_condition_diag_sample",
    default=None,
)
_current_model_cond = contextvars.ContextVar(
    "anima_condition_diag_model_cond",
    default=None,
)
_current_preprocess = contextvars.ContextVar(
    "anima_condition_diag_preprocess",
    default=None,
)
_id_counter = itertools.count(1)


def _env_enabled(name, default="1"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in _FALSE_VALUES


def is_enabled():
    return _env_enabled(MASTER_ENV_NAME) and _env_enabled(ENV_NAME)


def _next_id(prefix):
    return f"{prefix}-{os.getpid()}-{next(_id_counter)}"


def _bounded_put(mapping, key, value):
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > _HISTORY_LIMIT:
        mapping.popitem(last=False)


def _clean_field(value, limit=8000):
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _log_event(event, *, level=logging.INFO, **fields):
    payload = " ".join(
        f"{key}={_clean_field(value)}"
        for key, value in fields.items()
    )
    logger.log(
        level,
        "[AnimaConditionDiag] event=%s%s",
        event,
        " " + payload if payload else "",
    )


def _report_diagnostic_error(hook, error):
    key = (str(hook), type(error).__name__, str(error))
    with _error_lock:
        if key in _reported_errors:
            return
        _bounded_put(_reported_errors, key, True)
    try:
        _log_event(
            "diagnostic_error",
            level=logging.WARNING,
            hook=hook,
            error_type=type(error).__name__,
            error=str(error),
        )
    except Exception:
        pass


def _safe_observe(hook, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        _report_diagnostic_error(hook, error)
        return None


def _hash_bytes(data, size=10):
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def _hash_parts(*parts):
    digest = hashlib.blake2b(digest_size=10)
    for part in parts:
        encoded = repr(part).encode("utf-8", errors="backslashreplace")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _update_structured_hash(digest, value):
    if torch.is_tensor(value):
        digest.update(b"tensor")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        raw = (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_structured_hash(digest, key)
            _update_structured_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list" if isinstance(value, list) else b"tuple")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _update_structured_hash(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8", errors="replace"))
    digest.update(repr(value).encode("utf-8", errors="backslashreplace"))


def _structured_fingerprint(value):
    digest = hashlib.blake2b(digest_size=10)
    _update_structured_hash(digest, value)
    return digest.hexdigest()


def _unwrap_tensor(value):
    if torch.is_tensor(value):
        return value
    try:
        nested = getattr(value, "cond", None)
    except Exception:
        nested = None
    return nested if torch.is_tensor(nested) else None


def _tensor_snapshot(value):
    tensor = _unwrap_tensor(value)
    if tensor is None:
        return {
            "is_tensor": False,
            "type": type(value).__name__,
            "bad": None,
            "nan": None,
            "inf": None,
            "fingerprint": None,
        }

    try:
        version = int(tensor._version)
    except Exception:
        version = None
    snapshot = {
        "is_tensor": True,
        "object_id": f"{id(tensor):x}",
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "version": version,
        "bad": None,
        "nan": None,
        "inf": None,
        "finite_max_abs": None,
        "fingerprint": None,
    }
    try:
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        finite = torch.isfinite(cpu_tensor)
        snapshot["bad"] = int((~finite).sum().item())
        if torch.is_floating_point(cpu_tensor) or torch.is_complex(cpu_tensor):
            snapshot["nan"] = int(torch.isnan(cpu_tensor).sum().item())
            snapshot["inf"] = int(torch.isinf(cpu_tensor).sum().item())
        else:
            snapshot["nan"] = 0
            snapshot["inf"] = 0
        finite_values = cpu_tensor[finite]
        if finite_values.numel():
            snapshot["finite_max_abs"] = float(
                finite_values.abs().max().item()
            )
        raw = (
            cpu_tensor.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        snapshot["fingerprint"] = _hash_bytes(raw, size=8)
    except Exception as error:
        snapshot["snapshot_error"] = str(error)
    return snapshot


def _snapshot_key(snapshot):
    if not isinstance(snapshot, dict) or not snapshot.get("is_tensor", False):
        return None
    fingerprint = snapshot.get("fingerprint")
    if fingerprint is None:
        return None
    return (
        tuple(snapshot.get("shape") or ()),
        str(snapshot.get("dtype")),
        str(fingerprint),
    )


def _snapshot_fp(snapshot):
    key = _snapshot_key(snapshot)
    return "none" if key is None else key[2]


def _snapshot_bad(snapshot):
    value = snapshot.get("bad") if isinstance(snapshot, dict) else None
    return None if value is None else int(value)


def _snapshot_summary(snapshot):
    if not isinstance(snapshot, dict) or not snapshot.get("is_tensor", False):
        return f"type:{snapshot.get('type', 'unknown') if isinstance(snapshot, dict) else 'unknown'}"
    shape = "x".join(str(value) for value in snapshot.get("shape") or ())
    return (
        f"shape:{shape},dtype:{snapshot.get('dtype')},device:{snapshot.get('device')},"
        f"object:{snapshot.get('object_id')},version:{snapshot.get('version')},"
        f"bad:{snapshot.get('bad')},nan:{snapshot.get('nan')},"
        f"inf:{snapshot.get('inf')},max:{snapshot.get('finite_max_abs')},"
        f"fp:{snapshot.get('fingerprint')}"
    )


def _value_identity(value):
    tensor = _unwrap_tensor(value)
    if tensor is None:
        return (type(value).__name__, id(value))
    try:
        version = int(tensor._version)
    except Exception:
        version = None
    return (
        id(tensor),
        version,
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _condition_entry(item, index):
    raw = None
    metadata = {}
    if isinstance(item, dict):
        metadata = item
        raw = item.get("cross_attn")
    elif isinstance(item, (list, tuple)) and item:
        raw = item[0]
        if len(item) > 1 and isinstance(item[1], dict):
            metadata = item[1]

    model_conds = metadata.get("model_conds")
    if not isinstance(model_conds, dict):
        model_conds = {}
    return {
        "index": int(index),
        "uuid": str(metadata.get("uuid", "none")),
        "raw": _tensor_snapshot(raw),
        "ids": _tensor_snapshot(metadata.get("t5xxl_ids")),
        "weights": _tensor_snapshot(metadata.get("t5xxl_weights")),
        "pooled": _tensor_snapshot(metadata.get("pooled_output")),
        "attention": _tensor_snapshot(metadata.get("attention_mask")),
        "model": _tensor_snapshot(model_conds.get("c_crossattn")),
    }


def _conditioning_records(conditioning):
    if not isinstance(conditioning, (list, tuple)):
        return ()
    return tuple(
        _condition_entry(item, index)
        for index, item in enumerate(conditioning)
    )


def _records_summary(records):
    parts = []
    for record in records or ():
        parts.append(
            f"{record.get('index')}[uuid:{record.get('uuid')};"
            f"raw:{{{_snapshot_summary(record.get('raw'))}}};"
            f"ids:{{{_snapshot_summary(record.get('ids'))}}};"
            f"weights:{{{_snapshot_summary(record.get('weights'))}}};"
            f"pooled:{{{_snapshot_summary(record.get('pooled'))}}};"
            f"attention:{{{_snapshot_summary(record.get('attention'))}}};"
            f"model:{{{_snapshot_summary(record.get('model'))}}}]"
        )
    return "|".join(parts) or "none"


def _record_value_key(records, field):
    values = tuple(_snapshot_key(record.get(field)) for record in records or ())
    return None if not values or any(value is None for value in values) else values


def _record_value_fp(records, field):
    key = _record_value_key(records, field)
    return "none" if key is None else _hash_parts(key)


def _records_bad(records, field):
    values = [_snapshot_bad(record.get(field)) for record in records or ()]
    return None if not values or any(value is None for value in values) else sum(values)


def _metadata_for_item(item):
    if isinstance(item, dict):
        return item
    if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict):
        return item[1]
    return {}


def _is_anima_conditioning(conditioning):
    if not isinstance(conditioning, (list, tuple)):
        return False
    for item in conditioning:
        metadata = _metadata_for_item(item)
        if "t5xxl_ids" in metadata:
            return True
        model_conds = metadata.get("model_conds")
        if isinstance(model_conds, dict) and "t5xxl_ids" in model_conds:
            return True
    return False


def _is_anima_tokens(tokens):
    return isinstance(tokens, dict) and (
        "t5xxl" in tokens
        or "t5xxl_ids" in tokens
        or "qwen3_06b" in tokens
    )


def _is_anima_model(model):
    if model is None:
        return False
    candidates = [model, getattr(model, "diffusion_model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        name = (
            f"{candidate.__class__.__module__}."
            f"{candidate.__class__.__qualname__}"
        ).lower()
        if "anima" in name:
            return True
    return False


def _patcher_identity(patcher):
    if patcher is None:
        return {
            "patcher": "none",
            "requested": "none",
            "loaded": "none",
            "patches": "none",
            "clone": "none",
        }
    model = getattr(patcher, "model", None)
    patches = getattr(patcher, "patches", None)
    try:
        patch_count = len(patches) if patches is not None else 0
    except Exception:
        patch_count = "unknown"
    return {
        "patcher": f"{id(patcher):x}",
        "requested": str(getattr(patcher, "patches_uuid", None)),
        "loaded": str(getattr(model, "current_weight_patches_uuid", None)),
        "patches": patch_count,
        "clone": str(getattr(patcher, "clone_base_uuid", None)),
    }


def _patch_summary(identity):
    identity = identity or {}
    return (
        f"patcher:{identity.get('patcher')},"
        f"requested:{identity.get('requested')},"
        f"loaded:{identity.get('loaded')},"
        f"patches:{identity.get('patches')},"
        f"clone:{identity.get('clone')}"
    )


def _bound_model_and_patch(model_function):
    model = getattr(model_function, "__self__", None)
    patcher = getattr(model, "current_patcher", None)
    return model, _patcher_identity(patcher)


def _caller_details():
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        for _depth in range(16):
            if frame is None:
                break
            module = str(frame.f_globals.get("__name__", "unknown"))
            function = str(frame.f_code.co_name)
            if module not in {__name__, "comfy.sd"}:
                text_value = frame.f_locals.get("text")
                if not isinstance(text_value, str):
                    text_value = frame.f_locals.get("base")
                if not isinstance(text_value, str):
                    text_value = None
                return {
                    "caller": f"{module}.{function}",
                    "text_fp": (
                        "none"
                        if text_value is None
                        else _hash_parts(text_value)
                    ),
                    "text_chars": (
                        "unknown" if text_value is None else len(text_value)
                    ),
                }
            frame = frame.f_back
    finally:
        del frame
    return {"caller": "unknown", "text_fp": "none", "text_chars": "unknown"}


def _register_encode_lineage(snapshot, lineage):
    key = _snapshot_key(snapshot)
    if key is None:
        return
    with _state_lock:
        _bounded_put(_encode_lineage, key, dict(lineage))


def _lookup_encode_lineage(snapshot):
    key = _snapshot_key(snapshot)
    if key is None:
        return None
    with _state_lock:
        value = _encode_lineage.get(key)
        return None if value is None else dict(value)


def _register_uuid(uuid_value, record):
    uuid_value = str(uuid_value or "none")
    if uuid_value == "none":
        return
    with _state_lock:
        _bounded_put(_uuid_lineage, uuid_value, dict(record))


def _lookup_uuid(uuid_value):
    uuid_value = str(uuid_value or "none")
    with _state_lock:
        value = _uuid_lineage.get(uuid_value)
        return None if value is None else dict(value)


def _set_guider_sample(guider, sample):
    with _state_lock:
        try:
            _guider_samples[guider] = sample
        except TypeError:
            _bounded_put(_guider_samples_by_id, id(guider), sample)


def _get_guider_sample(guider):
    with _state_lock:
        try:
            value = _guider_samples.get(guider)
        except TypeError:
            value = None
        if value is None:
            value = _guider_samples_by_id.get(id(guider))
        return value


def _sample_from_records(records):
    for record in records or ():
        lineage = _lookup_uuid(record.get("uuid"))
        if isinstance(lineage, dict):
            sample = lineage.get("sample")
            if isinstance(sample, dict):
                return sample
    return _current_sample.get()


def _lineage_summary(records):
    values = []
    for record in records or ():
        lineage = _lookup_encode_lineage(record.get("raw"))
        if lineage is None:
            values.append(f"{record.get('index')}:cache_or_unobserved")
        else:
            values.append(
                f"{record.get('index')}:encode:{lineage.get('encode_id')},"
                f"tokens:{lineage.get('tokens_fp')},text:{lineage.get('text_fp')},"
                f"caller:{lineage.get('caller')}"
            )
    return "|".join(values) or "none"


def _observe_encode(clip, tokens, result, elapsed_ms):
    if not _is_anima_conditioning(result):
        return
    encode_id = _next_id("encode")
    records = _conditioning_records(result)
    token_fp = _structured_fingerprint(tokens)
    caller = _caller_details()
    patch = _patcher_identity(getattr(clip, "patcher", None))
    output_fp = _record_value_fp(records, "raw")
    output_bad = _records_bad(records, "raw")
    history_key = (f"{id(clip):x}", caller["caller"])
    current = {
        "tokens_fp": token_fp,
        "text_fp": caller["text_fp"],
        "output_fp": output_fp,
        "patch": tuple(patch.items()),
    }
    with _state_lock:
        previous = _encode_history.get(history_key)
        _bounded_put(_encode_history, history_key, current)

    transition = "first_observation"
    suspect = "none"
    if isinstance(previous, dict):
        tokens_changed = previous.get("tokens_fp") != token_fp
        output_same = previous.get("output_fp") == output_fp and output_fp != "none"
        patch_same = previous.get("patch") == current["patch"]
        if tokens_changed and output_same and patch_same:
            transition = "tokens_changed_output_same_suspect"
            suspect = "stale_text_encode_output"
        elif not tokens_changed and not output_same and patch_same:
            transition = "tokens_same_output_changed_suspect"
            suspect = "unstable_text_encode_output"
        elif tokens_changed:
            transition = "tokens_changed_output_changed"
        elif output_same:
            transition = "tokens_same_output_same"
        else:
            transition = "tokens_same_output_changed_after_patch"

    for record in records:
        _register_encode_lineage(record.get("raw"), {
            "encode_id": encode_id,
            "tokens_fp": token_fp,
            "text_fp": caller["text_fp"],
            "caller": caller["caller"],
            "clip_patch": patch,
        })

    level = logging.INFO
    if output_bad not in (None, 0):
        level = logging.ERROR
        suspect = "nonfinite_text_encode_output"
    elif suspect != "none":
        level = logging.WARNING
    _log_event(
        "text_encode",
        level=level,
        encode=encode_id,
        caller=caller["caller"],
        text_fp=caller["text_fp"],
        text_chars=caller["text_chars"],
        tokens_fp=token_fp,
        transition=transition,
        suspect=suspect,
        clip=f"{id(clip):x}",
        clip_patch=_patch_summary(patch),
        elapsed_ms=f"{elapsed_ms:.3f}",
        output_fp=output_fp,
        output_bad=output_bad,
        entries=_records_summary(records),
    )


def _observe_guider_source(guider, conds, sample):
    model_patch = _patcher_identity(getattr(guider, "model_patcher", None))
    sample["model_patch_at_source"] = model_patch
    for role, conditioning in conds.items():
        records = _conditioning_records(conditioning)
        sample.setdefault("roles", {}).setdefault(str(role), {})["source"] = records
        bad = _records_bad(records, "raw")
        _log_event(
            "guider_source",
            level=logging.ERROR if bad not in (None, 0) else logging.INFO,
            sample=sample["sample_id"],
            role=role,
            raw_fp=_record_value_fp(records, "raw"),
            raw_bad=bad,
            lineage=_lineage_summary(records),
            model_patch=_patch_summary(model_patch),
            entries=_records_summary(records),
        )


def _observe_guider_converted(guider, sample):
    original_conds = getattr(guider, "original_conds", {})
    if not isinstance(original_conds, dict):
        return
    for role, conditioning in original_conds.items():
        records = _conditioning_records(conditioning)
        sample.setdefault("roles", {}).setdefault(str(role), {})["converted"] = records
        for record in records:
            _register_uuid(record.get("uuid"), {
                "sample_id": sample["sample_id"],
                "sample": sample,
                "role": str(role),
                "source": record,
            })
        _log_event(
            "guider_converted",
            sample=sample["sample_id"],
            role=role,
            raw_fp=_record_value_fp(records, "raw"),
            raw_bad=_records_bad(records, "raw"),
            entries=_records_summary(records),
        )


def _model_cond_transition(model_key, role, records_in, records_out, patch):
    dependencies = {
        "raw": _record_value_fp(records_in, "raw"),
        "ids": _record_value_fp(records_in, "ids"),
        "weights": _record_value_fp(records_in, "weights"),
        "patch": tuple(patch.items()),
    }
    output_fp = _record_value_fp(records_out, "model")
    current = {"dependencies": dependencies, "output_fp": output_fp}
    key = (str(model_key), str(role))
    with _state_lock:
        previous = _model_cond_history.get(key)
        _bounded_put(_model_cond_history, key, current)
    if not isinstance(previous, dict):
        return "first_observation", (), "none"

    old_dependencies = previous.get("dependencies") or {}
    changes = tuple(
        name
        for name in ("raw", "ids", "weights", "patch")
        if old_dependencies.get(name) != dependencies.get(name)
    )
    output_same = previous.get("output_fp") == output_fp and output_fp != "none"
    value_changes = tuple(name for name in changes if name != "patch")
    if value_changes and output_same:
        return (
            "inputs_changed_output_same_suspect",
            changes,
            "stale_preprocess_text_embeds_output",
        )
    if not changes and not output_same:
        return (
            "inputs_same_output_changed_suspect",
            changes,
            "unstable_preprocess_text_embeds_output",
        )
    if changes and output_same:
        return "patch_changed_output_same", changes, "none"
    if changes:
        return "inputs_changed_output_changed", changes, "none"
    return "inputs_same_output_same", changes, "none"


def _observe_model_conds(model_function, role, before, after):
    model, patch = _bound_model_and_patch(model_function)
    sample = _sample_from_records(before)
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else "unscoped"
    model_key = f"{id(model):x}" if model is not None else "none"
    transition, changes, suspect = _model_cond_transition(
        model_key,
        role,
        before,
        after,
        patch,
    )
    provenances = []
    for index, output_record in enumerate(after):
        input_record = before[index] if index < len(before) else {}
        input_bad = _snapshot_bad(input_record.get("raw"))
        output_bad = _snapshot_bad(output_record.get("model"))
        if input_bad not in (None, 0):
            provenance = "nonfinite_before_model_preprocess"
        elif output_bad not in (None, 0):
            provenance = "became_nonfinite_in_preprocess_text_embeds"
        else:
            provenance = "finite_through_model_preprocess"
        provenances.append(f"{index}:{provenance}")
        uuid_value = output_record.get("uuid")
        _register_uuid(uuid_value, {
            "sample_id": sample_id,
            "sample": sample,
            "role": str(role),
            "source": input_record,
            "model_cond": output_record,
            "model_patch": patch,
        })

    if isinstance(sample, dict):
        sample.setdefault("roles", {}).setdefault(str(role), {})["model_cond"] = after
    input_bad = _records_bad(before, "raw")
    output_bad = _records_bad(after, "model")
    level = logging.INFO
    if input_bad not in (None, 0) or output_bad not in (None, 0):
        level = logging.ERROR
    elif suspect != "none":
        level = logging.WARNING
    _log_event(
        "model_cond",
        level=level,
        sample=sample_id,
        role=role,
        model=model_key,
        model_patch=_patch_summary(patch),
        transition=transition,
        changed=",".join(changes) or "none",
        suspect=suspect,
        provenance="|".join(provenances) or "none",
        input_raw_fp=_record_value_fp(before, "raw"),
        input_raw_bad=input_bad,
        output_fp=_record_value_fp(after, "model"),
        output_bad=output_bad,
        lineage=_lineage_summary(before),
        before=_records_summary(before),
        after=_records_summary(after),
    )


def _observe_process_cond(conds, result, lineage):
    uuid_value = str(conds.get("uuid", "none"))
    model_conds = conds.get("model_conds") or {}
    before_value = model_conds.get("c_crossattn") if isinstance(model_conds, dict) else None
    after_value = None
    if result is not None:
        conditioning = getattr(result, "conditioning", None)
        if isinstance(conditioning, dict):
            after_value = conditioning.get("c_crossattn")
    before = _tensor_snapshot(before_value)
    after = _tensor_snapshot(after_value)
    before_bad = _snapshot_bad(before)
    after_bad = _snapshot_bad(after)
    if before_bad not in (None, 0):
        provenance = "nonfinite_before_process_cond"
    elif after_bad not in (None, 0):
        provenance = "became_nonfinite_in_process_cond"
    else:
        provenance = "finite_through_process_cond"
    _log_event(
        "process_cond",
        level=(
            logging.ERROR
            if before_bad not in (None, 0) or after_bad not in (None, 0)
            else logging.INFO
        ),
        sample=lineage.get("sample_id", "unscoped"),
        role=lineage.get("role", "unknown"),
        uuid=uuid_value,
        provenance=provenance,
        before=f"{{{_snapshot_summary(before)}}}",
        after=f"{{{_snapshot_summary(after)}}}",
    )


def _caller_batch_metadata():
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        for _depth in range(24):
            if frame is None:
                break
            local_values = frame.f_locals
            markers = local_values.get("cond_or_uncond")
            uuids = local_values.get("uuids")
            if isinstance(markers, (list, tuple)) or isinstance(
                uuids,
                (list, tuple),
            ):
                return (
                    tuple(markers)
                    if isinstance(markers, (list, tuple))
                    else (),
                    tuple(uuids)
                    if isinstance(uuids, (list, tuple))
                    else (),
                )
            frame = frame.f_back
    finally:
        del frame
    return (), ()


def _observe_cond_cat(c_list, result, markers, uuids, device):
    input_records = []
    known = []
    for index, conditioning in enumerate(c_list):
        value = (
            conditioning.get("c_crossattn")
            if isinstance(conditioning, dict)
            else None
        )
        snapshot = _tensor_snapshot(value)
        uuid_value = str(uuids[index]) if index < len(uuids) else "none"
        marker = str(markers[index]) if index < len(markers) else "unknown"
        lineage = _lookup_uuid(uuid_value) or {}
        known.append(lineage)
        input_records.append({
            "index": index,
            "marker": marker,
            "uuid": uuid_value,
            "role": lineage.get("role", "unknown"),
            "snapshot": snapshot,
        })
    output_value = result.get("c_crossattn") if isinstance(result, dict) else None
    output = _tensor_snapshot(output_value)
    input_bad_values = [
        _snapshot_bad(record["snapshot"])
        for record in input_records
    ]
    input_bad = (
        None
        if not input_bad_values or any(value is None for value in input_bad_values)
        else sum(input_bad_values)
    )
    output_bad = _snapshot_bad(output)
    if input_bad not in (None, 0):
        provenance = "nonfinite_before_cond_cat"
    elif output_bad not in (None, 0):
        provenance = "became_nonfinite_in_cond_cat_or_device_move"
    else:
        provenance = "finite_through_cond_cat"
    sample_ids = {
        value.get("sample_id")
        for value in known
        if isinstance(value, dict) and value.get("sample_id")
    }
    sample_id = ",".join(sorted(str(value) for value in sample_ids)) or "unscoped"
    input_summary = "|".join(
        f"{record['index']}[marker:{record['marker']};role:{record['role']};"
        f"uuid:{record['uuid']};tensor:{{{_snapshot_summary(record['snapshot'])}}}]"
        for record in input_records
    ) or "none"
    _log_event(
        "cond_cat",
        level=(
            logging.ERROR
            if input_bad not in (None, 0) or output_bad not in (None, 0)
            else logging.INFO
        ),
        sample=sample_id,
        device=device,
        markers=markers or "none",
        uuids=tuple(str(value) for value in uuids) or "none",
        provenance=provenance,
        input_bad=input_bad,
        output_bad=output_bad,
        inputs=input_summary,
        output=f"{{{_snapshot_summary(output)}}}",
    )


def _sample_for_cond_lists(conds):
    if isinstance(conds, (list, tuple)):
        for conditioning in conds:
            if not isinstance(conditioning, (list, tuple)):
                continue
            for item in conditioning:
                if not isinstance(item, dict):
                    continue
                lineage = _lookup_uuid(item.get("uuid"))
                if isinstance(lineage, dict) and isinstance(lineage.get("sample"), dict):
                    return lineage["sample"]
    return _current_sample.get()


def _observe_denoise(sample, model, input_snapshot, timestep_snapshot, result):
    outputs = tuple(_tensor_snapshot(value) for value in result or ())
    labels = ("positive", "negative")
    output_summary = "|".join(
        f"{labels[index] if index < len(labels) else index}:"
        f"{{{_snapshot_summary(snapshot)}}}"
        for index, snapshot in enumerate(outputs)
    ) or "none"
    output_bad_values = [_snapshot_bad(value) for value in outputs]
    output_bad = (
        None
        if not output_bad_values or any(value is None for value in output_bad_values)
        else sum(output_bad_values)
    )
    patch = _patcher_identity(getattr(model, "current_patcher", None))
    _log_event(
        "denoise_batch",
        level=logging.ERROR if output_bad not in (None, 0) else logging.INFO,
        sample=sample.get("sample_id", "unscoped"),
        model_patch=_patch_summary(patch),
        input=f"{{{_snapshot_summary(input_snapshot)}}}",
        timestep=f"{{{_snapshot_summary(timestep_snapshot)}}}",
        output_bad=output_bad,
        outputs=output_summary,
    )


def _observe_cfg_output(sample, result):
    snapshot = _tensor_snapshot(result)
    bad = _snapshot_bad(snapshot)
    _log_event(
        "cfg_output",
        level=logging.ERROR if bad not in (None, 0) else logging.INFO,
        sample=sample.get("sample_id", "unscoped"),
        output=f"{{{_snapshot_summary(snapshot)}}}",
    )


def _model_cond_context_fields():
    context = _current_model_cond.get()
    if not isinstance(context, dict):
        return {
            "sample_id": "unscoped",
            "role": "unknown",
            "model": "none",
            "model_patch": _patch_summary(None),
        }
    return {
        "sample_id": context.get("sample_id", "unscoped"),
        "role": context.get("role", "unknown"),
        "model": context.get("model", "none"),
        "model_patch": _patch_summary(context.get("patch")),
    }


def _adapter_stage_modules(adapter):
    candidates = [
        ("embed", getattr(adapter, "embed", None)),
        ("in_proj", getattr(adapter, "in_proj", None)),
    ]
    blocks = getattr(adapter, "blocks", ())
    try:
        candidates.extend(
            (f"block_{index}", block)
            for index, block in enumerate(blocks)
        )
    except Exception as error:
        _report_diagnostic_error("llm_adapter_blocks", error)
    candidates.extend((
        ("out_proj", getattr(adapter, "out_proj", None)),
        ("norm", getattr(adapter, "norm", None)),
    ))

    seen = set()
    for name, module in candidates:
        if module is None or not hasattr(module, "register_forward_hook"):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield name, module


def _install_adapter_stage_hooks(adapter, stages, handles):
    for name, module in _adapter_stage_modules(adapter):
        def observe_stage(_module, _inputs, output, stage=name):
            snapshot = _safe_observe(
                f"llm_adapter_stage_{stage}",
                _tensor_snapshot,
                output,
            )
            if snapshot is not None:
                stages.append((stage, snapshot))

        try:
            handles.append(module.register_forward_hook(observe_stage))
        except Exception as error:
            _report_diagnostic_error(f"llm_adapter_hook_{name}", error)


def _remove_adapter_stage_hooks(handles):
    for handle in reversed(handles):
        try:
            handle.remove()
        except Exception as error:
            _report_diagnostic_error("llm_adapter_hook_remove", error)


def _adapter_stage_summary(stages):
    return "|".join(
        f"{name}:{{{_snapshot_summary(snapshot)}}}"
        for name, snapshot in stages
    ) or "none"


def _observe_llm_adapter(
    adapter,
    source_hidden_states,
    target_input_ids,
    result,
    stages,
    elapsed_ms,
    error=None,
):
    preprocess = _current_preprocess.get()
    preprocess = preprocess if isinstance(preprocess, dict) else {}
    fields = _model_cond_context_fields()
    source = _tensor_snapshot(source_hidden_states)
    ids = _tensor_snapshot(target_input_ids)
    output = _tensor_snapshot(result)
    source_bad = _snapshot_bad(source)
    ids_bad = _snapshot_bad(ids)
    output_bad = _snapshot_bad(output)
    first_bad_stage = next(
        (
            name
            for name, snapshot in stages
            if _snapshot_bad(snapshot) not in (None, 0)
        ),
        None,
    )
    if source_bad not in (None, 0) or ids_bad not in (None, 0):
        provenance = "nonfinite_before_llm_adapter"
    elif first_bad_stage is not None:
        provenance = f"became_nonfinite_at_{first_bad_stage}"
    elif output_bad not in (None, 0):
        provenance = "became_nonfinite_in_uninstrumented_adapter_op"
    elif error is not None:
        provenance = "llm_adapter_raised"
    else:
        provenance = "finite_through_llm_adapter"
    level = logging.INFO
    if error is not None or any(
        value not in (None, 0)
        for value in (source_bad, ids_bad, output_bad)
    ):
        level = logging.ERROR
    _log_event(
        "llm_adapter",
        level=level,
        preprocess=preprocess.get("preprocess_id", "unscoped"),
        sample=fields["sample_id"],
        role=fields["role"],
        model=fields["model"],
        model_patch=fields["model_patch"],
        adapter=f"{id(adapter):x}",
        outcome="exception" if error is not None else "complete",
        error_type=type(error).__name__ if error is not None else "none",
        provenance=provenance,
        first_bad_stage=first_bad_stage or "none",
        elapsed_ms=f"{elapsed_ms:.3f}",
        source=f"{{{_snapshot_summary(source)}}}",
        ids=f"{{{_snapshot_summary(ids)}}}",
        output=f"{{{_snapshot_summary(output)}}}",
        stages=_adapter_stage_summary(stages),
    )


def _observe_preprocess(
    owner,
    preprocess,
    text_embeds,
    text_ids,
    weights,
    result,
    elapsed_ms,
    error=None,
):
    fields = _model_cond_context_fields()
    embeds = _tensor_snapshot(text_embeds)
    ids = _tensor_snapshot(text_ids)
    weight_snapshot = _tensor_snapshot(weights)
    output = _tensor_snapshot(result)
    input_bad_values = (
        _snapshot_bad(embeds),
        _snapshot_bad(ids),
        _snapshot_bad(weight_snapshot),
    )
    output_bad = _snapshot_bad(output)
    if any(value not in (None, 0) for value in input_bad_values):
        provenance = "nonfinite_before_preprocess_text_embeds"
    elif output_bad not in (None, 0):
        provenance = "became_nonfinite_in_preprocess_text_embeds"
    elif error is not None:
        provenance = "preprocess_text_embeds_raised"
    else:
        provenance = "finite_through_preprocess_text_embeds"
    level = logging.INFO
    if error is not None or output_bad not in (None, 0) or any(
        value not in (None, 0) for value in input_bad_values
    ):
        level = logging.ERROR
    _log_event(
        "preprocess_text_embeds",
        level=level,
        preprocess=preprocess.get("preprocess_id", "unscoped"),
        sample=fields["sample_id"],
        role=fields["role"],
        model=fields["model"],
        model_patch=fields["model_patch"],
        diffusion_model=f"{id(owner):x}",
        adapter=f"{id(getattr(owner, 'llm_adapter', None)):x}",
        outcome="exception" if error is not None else "complete",
        error_type=type(error).__name__ if error is not None else "none",
        provenance=provenance,
        elapsed_ms=f"{elapsed_ms:.3f}",
        embeds=f"{{{_snapshot_summary(embeds)}}}",
        ids=f"{{{_snapshot_summary(ids)}}}",
        weights=f"{{{_snapshot_summary(weight_snapshot)}}}",
        output=f"{{{_snapshot_summary(output)}}}",
    )


def _make_llm_adapter_wrapper(original):
    @functools.wraps(original)
    def wrapped(self, source_hidden_states, target_input_ids, *args, **kwargs):
        if not is_enabled():
            return original(
                self,
                source_hidden_states,
                target_input_ids,
                *args,
                **kwargs,
            )
        stages = []
        handles = []
        _safe_observe(
            "llm_adapter_stage_hooks",
            _install_adapter_stage_hooks,
            self,
            stages,
            handles,
        )
        started = time.perf_counter()
        try:
            result = original(
                self,
                source_hidden_states,
                target_input_ids,
                *args,
                **kwargs,
            )
        except BaseException as error:
            _safe_observe(
                "llm_adapter_hook_remove",
                _remove_adapter_stage_hooks,
                handles,
            )
            _safe_observe(
                "llm_adapter",
                _observe_llm_adapter,
                self,
                source_hidden_states,
                target_input_ids,
                None,
                stages,
                (time.perf_counter() - started) * 1000.0,
                error,
            )
            raise
        _safe_observe(
            "llm_adapter_hook_remove",
            _remove_adapter_stage_hooks,
            handles,
        )
        _safe_observe(
            "llm_adapter",
            _observe_llm_adapter,
            self,
            source_hidden_states,
            target_input_ids,
            result,
            stages,
            (time.perf_counter() - started) * 1000.0,
        )
        return result
    return wrapped


def _make_preprocess_text_embeds_wrapper(original):
    @functools.wraps(original)
    def wrapped(self, text_embeds, text_ids, *args, **kwargs):
        if not is_enabled():
            return original(self, text_embeds, text_ids, *args, **kwargs)
        weights = (
            args[0]
            if args
            else kwargs.get("t5xxl_weights")
        )
        preprocess = {
            "preprocess_id": _next_id("preprocess"),
            **_model_cond_context_fields(),
        }
        token = _current_preprocess.set(preprocess)
        started = time.perf_counter()
        try:
            result = original(self, text_embeds, text_ids, *args, **kwargs)
        except BaseException as error:
            _safe_observe(
                "preprocess_text_embeds",
                _observe_preprocess,
                self,
                preprocess,
                text_embeds,
                text_ids,
                weights,
                None,
                (time.perf_counter() - started) * 1000.0,
                error,
            )
            raise
        else:
            _safe_observe(
                "preprocess_text_embeds",
                _observe_preprocess,
                self,
                preprocess,
                text_embeds,
                text_ids,
                weights,
                result,
                (time.perf_counter() - started) * 1000.0,
            )
            return result
        finally:
            _current_preprocess.reset(token)
    return wrapped


def _make_clip_encode_wrapper(original):
    @functools.wraps(original)
    def wrapped(self, tokens, *args, **kwargs):
        if not is_enabled() or not _is_anima_tokens(tokens):
            return original(self, tokens, *args, **kwargs)
        started = time.perf_counter()
        try:
            result = original(self, tokens, *args, **kwargs)
        except BaseException as error:
            _safe_observe(
                "text_encode_exception",
                _log_event,
                "text_encode_exception",
                level=logging.ERROR,
                caller=_caller_details().get("caller"),
                tokens_fp=_safe_observe(
                    "tokens_fp",
                    _structured_fingerprint,
                    tokens,
                ) or "unavailable",
                error_type=type(error).__name__,
            )
            raise
        _safe_observe(
            "text_encode",
            _observe_encode,
            self,
            tokens,
            result,
            (time.perf_counter() - started) * 1000.0,
        )
        return result
    return wrapped


def _make_inner_set_conds_wrapper(original):
    @functools.wraps(original)
    def wrapped(self, conds, *args, **kwargs):
        if (
            not is_enabled()
            or not isinstance(conds, dict)
            or not any(_is_anima_conditioning(value) for value in conds.values())
        ):
            return original(self, conds, *args, **kwargs)
        sample = {
            "sample_id": _next_id("sample"),
            "roles": {},
            "seen_process": set(),
            "seen_cat": set(),
            "denoise_observed": False,
            "cfg_observed": False,
        }
        _safe_observe("guider_source", _observe_guider_source, self, conds, sample)
        try:
            result = original(self, conds, *args, **kwargs)
        except BaseException as error:
            _safe_observe(
                "guider_convert_exception",
                _log_event,
                "guider_convert_exception",
                level=logging.ERROR,
                sample=sample["sample_id"],
                error_type=type(error).__name__,
            )
            raise
        _safe_observe("guider_converted", _observe_guider_converted, self, sample)
        _set_guider_sample(self, sample)
        return result
    return wrapped


def _shape_only(value):
    tensor = _unwrap_tensor(value)
    if tensor is None:
        return "none"
    return f"{tuple(tensor.shape)}:{tensor.dtype}:{tensor.device}"


def _make_guider_sample_wrapper(original):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        sample = _get_guider_sample(self)
        if not is_enabled() or not isinstance(sample, dict):
            return original(self, *args, **kwargs)
        token = _current_sample.set(sample)
        started = time.perf_counter()
        noise = args[0] if len(args) > 0 else kwargs.get("noise")
        latent = args[1] if len(args) > 1 else kwargs.get("latent_image")
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        _safe_observe(
            "sample_start",
            _log_event,
            "sample_start",
            sample=sample["sample_id"],
            model_patch=_patch_summary(
                _patcher_identity(getattr(self, "model_patcher", None))
            ),
            cfg=getattr(self, "cfg", "unknown"),
            noise=_shape_only(noise),
            latent=_shape_only(latent),
            sigmas=_shape_only(sigmas),
        )
        try:
            result = original(self, *args, **kwargs)
        except BaseException as error:
            _safe_observe(
                "sample_end",
                _log_event,
                "sample_end",
                level=logging.WARNING,
                sample=sample["sample_id"],
                outcome="exception",
                error_type=type(error).__name__,
                elapsed_ms=f"{(time.perf_counter() - started) * 1000.0:.3f}",
            )
            raise
        else:
            _safe_observe(
                "sample_end",
                _log_event,
                "sample_end",
                sample=sample["sample_id"],
                outcome="complete",
                elapsed_ms=f"{(time.perf_counter() - started) * 1000.0:.3f}",
            )
            return result
        finally:
            _current_sample.reset(token)
    return wrapped


def _make_encode_model_conds_wrapper(original):
    @functools.wraps(original)
    def wrapped(model_function, conds, noise, device, prompt_type, **kwargs):
        model = getattr(model_function, "__self__", None)
        observed = (
            is_enabled()
            and (
                _is_anima_conditioning(conds)
                or _is_anima_model(model)
            )
        )
        before = (
            _safe_observe("model_cond_before", _conditioning_records, conds)
            if observed else None
        )
        context_token = None
        if observed:
            sample = _sample_from_records(before or ())
            patch = _patcher_identity(getattr(model, "current_patcher", None))
            context_token = _current_model_cond.set({
                "sample_id": (
                    sample.get("sample_id", "unscoped")
                    if isinstance(sample, dict)
                    else "unscoped"
                ),
                "role": str(prompt_type),
                "model": f"{id(model):x}" if model is not None else "none",
                "patch": patch,
            })
        try:
            result = original(
                model_function,
                conds,
                noise,
                device,
                prompt_type,
                **kwargs,
            )
        except BaseException as error:
            if observed:
                _safe_observe(
                    "model_cond_exception",
                    _log_event,
                    "model_cond_exception",
                    level=logging.ERROR,
                    sample=(
                        (_sample_from_records(before or ()) or {}).get(
                            "sample_id",
                            "unscoped",
                        )
                    ),
                    role=prompt_type,
                    error_type=type(error).__name__,
                    before=_records_summary(before or ()),
                )
            raise
        finally:
            if context_token is not None:
                _current_model_cond.reset(context_token)
        if observed and before is not None:
            after = _safe_observe(
                "model_cond_after",
                _conditioning_records,
                result,
            )
            if after is not None:
                _safe_observe(
                    "model_cond",
                    _observe_model_conds,
                    model_function,
                    prompt_type,
                    before,
                    after,
                )
        return result
    return wrapped


def _make_get_area_wrapper(original):
    @functools.wraps(original)
    def wrapped(conds, *args, **kwargs):
        lineage = (
            _lookup_uuid(conds.get("uuid"))
            if is_enabled() and isinstance(conds, dict)
            else None
        )
        trace = False
        seen_key = None
        sample = lineage.get("sample") if isinstance(lineage, dict) else None
        if isinstance(lineage, dict):
            model_conds = conds.get("model_conds") or {}
            value = model_conds.get("c_crossattn") if isinstance(model_conds, dict) else None
            seen_key = (str(conds.get("uuid")), _value_identity(value))
            with _state_lock:
                seen = sample.get("seen_process", set()) if isinstance(sample, dict) else set()
                trace = seen_key not in seen
        try:
            result = original(conds, *args, **kwargs)
        except BaseException as error:
            if isinstance(lineage, dict):
                _safe_observe(
                    "process_cond_exception",
                    _log_event,
                    "process_cond_exception",
                    level=logging.ERROR,
                    sample=lineage.get("sample_id", "unscoped"),
                    role=lineage.get("role", "unknown"),
                    uuid=conds.get("uuid"),
                    error_type=type(error).__name__,
                )
            raise
        if trace:
            _safe_observe("process_cond", _observe_process_cond, conds, result, lineage)
            if isinstance(sample, dict):
                with _state_lock:
                    sample.setdefault("seen_process", set()).add(seen_key)
        return result
    return wrapped


def _make_cond_cat_wrapper(original):
    @functools.wraps(original)
    def wrapped(c_list, device=None):
        markers, uuids = _safe_observe(
            "cond_cat_metadata",
            _caller_batch_metadata,
        ) or ((), ())
        lineages = [_lookup_uuid(value) for value in uuids]
        known = is_enabled() and any(isinstance(value, dict) for value in lineages)
        sample = next(
            (
                value.get("sample")
                for value in lineages
                if isinstance(value, dict) and isinstance(value.get("sample"), dict)
            ),
            None,
        )
        identities = tuple(
            _value_identity(
                value.get("c_crossattn") if isinstance(value, dict) else None
            )
            for value in c_list
        ) if known else ()
        seen_key = (
            tuple(str(value) for value in uuids),
            tuple(str(value) for value in markers),
            str(device),
            identities,
        )
        with _state_lock:
            seen = sample.get("seen_cat", set()) if isinstance(sample, dict) else set()
            trace = known and seen_key not in seen
        try:
            result = original(c_list, device=device)
        except BaseException as error:
            if known:
                _safe_observe(
                    "cond_cat_exception",
                    _log_event,
                    "cond_cat_exception",
                    level=logging.ERROR,
                    sample=(sample or {}).get("sample_id", "unscoped"),
                    markers=markers,
                    uuids=uuids,
                    error_type=type(error).__name__,
                )
            raise
        if trace:
            _safe_observe(
                "cond_cat",
                _observe_cond_cat,
                c_list,
                result,
                markers,
                uuids,
                device,
            )
            if isinstance(sample, dict):
                with _state_lock:
                    sample.setdefault("seen_cat", set()).add(seen_key)
        return result
    return wrapped


def _make_calc_cond_batch_wrapper(original):
    @functools.wraps(original)
    def wrapped(model, conds, x_in, timestep, model_options):
        sample = _sample_for_cond_lists(conds) if is_enabled() else None
        trace = isinstance(sample, dict) and not sample.get("denoise_observed", False)
        input_snapshot = (
            _safe_observe("denoise_input", _tensor_snapshot, x_in)
            if trace else None
        )
        timestep_snapshot = (
            _safe_observe("denoise_timestep", _tensor_snapshot, timestep)
            if trace else None
        )
        try:
            result = original(model, conds, x_in, timestep, model_options)
        except BaseException as error:
            if trace:
                _safe_observe(
                    "denoise_exception",
                    _log_event,
                    "denoise_exception",
                    level=logging.ERROR,
                    sample=sample.get("sample_id", "unscoped"),
                    error_type=type(error).__name__,
                    input=f"{{{_snapshot_summary(input_snapshot)}}}",
                    timestep=f"{{{_snapshot_summary(timestep_snapshot)}}}",
                )
            raise
        if trace:
            _safe_observe(
                "denoise_batch",
                _observe_denoise,
                sample,
                model,
                input_snapshot,
                timestep_snapshot,
                result,
            )
            sample["denoise_observed"] = True
        return result
    return wrapped


def _make_sampling_function_wrapper(original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        sample = _current_sample.get() if is_enabled() else None
        trace = isinstance(sample, dict) and not sample.get("cfg_observed", False)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            if trace:
                _safe_observe(
                    "cfg_exception",
                    _log_event,
                    "cfg_exception",
                    level=logging.ERROR,
                    sample=sample.get("sample_id", "unscoped"),
                    error_type=type(error).__name__,
                )
            raise
        if trace:
            _safe_observe("cfg_output", _observe_cfg_output, sample, result)
            sample["cfg_observed"] = True
        return result
    return wrapped


def _patch(owner, name, factory):
    current = getattr(owner, name)
    if getattr(current, _WRAPPER_MARKER, False):
        return False
    wrapped = factory(current)
    setattr(wrapped, _WRAPPER_MARKER, True)
    setattr(wrapped, _ORIGINAL_MARKER, current)
    setattr(owner, name, wrapped)
    _installed_patches.append((owner, name, current, wrapped))
    return True


def install(comfy_sd=None, samplers=None, anima_model=None):
    """Install every probe atomically; return ``True`` when active."""
    if not is_enabled():
        return False
    with _install_lock:
        if _installed_patches:
            return True
        try:
            if comfy_sd is None:
                import comfy.sd as comfy_sd
            if samplers is None:
                import comfy.samplers as samplers
            if anima_model is None:
                import comfy.ldm.anima.model as anima_model

            installed_names = []
            if _patch(
                anima_model.Anima,
                "preprocess_text_embeds",
                _make_preprocess_text_embeds_wrapper,
            ):
                installed_names.append("Anima.preprocess_text_embeds")
            if _patch(
                anima_model.LLMAdapter,
                "forward",
                _make_llm_adapter_wrapper,
            ):
                installed_names.append("LLMAdapter.forward")
            if _patch(
                comfy_sd.CLIP,
                "encode_from_tokens_scheduled",
                _make_clip_encode_wrapper,
            ):
                installed_names.append("CLIP.encode_from_tokens_scheduled")
            if _patch(
                samplers.CFGGuider,
                "inner_set_conds",
                _make_inner_set_conds_wrapper,
            ):
                installed_names.append("CFGGuider.inner_set_conds")
            if _patch(
                samplers.CFGGuider,
                "sample",
                _make_guider_sample_wrapper,
            ):
                installed_names.append("CFGGuider.sample")
            if _patch(
                samplers,
                "encode_model_conds",
                _make_encode_model_conds_wrapper,
            ):
                installed_names.append("encode_model_conds")
            if _patch(
                samplers,
                "get_area_and_mult",
                _make_get_area_wrapper,
            ):
                installed_names.append("get_area_and_mult")
            if _patch(samplers, "cond_cat", _make_cond_cat_wrapper):
                installed_names.append("cond_cat")
            if _patch(
                samplers,
                "_calc_cond_batch",
                _make_calc_cond_batch_wrapper,
            ):
                installed_names.append("_calc_cond_batch")
            if _patch(
                samplers,
                "sampling_function",
                _make_sampling_function_wrapper,
            ):
                installed_names.append("sampling_function")
        except Exception as error:
            uninstall()
            _report_diagnostic_error("install", error)
            return False

        _log_event(
            "installed",
            hooks=",".join(installed_names) or "already_installed",
            master_env=MASTER_ENV_NAME,
            probe_env=ENV_NAME,
        )
        return True


def uninstall():
    """Restore all original ComfyUI callables installed by this module."""
    restored = []
    with _install_lock:
        while _installed_patches:
            owner, name, original, wrapped = _installed_patches.pop()
            if getattr(owner, name, None) is wrapped:
                setattr(owner, name, original)
                restored.append(name)
        _reset_state()
    if restored:
        _log_event("uninstalled", hooks=",".join(reversed(restored)))
    return tuple(reversed(restored))


def get_install_status():
    with _install_lock:
        return tuple(name for _owner, name, _original, _wrapped in _installed_patches)


def _reset_state():
    with _state_lock:
        _encode_lineage.clear()
        _encode_history.clear()
        _model_cond_history.clear()
        _uuid_lineage.clear()
        _guider_samples.clear()
        _guider_samples_by_id.clear()
    with _error_lock:
        _reported_errors.clear()


def _reset_for_tests():
    uninstall()
    _reset_state()
