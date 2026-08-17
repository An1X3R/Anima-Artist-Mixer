"""Temporary, removable semantic-lineage diagnostics for Adapter Mixer.

Set ``ANIMA_MIXER_SEMANTIC_DIAG=0`` before starting ComfyUI to disable every
hook.  Integration sites are tagged ``TEMP_SEMANTIC_DIAG_HOOK`` so this module
and its callers can be removed mechanically after the intermittent fault is
identified.

The module stores hashes and counters only.  It never retains tensors, changes
model inputs, repairs values, or converts a diagnostic observation into a
sampling failure.  The ComfyUI plugin entrypoint also installs a filtered file
handler that mirrors Anima diagnostics and related processing exceptions to a
timestamped session log under ``E:\\codex logs`` by default.
"""

from collections import OrderedDict
from datetime import datetime
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import threading


logger = logging.getLogger(__name__)

ENV_NAME = "ANIMA_MIXER_SEMANTIC_DIAG"
FILE_LOG_ENV_NAME = "ANIMA_MIXER_DIAG_FILE_LOG"
LOG_DIR_ENV_NAME = "ANIMA_MIXER_DIAG_LOG_DIR"
DEFAULT_LOG_DIR = r"E:\codex logs"
_FALSE_VALUES = {"0", "false", "off", "no", "disable", "disabled"}
_HISTORY_LIMIT = 64
_history_lock = threading.Lock()
_file_log_lock = threading.Lock()
_last_context_by_model = OrderedDict()
_session_log_path = None
_FILE_HANDLER_MARKER = "_anima_mixer_diag_file_handler"
_RELEVANT_LOG_MARKERS = (
    "[Anima",
    "Processing interrupted",
    "!!! Exception during processing !!!",
    "Traceback (most recent call last)",
)

_METADATA_KEYS = (
    "pack_id",
    "request_fp",
    "base_prompt_fp",
    "artist_labels_fp",
    "artist_weights_fp",
    "artist_spec_fp",
    "mix_config_fp",
    "alignment_mode",
    "effective_weights",
    "strength",
    "prompt_chars",
    "artist_count",
    "initial_inputs",
    "encoded_inputs",
)


def is_enabled():
    value = os.environ.get(ENV_NAME, "1")
    return str(value).strip().lower() not in _FALSE_VALUES


def is_file_logging_enabled():
    value = os.environ.get(FILE_LOG_ENV_NAME, "1")
    return str(value).strip().lower() not in _FALSE_VALUES


class _RelevantComfyLogFilter(logging.Filter):
    def filter(self, record):
        logger_name = str(getattr(record, "name", "")).lower()
        if "anima_mixer" in logger_name:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return False
        return any(marker in message for marker in _RELEVANT_LOG_MARKERS)


def setup_comfy_file_logging(root_logger=None):
    """Mirror relevant ComfyUI/Anima records to a timestamped session file."""
    global _session_log_path

    if not is_file_logging_enabled():
        return None
    target_logger = root_logger or logging.getLogger()
    with _file_log_lock:
        for handler in tuple(getattr(target_logger, "handlers", ())):
            if getattr(handler, _FILE_HANDLER_MARKER, False):
                path = getattr(handler, "baseFilename", None)
                _session_log_path = None if path is None else str(path)
                return _session_log_path

        configured_dir = os.environ.get(LOG_DIR_ENV_NAME, DEFAULT_LOG_DIR)
        try:
            log_dir = Path(configured_dir).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / (
                f"AnimaMixer_ComfyUI_{timestamp}_pid{os.getpid()}.log"
            )
            handler = logging.FileHandler(
                log_path,
                mode="a",
                encoding="utf-8",
                delay=False,
            )
            handler.setLevel(logging.DEBUG)
            handler.addFilter(_RelevantComfyLogFilter())
            handler.setFormatter(logging.Formatter(
                fmt=(
                    "%(asctime)s.%(msecs)03d [%(levelname)s] "
                    "%(name)s: %(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            setattr(handler, _FILE_HANDLER_MARKER, True)
            target_logger.addHandler(handler)
            _session_log_path = str(log_path.resolve())
        except Exception as error:
            logger.warning(
                "[AnimaSemanticDiag] event=file_log_error directory=%s "
                "error=%s",
                configured_dir,
                error,
            )
            return None

        pointer_path = log_dir / "AnimaMixer_ComfyUI_latest.txt"
        try:
            pointer_path.write_text(
                _session_log_path + os.linesep,
                encoding="utf-8",
            )
        except Exception as error:
            logger.warning(
                "[AnimaSemanticDiag] event=file_log_pointer_error path=%s "
                "error=%s",
                pointer_path,
                error,
            )
        target_logger.info(
            "[AnimaSemanticDiag] event=file_log_ready path=%s "
            "latest_pointer=%s",
            _session_log_path,
            pointer_path,
        )
        return _session_log_path


def get_session_log_path():
    return _session_log_path


def _guard(default=None):
    """Keep temporary diagnostics observational even if logging itself fails."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                logger.warning(
                    "[AnimaSemanticDiag] event=diagnostic_error hook=%s "
                    "error=%s",
                    function.__name__,
                    error,
                )
                return default
        return wrapped
    return decorate


def _hash_payload(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=10).hexdigest()


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
    if not isinstance(snapshot, dict):
        return "unknown"
    value = snapshot.get("bad")
    return "unknown" if value is None else str(value)


def _snapshot_shape(snapshot):
    if not isinstance(snapshot, dict):
        return "none"
    shape = snapshot.get("shape")
    return "none" if shape is None else "x".join(str(value) for value in shape)


def _snapshot_values_match(left, right):
    left_key = _snapshot_key(left)
    right_key = _snapshot_key(right)
    return left_key is not None and left_key == right_key


def _snapshot_list_key(snapshots):
    values = tuple(_snapshot_key(item) for item in (snapshots or ()))
    return None if not values or any(value is None for value in values) else values


def _snapshot_list_fp(snapshots):
    key = _snapshot_list_key(snapshots)
    return "none" if key is None else _hash_payload(key)


def _artist_dependency_fp(inputs, effective_weights):
    if not isinstance(inputs, dict):
        return "none"
    raws = tuple(inputs.get("artist_raws") or ())
    ids = tuple(inputs.get("artist_ids") or ())
    t5_weights = tuple(inputs.get("artist_t5_weights") or ())
    weights = tuple(effective_weights or ())
    if not raws or len(raws) != len(weights):
        return "none"
    entries = []
    for index, (raw, weight) in enumerate(zip(raws, weights)):
        raw_key = _snapshot_key(raw)
        if raw_key is None:
            return "none"
        ids_key = _snapshot_key(ids[index]) if index < len(ids) else None
        t5_key = (
            _snapshot_key(t5_weights[index])
            if index < len(t5_weights)
            else None
        )
        entries.append((raw_key, ids_key, t5_key, round(float(weight), 7)))
    return _hash_payload(tuple(sorted(entries, key=repr)))


def _normalize_patch_identity(identity):
    if not isinstance(identity, (list, tuple)):
        return (str(identity), "none", "none")
    values = list(identity[:3])
    while len(values) < 3:
        values.append(None)
    return tuple("none" if value is None else str(value) for value in values)


def _diag_for_state(state):
    if not isinstance(state, dict):
        return None
    diag = state.get("_semantic_diag")
    if not isinstance(diag, dict):
        return None
    if diag.get("owner_id") == id(state):
        return diag

    cloned = {
        key: diag.get(key)
        for key in _METADATA_KEYS
    }
    cloned.update({
        "owner_id": id(state),
        "diag_id": secrets.token_hex(6),
        "active_run": None,
        "last_run": None,
    })
    state["_semantic_diag"] = cloned
    return cloned


def _active_run(state):
    diag = _diag_for_state(state)
    if not isinstance(diag, dict):
        return None
    run = diag.get("active_run")
    if not isinstance(run, dict) or not run.get("active", False):
        return None
    return run


@_guard()
def initialize_state(
    state,
    *,
    pack_id,
    base_prompt,
    labels,
    weights,
    alignment_mode,
    base_raw_snapshot,
    artist_raw_snapshots,
    encoded_raw_snapshots,
    base_ids_snapshot,
    base_weights_snapshot,
    normalize_weights=False,
    strength=1.0,
    apply_to_uncond=False,
    uncond_strength=1.0,
    artist_ids_snapshots=(),
    artist_t5_weights_snapshots=(),
):
    if not is_enabled():
        return

    base_prompt_value = str(base_prompt or "")
    label_values = tuple(str(label) for label in (labels or ()))
    weight_values = tuple(round(float(value), 7) for value in (weights or ()))
    if normalize_weights and weight_values:
        total = sum(abs(value) for value in weight_values)
        if total <= 1e-8:
            effective_weights = tuple(
                round(1.0 / len(weight_values), 7)
                for _value in weight_values
            )
        else:
            effective_weights = tuple(
                round(value / total, 7)
                for value in weight_values
            )
    else:
        effective_weights = weight_values
    base_prompt_fp = _hash_payload({"base_prompt": base_prompt_value})
    artist_labels_fp = _hash_payload({"labels": label_values})
    artist_weights_fp = _hash_payload({
        "weights": effective_weights,
    })
    artist_spec_fp = _hash_payload({
        "labels": label_values,
        "weights": effective_weights,
    })
    mix_config_fp = _hash_payload({
        "alignment": str(alignment_mode),
        "strength": round(float(strength), 7),
        "apply_to_uncond": bool(apply_to_uncond),
        "uncond_strength": round(float(uncond_strength), 7),
    })
    request_fp = _hash_payload({
        "base_prompt_fp": base_prompt_fp,
        "artist_spec_fp": artist_spec_fp,
        "mix_config_fp": mix_config_fp,
    })
    diag = {
        "owner_id": id(state),
        "diag_id": secrets.token_hex(6),
        "pack_id": str(pack_id or "unavailable"),
        "request_fp": request_fp,
        "base_prompt_fp": base_prompt_fp,
        "artist_labels_fp": artist_labels_fp,
        "artist_weights_fp": artist_weights_fp,
        "artist_spec_fp": artist_spec_fp,
        "mix_config_fp": mix_config_fp,
        "alignment_mode": str(alignment_mode),
        "effective_weights": effective_weights,
        "strength": round(float(strength), 7),
        "prompt_chars": len(base_prompt_value),
        "artist_count": len(label_values),
        "initial_inputs": {
            "base_raw": base_raw_snapshot,
            "artist_raws": tuple(artist_raw_snapshots or ()),
            "artist_ids": tuple(artist_ids_snapshots or ()),
            "artist_t5_weights": tuple(artist_t5_weights_snapshots or ()),
            "base_ids": base_ids_snapshot,
            "base_weights": base_weights_snapshot,
        },
        "encoded_inputs": {
            "base_raw": (
                encoded_raw_snapshots.get("base")
                if isinstance(encoded_raw_snapshots, dict)
                else None
            ),
            "artist_raws": tuple(
                encoded_raw_snapshots.get("artists") or ()
                if isinstance(encoded_raw_snapshots, dict)
                else ()
            ),
        },
        "active_run": None,
        "last_run": None,
    }
    state["_semantic_diag"] = diag
    logger.info(
        "[AnimaSemanticDiag] event=state_created diag=%s pack=%s "
        "request_fp=%s base_prompt_fp=%s artist_labels_fp=%s "
        "artist_weights_fp=%s mix_config_fp=%s prompt_chars=%d "
        "artists=%d base_raw_fp=%s "
        "artist_raws_fp=%s artist_ids_fp=%s artist_t5_weights_fp=%s "
        "encoded_base_raw_fp=%s "
        "encoded_artist_raws_fp=%s base_ids_fp=%s",
        diag["diag_id"],
        diag["pack_id"],
        diag["request_fp"],
        diag["base_prompt_fp"],
        diag["artist_labels_fp"],
        diag["artist_weights_fp"],
        diag["mix_config_fp"],
        diag["prompt_chars"],
        diag["artist_count"],
        _snapshot_fp(base_raw_snapshot),
        _snapshot_list_fp(artist_raw_snapshots),
        _snapshot_list_fp(artist_ids_snapshots),
        _snapshot_list_fp(artist_t5_weights_snapshots),
        _snapshot_fp(diag["encoded_inputs"].get("base_raw")),
        _snapshot_list_fp(diag["encoded_inputs"].get("artist_raws")),
        _snapshot_fp(base_ids_snapshot),
    )


@_guard(False)
def needs_run_start(state, execution_index):
    if not is_enabled():
        return False
    diag = _diag_for_state(state)
    if not isinstance(diag, dict):
        return False
    run = diag.get("active_run")
    return not (
        isinstance(run, dict)
        and run.get("active", False)
        and int(run.get("execution_index", -1)) == int(execution_index)
    )


def _input_drift(initial, current):
    drift = []
    for key in ("base_raw", "base_ids", "base_weights"):
        left = initial.get(key)
        right = current.get(key)
        if _snapshot_key(left) is not None and not _snapshot_values_match(left, right):
            drift.append(key)
    for key in ("artist_raws", "artist_ids", "artist_t5_weights"):
        initial_values = _snapshot_list_key(initial.get(key))
        current_values = _snapshot_list_key(current.get(key))
        if initial_values is not None and initial_values != current_values:
            drift.append(key)
    return tuple(drift)


def _format_cache_counts(caches):
    if not caches:
        return "none"
    return ",".join(
        f"{name}:h{counts.get('hit', 0)}/m{counts.get('miss', 0)}"
        for name, counts in sorted(caches.items())
    )


def _format_counts(values):
    if not values:
        return "none"
    return ",".join(
        f"{name}:{count}"
        for name, count in sorted(values.items())
    )


def _emit_run_end(diag, outcome):
    run = diag.get("active_run")
    if not isinstance(run, dict) or not run.get("active", False):
        return
    stages = run.get("stages") or {}
    stage_fps = ",".join(
        f"{name}:{_snapshot_fp(snapshot)}"
        for name, snapshot in sorted(stages.items())
    ) or "none"
    context = run.get("context") or {}
    level = logging.WARNING if outcome == "aborted" else logging.INFO
    logger.log(
        level,
        "[AnimaSemanticDiag] event=run_end run=%s outcome=%s "
        "context_transition=%s stages=%s caches=%s anchor=%s",
        run.get("run_id"),
        outcome,
        context.get("transition", "not_observed"),
        stage_fps,
        _format_cache_counts(run.get("caches")),
        _format_counts(run.get("anchor")),
    )
    history_complete = _snapshot_key(stages.get("mixed_context")) is not None
    if outcome == "aborted" or not history_complete:
        model_key = run.get("model_key")
        current = run.get("history_current")
        previous = run.get("history_previous")
        if model_key != "none" and isinstance(current, dict):
            with _history_lock:
                observed = _last_context_by_model.get(model_key)
                if (
                    isinstance(observed, dict)
                    and observed.get("run_id") == current.get("run_id")
                ):
                    if isinstance(previous, dict):
                        _last_context_by_model[model_key] = previous
                        _last_context_by_model.move_to_end(model_key)
                    else:
                        _last_context_by_model.pop(model_key, None)
    run["active"] = False
    run["outcome"] = outcome
    diag["last_run"] = run
    diag["active_run"] = None


@_guard()
def begin_run(
    state,
    *,
    execution_index,
    patcher_id,
    patch_identity,
    current_inputs,
):
    if not is_enabled():
        return
    diag = _diag_for_state(state)
    if not isinstance(diag, dict):
        return
    existing = diag.get("active_run")
    if (
        isinstance(existing, dict)
        and existing.get("active", False)
        and int(existing.get("execution_index", -1)) == int(execution_index)
    ):
        return
    if isinstance(existing, dict) and existing.get("active", False):
        _emit_run_end(diag, "superseded")

    patch = _normalize_patch_identity(patch_identity)
    shared_model = state.get("_shared_model_ref")
    model_key = f"{id(shared_model):x}" if shared_model is not None else "none"
    run_id = f"{diag['diag_id']}-{int(execution_index)}"
    state_drift = _input_drift(
        diag.get("initial_inputs") or {},
        current_inputs or {},
    )
    encode_to_state_drift = _input_drift(
        diag.get("encoded_inputs") or {},
        diag.get("initial_inputs") or {},
    )
    run = {
        "active": True,
        "run_id": run_id,
        "execution_index": int(execution_index),
        "patcher_id": "none" if patcher_id is None else f"{int(patcher_id):x}",
        "patch_identity": patch,
        "model_key": model_key,
        "current_inputs": dict(current_inputs or {}),
        "state_drift": state_drift,
        "encode_to_state_drift": encode_to_state_drift,
        "context": None,
        "stages": {},
        "caches": {},
        "anchor": {},
        "history_previous": None,
        "history_current": None,
    }
    diag["active_run"] = run
    logger.info(
        "[AnimaSemanticDiag] event=run_start run=%s pack=%s request_fp=%s "
        "base_prompt_fp=%s artist_labels_fp=%s artist_weights_fp=%s "
        "mix_config_fp=%s "
        "model=%s patcher=%s requested=%s loaded=%s patches=%s "
        "base_raw_fp=%s artist_raws_fp=%s artist_ids_fp=%s "
        "artist_t5_weights_fp=%s artist_input_fp=%s base_ids_fp=%s "
        "anchor_q=%s anchor_refresh=%s anchor_keyframes=%s",
        run_id,
        diag.get("pack_id"),
        diag.get("request_fp"),
        diag.get("base_prompt_fp"),
        diag.get("artist_labels_fp"),
        diag.get("artist_weights_fp"),
        diag.get("mix_config_fp"),
        model_key,
        run["patcher_id"],
        patch[0],
        patch[1],
        patch[2],
        _snapshot_fp(run["current_inputs"].get("base_raw")),
        _snapshot_list_fp(run["current_inputs"].get("artist_raws")),
        _snapshot_list_fp(run["current_inputs"].get("artist_ids")),
        _snapshot_list_fp(run["current_inputs"].get("artist_t5_weights")),
        _artist_dependency_fp(
            run["current_inputs"],
            diag.get("effective_weights"),
        ),
        _snapshot_fp(run["current_inputs"].get("base_ids")),
        bool(state.get("artist_anchor_q", False)),
        state.get("anchor_refresh_mode"),
        state.get("anchor_keyframe_mode"),
    )
    if state_drift:
        logger.warning(
            "[AnimaSemanticDiag] event=finite_input_drift run=%s "
            "boundary=state changed=%s",
            run_id,
            ",".join(state_drift),
        )
    if encode_to_state_drift:
        logger.warning(
            "[AnimaSemanticDiag] event=finite_input_drift run=%s "
            "boundary=encode_to_state changed=%s",
            run_id,
            ",".join(encode_to_state_drift),
        )


@_guard(False)
def should_capture_context(state):
    if not is_enabled():
        return False
    run = _active_run(state)
    return isinstance(run, dict) and run.get("context") is None


@_guard(False)
def should_capture_stage(state, stage):
    if not is_enabled():
        return False
    run = _active_run(state)
    return isinstance(run, dict) and str(stage) not in run.get("stages", {})


@_guard()
def record_context(
    state,
    *,
    snapshot,
    context_key,
    cond_or_uncond,
    conditioning_uuids,
):
    if not is_enabled():
        return
    diag = _diag_for_state(state)
    run = _active_run(state)
    if not isinstance(diag, dict) or not isinstance(run, dict):
        return
    if run.get("context") is not None:
        return

    current_inputs = run.get("current_inputs") or {}
    record = {
        "run_id": run.get("run_id"),
        "pack_id": diag.get("pack_id"),
        "request_fp": diag.get("request_fp"),
        "base_prompt_fp": diag.get("base_prompt_fp"),
        "artist_labels_fp": diag.get("artist_labels_fp"),
        "artist_weights_fp": diag.get("artist_weights_fp"),
        "artist_spec_fp": diag.get("artist_spec_fp"),
        "mix_config_fp": diag.get("mix_config_fp"),
        "alignment_mode": diag.get("alignment_mode"),
        "patch_identity": run.get("patch_identity"),
        "base_raw_fp": _snapshot_fp(current_inputs.get("base_raw")),
        "base_ids_fp": _snapshot_fp(current_inputs.get("base_ids")),
        "artist_raws_fp": _snapshot_list_fp(current_inputs.get("artist_raws")),
        "artist_ids_fp": _snapshot_list_fp(current_inputs.get("artist_ids")),
        "artist_t5_weights_fp": _snapshot_list_fp(
            current_inputs.get("artist_t5_weights")
        ),
        "artist_input_fp": _artist_dependency_fp(
            current_inputs,
            diag.get("effective_weights"),
        ),
        "strength": diag.get("strength"),
        "context_fp": _snapshot_fp(snapshot),
        "context_snapshot": snapshot,
        "context_key": str(context_key),
        "stages": {"base_context": snapshot},
    }
    previous = None
    model_key = run.get("model_key")
    if model_key != "none":
        with _history_lock:
            previous = _last_context_by_model.get(model_key)

    transition = "first_observation"
    suspects = []
    changes = []
    if isinstance(previous, dict):
        tracked_changes = (
            ("base_prompt", "base_prompt_fp"),
            ("artist_labels", "artist_labels_fp"),
            ("artist_weights", "artist_weights_fp"),
            ("mix_config", "mix_config_fp"),
            ("patch", "patch_identity"),
        )
        changes = [
            name
            for name, key in tracked_changes
            if previous.get(key) != record.get(key)
        ]
        base_prompt_changed = "base_prompt" in changes
        artist_labels_changed = "artist_labels" in changes
        same_context = _snapshot_values_match(
            previous.get("context_snapshot"),
            snapshot,
        )
        same_base_inputs = (
            record["base_raw_fp"] != "none"
            and record["base_ids_fp"] != "none"
            and previous.get("base_raw_fp") == record["base_raw_fp"]
            and previous.get("base_ids_fp") == record["base_ids_fp"]
        )
        same_artist_inputs = (
            record["artist_raws_fp"] != "none"
            and previous.get("artist_raws_fp") == record["artist_raws_fp"]
            and previous.get("artist_ids_fp") == record["artist_ids_fp"]
            and previous.get("artist_t5_weights_fp")
            == record["artist_t5_weights_fp"]
        )
        if base_prompt_changed and same_context:
            transition = "base_prompt_changed_context_same_suspect"
            suspects.append("base_context_same_after_base_prompt_change")
        elif base_prompt_changed:
            transition = "base_prompt_changed_context_changed"
        elif same_context:
            transition = "base_prompt_unchanged_context_same"
        else:
            transition = "base_prompt_unchanged_context_changed"
        if base_prompt_changed and same_base_inputs:
            suspects.append("base_encoded_same_after_base_prompt_change")
        if artist_labels_changed and same_artist_inputs:
            suspects.append("artist_encoded_same_after_label_change")

    uuid_values = (
        conditioning_uuids
        if isinstance(conditioning_uuids, (list, tuple))
        else (() if conditioning_uuids is None else (conditioning_uuids,))
    )
    marker_values = (
        cond_or_uncond
        if isinstance(cond_or_uncond, (list, tuple))
        else (() if cond_or_uncond is None else (cond_or_uncond,))
    )
    uuids = tuple(str(value) for value in uuid_values)
    markers = tuple(str(value) for value in marker_values)
    context_record = dict(record)
    context_record.update({
        "transition": transition,
        "suspects": tuple(suspects),
        "uuids": uuids,
        "markers": markers,
    })
    run["context"] = context_record
    run.setdefault("stages", {})["base_context"] = snapshot
    run["history_previous"] = previous
    run["history_current"] = record

    if model_key != "none":
        with _history_lock:
            _last_context_by_model[model_key] = record
            _last_context_by_model.move_to_end(model_key)
            while len(_last_context_by_model) > _HISTORY_LIMIT:
                _last_context_by_model.popitem(last=False)

    level = logging.WARNING if suspects else logging.INFO
    logger.log(
        level,
        "[AnimaSemanticDiag] event=context run=%s transition=%s "
        "changed=%s suspect=%s key=%s shape=%s bad=%s context_fp=%s "
        "base_raw_fp=%s base_ids_fp=%s artist_raws_fp=%s "
        "artist_ids_fp=%s artist_input_fp=%s markers=%s cond_uuids=%s",
        run.get("run_id"),
        transition,
        ",".join(changes) or "none",
        ",".join(suspects) or "none",
        context_key,
        _snapshot_shape(snapshot),
        _snapshot_bad(snapshot),
        record["context_fp"],
        record["base_raw_fp"],
        record["base_ids_fp"],
        record["artist_raws_fp"],
        record["artist_ids_fp"],
        record["artist_input_fp"],
        markers or "none",
        uuids or "none",
    )


@_guard()
def record_stage(state, stage, snapshot):
    if not is_enabled():
        return
    run = _active_run(state)
    if not isinstance(run, dict):
        return
    stage = str(stage)
    stages = run.setdefault("stages", {})
    if stage in stages:
        return
    stages[stage] = snapshot
    previous = run.get("history_previous")
    current = run.get("history_current")
    previous_snapshot = None
    dependency_changes = []
    if isinstance(previous, dict):
        previous_snapshot = (previous.get("stages") or {}).get(stage)
        if stage == "artist_sum":
            dependencies = (
                ("artist_inputs", "artist_input_fp"),
                ("alignment", "alignment_mode"),
                ("base_ids", "base_ids_fp"),
                ("patch", "patch_identity"),
            )
            dependency_changes = [
                name
                for name, key in dependencies
                if previous.get(key) != current.get(key)
            ] if isinstance(current, dict) else []
        elif stage == "mixed_context":
            current_stages = (
                current.get("stages") or {}
                if isinstance(current, dict)
                else {}
            )
            previous_stages = previous.get("stages") or {}
            if not _snapshot_values_match(
                previous_stages.get("base_context"),
                current_stages.get("base_context"),
            ):
                dependency_changes.append("base_context")
            if not _snapshot_values_match(
                previous_stages.get("artist_sum"),
                current_stages.get("artist_sum"),
            ) and (
                abs(float(previous.get("strength") or 0.0)) > 1e-12
                or abs(float(current.get("strength") or 0.0)) > 1e-12
            ):
                dependency_changes.append("artist_sum")
            if (
                isinstance(current, dict)
                and previous.get("mix_config_fp") != current.get("mix_config_fp")
            ):
                dependency_changes.append("mix_config")

    transition = "first_observation"
    suspects = []
    if _snapshot_key(previous_snapshot) is not None:
        same_output = _snapshot_values_match(previous_snapshot, snapshot)
        if dependency_changes and same_output:
            transition = "inputs_changed_output_same_suspect"
            suspects.append(f"{stage}_same_after_input_change")
        elif dependency_changes:
            transition = "inputs_changed_output_changed"
        elif same_output:
            transition = "inputs_unchanged_output_same"
        else:
            transition = "inputs_unchanged_output_changed"

    if isinstance(current, dict):
        current.setdefault("stages", {})[stage] = snapshot
    level = logging.WARNING if suspects else logging.INFO
    logger.log(
        level,
        "[AnimaSemanticDiag] event=tensor_stage run=%s stage=%s "
        "transition=%s dependency_changed=%s suspect=%s "
        "shape=%s bad=%s fp=%s",
        run.get("run_id"),
        stage,
        transition,
        ",".join(dependency_changes) or "none",
        ",".join(suspects) or "none",
        _snapshot_shape(snapshot),
        _snapshot_bad(snapshot),
        _snapshot_fp(snapshot),
    )


@_guard()
def record_cache_lookup(state, cache_name, hit):
    if not is_enabled():
        return
    run = _active_run(state)
    if not isinstance(run, dict):
        return
    counts = run.setdefault("caches", {}).setdefault(
        str(cache_name),
        {"hit": 0, "miss": 0},
    )
    counts["hit" if hit else "miss"] += 1


@_guard()
def record_anchor_event(state, event):
    if not is_enabled():
        return
    run = _active_run(state)
    if not isinstance(run, dict):
        return
    event = str(event)
    anchor = run.setdefault("anchor", {})
    anchor[event] = int(anchor.get(event, 0)) + 1


@_guard()
def end_run(state, *, outcome):
    if not is_enabled():
        return
    diag = _diag_for_state(state)
    if isinstance(diag, dict):
        _emit_run_end(diag, str(outcome))


def _reset_history_for_tests():
    with _history_lock:
        _last_context_by_model.clear()


def _reset_file_logging_for_tests(root_logger=None):
    global _session_log_path

    target_logger = root_logger or logging.getLogger()
    with _file_log_lock:
        for handler in tuple(getattr(target_logger, "handlers", ())):
            if not getattr(handler, _FILE_HANDLER_MARKER, False):
                continue
            target_logger.removeHandler(handler)
            handler.close()
        _session_log_path = None
