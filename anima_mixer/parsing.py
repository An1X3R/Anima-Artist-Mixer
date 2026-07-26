"""Parsing helpers for artist chains, weights, layer filters, and prompts."""

import re

from .constants import WEIGHT_MAX, WEIGHT_MIN


def clamp_float(value, lo, hi):
    value = float(value)
    if value != value:
        raise ValueError("NaN is not a valid value")
    return max(lo, min(hi, value))


def split_artist_chain(chain):
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _is_float(text):
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _strip_colon_markers(name):
    s = str(name).strip()
    if s.startswith("::") and s.endswith("::"):
        inner = s[2:-2].strip()
        if inner and "::" not in inner:
            return inner
    if not s.startswith("::") and s.endswith("::") and "::" not in s[:-2]:
        head = s[:-2].strip()
        if head:
            return head
    if not s.startswith("::") and "::" in s:
        name_part, _, w_part = s.rpartition("::")
        if name_part.strip() and w_part.strip() and _is_float(w_part.strip()):
            return name_part.strip()
    return s


def parse_artist_entries(parts):
    entries = []
    for raw in parts:
        s = str(raw).strip()
        if not s:
            continue
        weight = 1.0
        explicit = False

        if "::" in s:
            head, _, tail = s.partition("::")
            head_stripped = head.strip()
            tail_stripped = tail.strip()
            if tail_stripped.endswith("::"):
                tail_stripped = tail_stripped[:-2].strip()
            if head_stripped and tail_stripped:
                try:
                    weight = clamp_float(head_stripped, WEIGHT_MIN, WEIGHT_MAX)
                    explicit = True
                    s = tail_stripped
                except ValueError:
                    pass

            if not explicit:
                tmp = s[2:] if s.startswith("::") else s
                if "::" in tmp:
                    name_part, _, w_part = tmp.rpartition("::")
                    try:
                        weight = clamp_float(w_part.strip(), WEIGHT_MIN, WEIGHT_MAX)
                        explicit = True
                        s = name_part.strip()
                    except ValueError:
                        pass
                elif s.startswith("::"):
                    s = tmp.strip()

            s = _strip_colon_markers(s)

        if not s:
            continue
        entries.append((s, weight, explicit))
    return entries


def parse_artist_weights(parts):
    names = []
    weights = []
    has_explicit = False
    for name, weight, explicit in parse_artist_entries(parts):
        names.append(name)
        weights.append(weight)
        if explicit:
            has_explicit = True
    return names, weights, has_explicit


def parse_layer_filter(text, num_blocks):
    if not text:
        return None
    s = str(text).replace("，", ",").replace(" ", "")
    if not s:
        return None
    result = set()
    recognized = False
    for part in s.split(","):
        if not part:
            continue
        if "-" in part[1:]:
            dash_idx = part.index("-", 1)
            try:
                lo = int(part[:dash_idx])
                hi = int(part[dash_idx + 1:])
            except ValueError:
                continue
            recognized = True
            if lo < 0:
                lo += num_blocks
            if hi < 0:
                hi += num_blocks
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(num_blocks - 1, hi)
            if lo <= hi:
                result.update(range(lo, hi + 1))
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            recognized = True
            if v < 0:
                v += num_blocks
            if 0 <= v < num_blocks:
                result.add(v)
    if result:
        return sorted(result)
    if recognized:
        return []
    return None


def parse_anchor_seed_list(text, max_count):
    if not text:
        return []
    s = str(text).replace("，", ",")
    parts = re.split(r"[\s,;]+", s)
    seeds = []
    seen = set()
    for part in parts:
        if not part:
            continue
        try:
            seed = int(part)
        except ValueError:
            continue
        if seed < 0:
            continue
        seed = seed % (2 ** 64)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
        if len(seeds) >= max_count:
            break
    return seeds


def normalize_weights(weights):
    total = sum(abs(w) for w in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def expand_prompt_weights(text):
    if not text or "::" not in text:
        return text

    result = []
    i = 0
    n = len(text)
    while i < n:
        sep = text.find("::", i)
        if sep < 0:
            result.append(text[i:])
            break

        boundary_left = i
        for j in range(sep - 1, i - 1, -1):
            if text[j] in ",，\n\r":
                boundary_left = j + 1
                break

        weight_str = text[boundary_left:sep].strip()
        weight_val = None
        try:
            weight_val = max(0.0, min(4.0, float(weight_str)))
        except ValueError:
            pass

        if weight_val is None:
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        target_start = sep + 2
        while target_start < n and text[target_start] == " ":
            target_start += 1

        end_marker = -1
        k = target_start
        while k < n - 1:
            if text[k] in "\n\r":
                break
            if text[k] == ":" and text[k + 1] == ":":
                end_marker = k
                break
            k += 1

        if end_marker < 0:
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        target = text[target_start:end_marker].strip()
        if not target:
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        result.append(text[i:boundary_left])
        result.append(f"({target}:{weight_val:g})")
        i = end_marker + 2

    return "".join(result)
