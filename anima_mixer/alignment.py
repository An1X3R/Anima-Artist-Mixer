"""Lossless token-ID alignment for post-adapter artist embeddings."""

import torch


def token_ids_to_list(token_ids, name="token IDs"):
    if token_ids is None or not torch.is_tensor(token_ids):
        raise ValueError(f"{name} must be a tensor")
    ids = token_ids.detach()
    if ids.dim() == 0:
        ids = ids.reshape(1)
    elif ids.dim() > 1:
        leading = 1
        for size in ids.shape[:-1]:
            leading *= int(size)
        if leading != 1:
            raise ValueError(
                f"{name} must contain one token sequence, got shape {tuple(ids.shape)}"
            )
        ids = ids.reshape(-1)
    return [int(value) for value in ids.to(device="cpu", dtype=torch.int64).tolist()]


def _find_contiguous_subsequence(sequence, subsequence):
    if len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return None


def _lcs_pairs(base_tokens, artist_tokens):
    """Return monotonic equal-token pairs, preferring later artist positions."""
    base_count = len(base_tokens)
    artist_count = len(artist_tokens)
    dp = [[0] * (artist_count + 1) for _ in range(base_count + 1)]

    for base_index in range(base_count - 1, -1, -1):
        row = dp[base_index]
        next_row = dp[base_index + 1]
        base_token = base_tokens[base_index]
        for artist_index in range(artist_count - 1, -1, -1):
            if base_token == artist_tokens[artist_index]:
                row[artist_index] = 1 + next_row[artist_index + 1]
            else:
                row[artist_index] = max(
                    next_row[artist_index], row[artist_index + 1]
                )

    pairs = []
    base_index = 0
    artist_index = 0
    while base_index < base_count and artist_index < artist_count:
        current = dp[base_index][artist_index]
        if dp[base_index][artist_index + 1] == current:
            artist_index += 1
            continue
        if (
            base_tokens[base_index] == artist_tokens[artist_index]
            and current == 1 + dp[base_index + 1][artist_index + 1]
        ):
            pairs.append((base_index, artist_index))
            base_index += 1
            artist_index += 1
            continue
        base_index += 1
    return pairs


def _match_base_tokens(base_tokens, artist_tokens):
    start = _find_contiguous_subsequence(artist_tokens, base_tokens)
    if start is not None:
        return (
            [(index, start + index) for index in range(len(base_tokens))],
            "exact",
        )
    return _lcs_pairs(base_tokens, artist_tokens), "lcs"


def build_base_anchored_plan(base_ids, artist_ids_list):
    """Build a common sequence canvas without dropping any source token row."""
    base_tokens = token_ids_to_list(base_ids, "base t5xxl_ids")
    if not base_tokens:
        raise ValueError("base t5xxl_ids is empty")
    if not artist_ids_list:
        raise ValueError("at least one artist token sequence is required")

    base_count = len(base_tokens)
    layouts = []
    gap_widths = [0] * (base_count + 1)

    for artist_number, artist_ids in enumerate(artist_ids_list):
        artist_tokens = token_ids_to_list(
            artist_ids, f"artist[{artist_number}] t5xxl_ids"
        )
        if not artist_tokens:
            raise ValueError(f"artist[{artist_number}] t5xxl_ids is empty")

        pairs, method = _match_base_tokens(base_tokens, artist_tokens)
        matched = {base_index: artist_index for base_index, artist_index in pairs}
        gaps = [[] for _ in range(base_count + 1)]
        previous_artist = -1
        for base_index, artist_index in pairs:
            gaps[base_index].extend(range(previous_artist + 1, artist_index))
            previous_artist = artist_index
        gaps[base_count].extend(range(previous_artist + 1, len(artist_tokens)))

        for gap_index, indices in enumerate(gaps):
            gap_widths[gap_index] = max(gap_widths[gap_index], len(indices))
        layouts.append({
            "token_count": len(artist_tokens),
            "matched": matched,
            "gaps": gaps,
            "method": method,
            "matched_count": len(pairs),
        })

    gap_starts = []
    base_positions = []
    cursor = 0
    for gap_index in range(base_count + 1):
        gap_starts.append(cursor)
        cursor += gap_widths[gap_index]
        if gap_index < base_count:
            base_positions.append(cursor)
            cursor += 1

    artist_positions = []
    methods = []
    matched_counts = []
    for layout in layouts:
        positions = [None] * layout["token_count"]
        for base_index, artist_index in layout["matched"].items():
            positions[artist_index] = base_positions[base_index]

        for gap_index, artist_indices in enumerate(layout["gaps"]):
            if gap_index < base_count:
                start = (
                    gap_starts[gap_index]
                    + gap_widths[gap_index]
                    - len(artist_indices)
                )
            else:
                start = gap_starts[gap_index]
            for offset, artist_index in enumerate(artist_indices):
                positions[artist_index] = start + offset

        if any(position is None for position in positions):
            raise RuntimeError("internal alignment error: an artist token was not placed")
        if len(set(positions)) != len(positions):
            raise RuntimeError("internal alignment error: artist token positions overlap")
        artist_positions.append(tuple(int(position) for position in positions))
        methods.append(layout["method"])
        matched_counts.append(layout["matched_count"])

    return {
        "length": cursor,
        "base_token_count": base_count,
        "base_positions": tuple(base_positions),
        "artist_token_counts": tuple(
            layout["token_count"] for layout in layouts
        ),
        "artist_positions": tuple(artist_positions),
        "methods": tuple(methods),
        "matched_counts": tuple(matched_counts),
    }


def scatter_embedding_rows(embedding, positions, output_length):
    if not torch.is_tensor(embedding) or embedding.dim() != 3:
        raise ValueError(
            f"embedding must be [B, T, D], got {getattr(embedding, 'shape', None)}"
        )
    token_count = len(positions)
    if embedding.shape[1] < token_count:
        raise ValueError(
            f"Adapter output has {embedding.shape[1]} rows for {token_count} token IDs"
        )
    if any(position < 0 or position >= output_length for position in positions):
        raise ValueError("alignment position is outside the common canvas")

    output = torch.zeros(
        (embedding.shape[0], output_length, embedding.shape[2]),
        device=embedding.device,
        dtype=embedding.dtype,
    )
    index = torch.tensor(positions, device=embedding.device, dtype=torch.long)
    output.index_copy_(1, index, embedding[:, :token_count, :])
    return output


def align_artist_embeddings(embeddings, plan):
    if len(embeddings) != len(plan["artist_positions"]):
        raise ValueError(
            f"artist embedding/plan count differs: {len(embeddings)} != "
            f"{len(plan['artist_positions'])}"
        )
    return [
        scatter_embedding_rows(embedding, positions, plan["length"])
        for embedding, positions in zip(embeddings, plan["artist_positions"])
    ]


def align_base_context(context, plan):
    return scatter_embedding_rows(
        context,
        plan["base_positions"],
        plan["length"],
    )
