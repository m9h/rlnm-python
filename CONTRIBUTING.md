# Contributing

Use `uv sync --locked`, `uv run pytest`, and `uv run ruff check .` before opening
a pull request. Add a numerical or spatial regression test for changes to
scientific behaviour. Preserve the upstream source pin, citation and attribution.

Method changes must be explicit and documented independently of compatibility
fixes. Reference fixtures should state their engine, upstream version, input
hashes, random samples and tolerance. Do not add patient data or restricted
reference connectomes. Use synthetic examples or download instructions instead.

Performance changes should report wall time and peak memory, plus numerical
agreement on the same inputs and permutation set. Do not assume changing a
connectome resolution, aggregation scheme or approximation preserves the method.
