<!-- 6ec54c70-9ed9-45d8-89bb-93de31e93c92 52740b5c-9dfa-40c8-bc33-f1fac2d56fef -->
# Benchmark Data Automation Plan

## 1. Share the canonical benchmark suite loader

- Extract the raw-suite reader from `website/lib/marketing-benchmarks.ts` into a new helper (e.g. `website/lib/benchmark-suite.ts`) that simply reads/parses `content/benchmarks.json` (or `MARKETING_BENCH_PATH`).
- Update `marketing-benchmarks.ts` to import this helper so existing hero cards keep working, and re-export a typed `getBenchmarkCase(caseId)` utility that other modules (components/scripts) can reuse.

## 2. Derive sweep tables from benchmark cases

- Build a pure utility (e.g. `website/lib/benchmark-sweeps.ts`) that, given a case id plus implementation ids, produces the rows/summary needed for the sweep tables (param label, RunMat median, PyTorch/NumPy comparisons, min/max ranges). Reuse the shared suite loader and keep formatting helpers (number formatting, “× faster/as fast” wording) centralized.
- Ensure the helper gracefully handles missing cases or incomplete data (fallback to cached/static rows so the UI doesn’t crash while new benchmarks are still cooking).

## 3. Make `<BenchmarkSweepCarousel>` data-driven

- Turn `BenchmarkSweepCarousel` into an async server component that fetches sweep data via the new helper, then render a lightweight client component that maintains carousel interaction state.
- Replace the three hardcoded row arrays in `FourKImagePipelineSweep`, `MonteCarloSweep`, and `ElementwiseMathSweep` with props driven by the computed data; keep presentational markup intact so Tailwind classes remain unchanged.
- Confirm the data shown in the carousel now follows the exact same JSON that powers the homepage hero chart, so new benchmark runs instantly update both surfaces.

## 4. Auto-update README benchmark tables

- Add HTML comment guards around the three benchmark tables in the repo-root `README.md` (e.g. `<!-- BENCHMARK_TABLE:4k -->…<!-- /BENCHMARK_TABLE:4k -->`) to allow scripted replacements without touching the rest of the document.
- Create a Node/TS script (e.g. `scripts/update-benchmark-docs.ts`) that reuses the sweep utility to regenerate those markdown tables and rewrites the guarded sections. Wire it up to `package.json` (maybe `pnpm benchmarks:update-docs`) so maintainers can run it after pulling new benchmark data.
- Document the workflow (in `README` or CONTRIBUTING) so whoever lands new benchmark runs knows to regenerate the docs; optionally hook the script into CI to fail if the checked-in tables are stale.

## Implementation Todos

- `suite-helper`: Extract `readSuite()/getBenchmarkCase()` helpers shared across website + scripts.
- `sweep-data`: Add utilities that compute sweep rows + deltas from the benchmark suite.
- `carousel-dynamic`: Refactor `BenchmarkSweepCarousel` + child components to consume computed data.
- `readme-sync`: Guard README tables and add the script/CLI hook that rewrites them from the sweep data.