# Local duration evidence

Local execution choices may show a duration only when the backend has applicable measurements. Memory admission, deadlines, CPU count, and browser hardware labels are not timing estimates.

`RunMatSession::fea_execution_timing_profile` adapts the existing bounded placement feedback profile. It creates no second timing database or solver route. The native desktop adapter measures the existing checked-source execution, including preparation, assessment and successful publication. The browser adapter measures the prepared-worker route, including startup and successful publication. Merely opening execution choices never runs a solve.

An observation is reusable only for the same checked revision, program/compiler environment, runtime session, execution route and exact thread allocation. Native measurements and browser measurements are separate. A serial measurement is never divided by core count to predict multi-core time. The session authority also prevents exported placement profiles from silently becoming another machine's measurements.

The display adapter requires at least three successful observations, an existing medium-or-high-confidence placement observation, and sample coefficient of variation no greater than 25%. It emits the observed mean and a range of two sample standard deviations. This describes recent observed variability, not a guaranteed completion interval. Evidence uses the existing placement profile’s staleness and invalidation policies. It is not placed in a short clock bucket that would discard a measurement while a long simulation runs. Failures retire the applicable timing series; cancellations and abandoned work add no successful sample. In-flight observations use the existing revocable observation permits, so profile replacement or invalidation also rejects late completions.

`estimatedMillis` remains the compatible scalar field. Optional `durationEvidence` supplies the range and sample count. The SDK validates these fields; the UI omits unqualified scalar durations. A missing estimate is normal and does not change execution eligibility.

## Deliberate limits

There is no first-run prediction, extrapolation between meshes, or browser-to-native speed multiplier. Current evidence is session-local. A browser cannot promise how quickly an unmeasured native installation will run, even on the same computer. Supporting that later requires actual matching native evidence and an explicit device/runtime identity bridge, not matching CPU names or sharing raw session profiles.

Backend measurements are collected only from the default singleton CPU route with one worker and explicit thread allocation. Sweeps, accelerator routes, custom root selections, generated geometry, custom input or assessment functions, and implicit allocations are excluded from this display estimate. Timing-aware native entrypoints accept an explicit session profile; callers without that owner continue to run normally without fabricating timing evidence.
