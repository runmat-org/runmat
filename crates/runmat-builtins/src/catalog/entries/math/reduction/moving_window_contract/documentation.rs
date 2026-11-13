//! Shared public explanation of the actual moving-window numerical owner.
pub(in crate::catalog::entries::math::reduction) const MOVING_WINDOW_CLASSES_TEXT: &str = concat!(
    "Real single/double inputs retain their floating output class. Logical data computes double. movsum, movmean and movprod materialize integer data once into double; movmin, movmax and movmedian retain native integer data ",
    "and compare exact int64/uint64 values without binary64 conversion. movstd and movvar reject integer data; their typed integer normalization/dimension controls remain a compatibility extension. Host complex single/double ",
    "reads canonical native storage without widening: movsum, movmean, movprod, movmin, movmax and movmedian return complex results of that class; movstd and movvar return real results of the same native floating class, using ",
    "squared complex distance. Complex-single scalar results retain tensor storage because the scalar complex value variant is double. Complex integer, sparse numeric and complex GPU moving-window inputs are not admitted by ",
    "this dense family. SimpleProvider currently computes physical double; mismatched floating precision and integer/logical provider input select only an authorized automatic host fallback before invoking its hook, never a ",
    "precision-changing attempted result.",
);

pub(in crate::catalog::entries::math::reduction) const MOVING_WINDOW_CONTROLS_TEXT: &str = concat!(
    "Data and a window are required, with at most one output. Count windows accept a positive scalar k or exactly two nonnegative backward/forward counts; even scalar windows contain k/2 preceding and (k-1)/2 following ",
    "positions. The default axis is the first extent unequal to one, including a zero extent; an explicit positive scalar dimension extends missing axes with singleton extents. Empty output geometry is retained, and discard ",
    "removes the unavailable window margins rather than inventing samples. movstd/movvar consume normalization 0 (sample) or 1 (population), or an empty normalization placeholder, before a dimension; singleton sample windows ",
    "divide by one. NaN/missing keywords and Endpoints name-value pairs may be interleaved with numeric controls; repeated missing flags and endpoint selections use the last one. SamplePoints may occur only once and must be ",
    "finite, nonempty, strictly increasing, and length-matched to the selected input axis; its endpoints must be shrink. Window and SamplePoints controls must be scalars or row/column vectors; non-singleton higher-dimensional ",
    "pages and matrices are rejected before numeric materialization. Trailing singleton axes do not change vector geometry. Endpoint fill uses the operation's documented default or a real numeric scalar, preserving exact ",
    "typed fills for native integer order statistics. All-NaN omitted windows use sum/product identities and otherwise NaN; including NaN propagates it. Complex order statistics use magnitude then phase and average the two ",
    "middle values for even median counts.",
);

pub(in crate::catalog::entries::math::reduction) const MOVING_WINDOW_ERRORS_TEXT: &str = concat!(
    "Output-count rejection precedes preparation. Missing data/window, invalid window counts or overflow, invalid dimensions, malformed name-value pairs, repeated SamplePoints, invalid/mismatched sample coordinates and ",
    "incompatible endpoints are argument errors; unsupported data representations and integer moving-dispersion data are input errors. Compatibility-gated controls keep their declared extension diagnostics. Explicit resident ",
    "controls or data requiring host fallback are rejected before gather; language compatibility does not grant placement permission. Finite host scratch without a complete owner fails closed, and exact provider request ",
    "metadata may fail admission independently. Cancellation and stale issuer/generation failures use the installed runtime's canonical control diagnostics, not numeric failure or fallback. Provider execution, ",
    "malformed/foreign/wrong-class results, provenance and restore failures retain their causes and terminal no-replay policy. This numerical owner does not read or mutate language workspace, environment, files, network, ",
    "clock, RNG, or UI; it therefore requires no such capability. Optional accelerator services and possible suspension/errors do not imply ambient service authority or full finite allocation support.",
);

pub(in crate::catalog::entries::math::reduction) const MOVING_WINDOW_PERFORMANCE_TEXT: &str = concat!("Moving windows materialize results and are not fused elementwise or reduction expressions. Host work enumerates each selected window, and median additionally orders its values; irregular SamplePoints builds membership ", "scratch. Complex kernels borrow native interleaved input directly, never materializing the whole input per element or widening a complex-single payload, and publish only the selected real or complex output payload. This ", "removes the repeated whole-payload copy; it is not a measured speedup or an asymptotically optimized prefix/rolling/tree algorithm. Complete finite host workspace ownership, cooperative bounds for all host ", "loops/order-statistics work, and provider-private allocation qualification remain open. No universal GPU residency or provider precision parity is promised.",);
