mod coverage;
mod interop;
mod invocation;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod portable;
mod portable_entrypoint;
mod revision;
mod source;
mod source_map;
mod unit;

pub use coverage::CoveragePlan;
pub(crate) use interop::python_environment_requirement;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use invocation::ExecutableBackendPolicy;
pub use invocation::{InvocationControl, ProcedureInvocation, ProcedureTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use native::{AotObjectDataOptions, NativeCompilationInput};
pub use revision::ExecutableRevision;
pub(crate) use revision::ExecutableSourceSelection;
pub use runmat_test::coverage::{CoverageFragment, CoverageMetric, CoverageSite};
pub(crate) use source::CapturedPrimarySource;
pub use source::ExecutableSource;
pub use source_map::{ExecutableSourceMap, SourceMapEntry};
pub(crate) use unit::ExecutableUnitDraft;
pub use unit::{reachability_requires_python, ExecutableUnit};
