pub(crate) mod callback;
mod coverage;
mod dependencies;
mod import;
mod interop;
mod invocation;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod portable;
mod portable_entrypoint;
mod portable_selection;
mod revision;
mod source;
mod source_map;
mod unit;

pub use coverage::CoveragePlan;
pub(crate) use interop::python_environment_requirement;
pub(crate) use invocation::AdmittedProcedureInvocation;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use invocation::ExecutableBackendPolicy;
pub use invocation::{InvocationControl, ProcedureArguments, ProcedureInvocation, ProcedureTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use native::{AotObjectDataOptions, NativeCompilationInput};
pub(crate) use portable_selection::SourceProductSelection;
pub use revision::ExecutableRevision;
pub(crate) use revision::ExecutableSourceSelection;
pub use runmat_test::coverage::{CoverageFragment, CoverageMetric, CoverageSite};
pub use source::ExecutableSource;
pub(crate) use source::{AdmittedProjectPrimary, CapturedPrimarySource};
pub use source_map::{ExecutableSourceMap, SourceMapEntry};
pub(crate) use unit::ExecutableUnitDraft;
pub(crate) use unit::RuntimeConstructorAdmissions;
pub use unit::{
    reachability_requires_python, ExecutableFunctionInterface, ExecutableFunctionParameter,
    ExecutableUnit,
};
