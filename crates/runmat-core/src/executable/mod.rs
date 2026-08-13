#[cfg(feature = "jit")]
mod backend;
mod coverage;
mod invocation;
mod portable;
mod revision;
mod source;
mod source_map;
mod unit;

pub use coverage::CoveragePlan;
pub use invocation::{InvocationControl, ProcedureInvocation, ProcedureTarget};
pub use revision::ExecutableRevision;
pub use runmat_test::coverage::{CoverageFragment, CoverageMetric, CoverageSite};
pub use source::ExecutableSource;
pub use source_map::{ExecutableSourceMap, SourceMapEntry};
pub use unit::ExecutableUnit;
