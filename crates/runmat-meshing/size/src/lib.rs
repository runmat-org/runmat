/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub mod adaptive;
pub mod field;
pub mod grading;
pub mod incidence;
pub mod metric;
pub mod refinement;
pub mod sources;
