mod artifact;
mod attempt;
mod diagnostic;
mod merge;
mod model;
mod status;

pub use artifact::{Artifact, ArtifactLocation};
pub use attempt::AttemptResult;
pub use diagnostic::{Diagnostic, DiagnosticDetail, DiagnosticSeverity};
pub use merge::{aggregate_run_state, merge_attempts};
pub use model::{RunResult, TestResult};
pub use status::{ResultState, TerminalDisposition};
