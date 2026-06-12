//! CAD interop namespace (locked placement).

mod step;

pub(crate) use step::{parse_step_summary, StepImportSummary};

/// Marker type for CAD interop pipeline activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CadInteropModule;
