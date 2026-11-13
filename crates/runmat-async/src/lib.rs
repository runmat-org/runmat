pub mod runtime_error;
pub use runtime_error::{
    runtime_error, CallFrame, CallFrameDiagnosticSource, CallFrameSourceOwner,
    DiagnosticAllocationLease, DiagnosticConstructionError, DiagnosticCopyProgress, DiagnosticText,
    DiagnosticTextCopy, ErrorContext, GpuGatherRetry, ManagedDiagnosticAllocation,
    ManagedDiagnosticAllocationLease, RetainedDiagnosticAllocation, RuntimeDiagnosticConstruction,
    RuntimeDiagnosticField, RuntimeDiagnosticLayout, RuntimeError, RuntimeErrorBuilder,
    RuntimeErrorOwner, RuntimeErrorSourceBound, RuntimeFailureOrigin, SharedRuntimeErrorSource,
};

/// Narrow set of interaction kinds used for host I/O hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    Line { echo: bool },
    KeyPress,
}
