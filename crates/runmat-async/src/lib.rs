pub mod runtime_error;
pub use runtime_error::{
    runtime_error, CallFrame, ErrorContext, RuntimeError, RuntimeErrorBuilder,
};

/// Narrow set of interaction kinds used for host I/O hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    Line {
        echo: bool,
    },
    KeyPress,
}
