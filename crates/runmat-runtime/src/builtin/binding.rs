use runmat_builtins::BuiltinBindingIdentity;
use runmat_value::{Value, ValueSequence};
use std::{future::Future, pin::Pin};

use crate::context::RuntimeContext;
use crate::RuntimeError;

pub type RuntimeBuiltinFuture =
    Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>> + 'static>>;
/// One admitted builtin call. The arguments are borrowed only until the
/// implementation constructs its owned future; the session authority is an
/// inexpensive `Rc`-backed clone that may be retained across suspension.
pub struct RuntimeBuiltinInvocation<'a> {
    pub args: &'a [Value],
    pub context: Option<RuntimeContext>,
    pub requested_outputs: Option<usize>,
}

impl<'a> RuntimeBuiltinInvocation<'a> {
    pub fn new(
        args: &'a [Value],
        context: Option<RuntimeContext>,
        requested_outputs: Option<usize>,
    ) -> Self {
        Self {
            args,
            context,
            requested_outputs,
        }
    }
}

pub type RuntimeBuiltinImplementation =
    for<'a> fn(RuntimeBuiltinInvocation<'a>) -> RuntimeBuiltinFuture;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RuntimeBuiltinBinding {
    pub identity: BuiltinBindingIdentity,
    pub implementation: RuntimeBuiltinImplementation,
}

impl RuntimeBuiltinBinding {
    pub const fn new(
        identity: BuiltinBindingIdentity,
        implementation: RuntimeBuiltinImplementation,
    ) -> Self {
        Self {
            identity,
            implementation,
        }
    }
}

impl std::fmt::Debug for RuntimeBuiltinBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeBuiltinBinding")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}
