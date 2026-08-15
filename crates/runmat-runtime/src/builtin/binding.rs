use runmat_builtins::BuiltinBindingIdentity;
use runmat_value::Value;
use std::{future::Future, pin::Pin};

use crate::RuntimeError;

pub type RuntimeBuiltinFuture =
    Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>>;
pub type RuntimeBuiltinImplementation = fn(&[Value]) -> RuntimeBuiltinFuture;

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
