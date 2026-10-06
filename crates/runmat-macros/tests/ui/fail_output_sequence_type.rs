use runmat_macros::runtime_builtin;
use runmat_value::{Value, ValueSequence};
use std::future::Future;
use std::pin::Pin;

mod builtin {
    use super::*;

    pub struct RuntimeBuiltinInvocation<'a> {
        pub args: &'a [Value],
    }

    pub type RuntimeBuiltinFuture =
        Pin<Box<dyn Future<Output = Result<ValueSequence, String>> + 'static>>;

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct RuntimeBuiltinBinding {
        pub identity: runmat_builtins::BuiltinBindingIdentity,
        pub implementation: for<'a> fn(RuntimeBuiltinInvocation<'a>) -> RuntimeBuiltinFuture,
    }

    impl RuntimeBuiltinBinding {
        pub const fn new(
            identity: runmat_builtins::BuiltinBindingIdentity,
            implementation: for<'a> fn(RuntimeBuiltinInvocation<'a>) -> RuntimeBuiltinFuture,
        ) -> Self {
            Self {
                identity,
                implementation,
            }
        }
    }

    runmat_builtins::inventory::collect!(RuntimeBuiltinBinding);
}

#[runtime_builtin(
    name = "wrong_sequence_type",
    binding_variant = "default",
    output_sequence = true,
    builtin_path = "tests::wrong_sequence_type"
)]
fn wrong_sequence_type() -> Result<Value, String> {
    Ok(Value::Num(1.0))
}

fn main() {}
