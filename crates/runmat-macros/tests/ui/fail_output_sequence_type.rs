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

    pub trait CanonicalBuiltinOutput {
        fn into_rooted_sequence(self) -> Result<ValueSequence, String>;
    }

    impl CanonicalBuiltinOutput for ValueSequence {
        fn into_rooted_sequence(self) -> Result<ValueSequence, String> {
            Ok(self)
        }
    }

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

mod execution {
    use super::Value;
    use std::future::Future;

    pub struct RootedValues(Vec<Value>);

    pub struct RootError;

    impl RootError {
        pub fn into_runtime_error(self) -> String {
            "root admission failed".into()
        }
    }

    impl RootedValues {
        pub fn new(values: Vec<Value>) -> Result<Self, RootError> {
            Ok(Self(values))
        }

        pub async fn with_owned_values_async<R, F>(self, consume: impl FnOnce(Vec<Value>) -> F) -> R
        where
            F: Future<Output = R>,
        {
            consume(self.0).await
        }
    }
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
