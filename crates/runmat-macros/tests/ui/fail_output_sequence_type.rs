use runmat_macros::runtime_builtin;
use runmat_value::{Value, ValueSequence};
use std::future::Future;
use std::pin::Pin;

type RuntimeError = String;

mod context {
    pub mod host_allocation {
        pub type HostAllocationError = String;
    }
}

mod builtin {
    use super::*;

    pub struct RuntimeBuiltinInvocation<'a> {
        pub args: &'a [Value],
        pub context: Option<()>,
    }

    pub struct BuiltinContinuationInput {
        pub args: crate::execution::RootedValues,
        pub context: Option<()>,
    }

    impl RuntimeBuiltinInvocation<'_> {
        pub fn into_continuation_input(self) -> Result<BuiltinContinuationInput, String> {
            Ok(BuiltinContinuationInput {
                args: crate::execution::RootedValues::new(self.args.to_vec())
                    .map_err(|error| error.into_runtime_error())?,
                context: self.context,
            })
        }
    }

    pub type RuntimeBuiltinFuture =
        Pin<Box<dyn Future<Output = Result<ValueSequence, String>> + 'static>>;

    pub fn rejected_builtin_future(error: String) -> RuntimeBuiltinFuture {
        Box::pin(std::future::ready(Err(error)))
    }

    pub fn admitted_builtin_future_with<F>(
        context: Option<()>,
        _name: &'static str,
        make_future: impl FnOnce(Option<()>) -> F,
    ) -> RuntimeBuiltinFuture
    where
        F: Future<Output = Result<ValueSequence, String>> + 'static,
    {
        Box::pin(make_future(context))
    }

    pub fn continuation_layout_for<F>(_: fn(BuiltinContinuationInput) -> F) -> Result<u64, String>
    where
        F: Future<Output = Result<ValueSequence, String>>,
    {
        Ok(std::mem::size_of::<F>() as u64)
    }

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
        pub continuation_layout: Option<fn() -> Result<u64, String>>,
        pub provenance: Option<&'static runmat_builtins::BuiltinImplementationProvenance>,
    }

    impl RuntimeBuiltinBinding {
        pub const fn new(
            identity: runmat_builtins::BuiltinBindingIdentity,
            implementation: for<'a> fn(RuntimeBuiltinInvocation<'a>) -> RuntimeBuiltinFuture,
        ) -> Self {
            Self {
                identity,
                implementation,
                continuation_layout: None,
                provenance: None,
            }
        }

        pub const fn with_continuation_layout(
            mut self,
            layout: fn() -> Result<u64, String>,
        ) -> Self {
            self.continuation_layout = Some(layout);
            self
        }

        pub const fn with_provenance(
            mut self,
            provenance: &'static runmat_builtins::BuiltinImplementationProvenance,
        ) -> Self {
            self.provenance = Some(provenance);
            self
        }
    }

    runmat_builtins::inventory::collect!(RuntimeBuiltinBinding);
}

mod execution {
    use super::Value;
    use std::future::Future;

    pub struct RootedValues(Vec<Value>);
    pub type RootedValueSequence = runmat_value::ValueSequence;

    pub struct RootError;

    impl RootError {
        pub fn into_runtime_error(self) -> String {
            "root admission failed".into()
        }
    }

    impl RootedValues {
        pub fn fork(&self) -> Self {
            Self(self.0.clone())
        }

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

    impl std::ops::Deref for RootedValues {
        type Target = [Value];

        fn deref(&self) -> &Self::Target {
            &self.0
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
