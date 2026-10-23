mod error_map;
mod shorthand;

use crate::builtins::common::mapped_callable::MappedCallable;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_value::Value;

use shorthand::CellfunShorthand;

#[derive(Clone)]
pub(super) enum Callable {
    Mapped(MappedCallable),
    Shorthand(CellfunShorthand),
}

impl Callable {
    pub(super) fn validate_outputs(&self, outputs: usize) -> BuiltinResult<()> {
        if outputs > 1 && matches!(self, Self::Shorthand(_)) {
            return Err(super::error::invalid(
                "cellfun: shorthand callbacks return one output",
            ));
        }
        Ok(())
    }
    pub(super) fn parse(value: Value) -> BuiltinResult<Self> {
        if let Some(shorthand) = CellfunShorthand::parse_bare_value(&value)? {
            return Ok(Self::Shorthand(shorthand));
        }
        MappedCallable::parse(value)
            .map(Self::Mapped)
            .map_err(error_map::parse)
    }

    pub(super) async fn call_retained(
        &self,
        context: Option<&crate::context::RuntimeContext>,
        arguments: crate::call::BuiltinArguments,
        outputs: usize,
    ) -> BuiltinResult<RootedValue> {
        match self {
            Self::Mapped(callable) => callable
                .invoke_retained(context, arguments, outputs)
                .await
                .map_err(error_map::call),
            Self::Shorthand(callable) => {
                self.validate_outputs(outputs)?;
                let call = callable.call(arguments.view());
                let value = match context {
                    Some(context) => context.scope(call).await?,
                    None => call.await?,
                };
                match context {
                    Some(context) => RootedValue::in_context(value, context),
                    None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
                }
            }
        }
    }
}
