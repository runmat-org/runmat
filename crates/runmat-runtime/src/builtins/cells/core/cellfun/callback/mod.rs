mod error_map;
mod shorthand;

use crate::builtins::common::mapped_callable::MappedCallable;
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
        if let Some(shorthand) = CellfunShorthand::parse_bare_value(&value) {
            return Ok(Self::Shorthand(shorthand));
        }
        MappedCallable::parse(value)
            .map(Self::Mapped)
            .map_err(error_map::parse)
    }

    pub(super) async fn call(&self, arguments: &[Value], outputs: usize) -> BuiltinResult<Value> {
        match self {
            Self::Mapped(callable) => callable
                .invoke_with_outputs(arguments, outputs)
                .await
                .map_err(error_map::call),
            Self::Shorthand(callable) => {
                self.validate_outputs(outputs)?;
                callable.call(arguments).await
            }
        }
    }
}
