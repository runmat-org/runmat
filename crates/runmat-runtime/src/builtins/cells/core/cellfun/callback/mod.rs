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
    pub(super) fn parse(value: Value) -> BuiltinResult<Self> {
        if let Some(shorthand) = CellfunShorthand::parse_bare_value(&value) {
            return Ok(Self::Shorthand(shorthand));
        }
        MappedCallable::parse(value)
            .map(Self::Mapped)
            .map_err(error_map::parse)
    }

    pub(super) async fn call(&self, arguments: &[Value]) -> BuiltinResult<Value> {
        match self {
            Self::Mapped(callable) => callable.invoke(arguments).await.map_err(error_map::call),
            Self::Shorthand(callable) => callable.call(arguments).await,
        }
    }
}
