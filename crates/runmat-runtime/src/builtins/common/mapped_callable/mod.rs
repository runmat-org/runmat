mod invoke;
pub(crate) use invoke::prepend_rooted_argument;
mod parse;
mod roots;
pub(crate) use roots::root_callable_value;

use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

#[derive(Clone)]
pub(crate) enum MappedCallable {
    Builtin { identity: BuiltinCatalogIdentity },
    DynamicName { name: String },
    ExternalName { name: String },
    Closure(crate::call::descriptor::RetainedClosureValue),
    Scoped(runmat_value::ScopedFunctionHandle),
}

#[derive(Debug)]
pub(crate) enum CallableParseError {
    Runtime(crate::RuntimeError),
    EmptyText,
    EmptyHandle,
    CharacterNameMustBeRow,
    StringNameMustBeScalar,
    ScalarValue,
    UnsupportedValue(Value),
}

pub(crate) enum CallableCallError {
    Runtime(crate::RuntimeError),
    UndefinedExternal {
        identity: runmat_types::CallableIdentity,
    },
    SemanticUnavailable {
        function_name: runmat_value::Utf8Text,
        function: String,
    },
}

impl MappedCallable {
    pub(crate) fn builtin_identity(&self) -> Option<BuiltinCatalogIdentity> {
        match self {
            Self::Builtin { identity } => Some(*identity),
            Self::DynamicName { .. }
            | Self::ExternalName { .. }
            | Self::Closure(_)
            | Self::Scoped(_) => None,
        }
    }
}
#[cfg(test)]
mod tests;
