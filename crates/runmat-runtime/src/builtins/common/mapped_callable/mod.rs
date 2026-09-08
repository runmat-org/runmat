mod invoke;
mod parse;

use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::{Closure, Value};

#[derive(Clone)]
pub(crate) enum MappedCallable {
    Builtin { identity: BuiltinCatalogIdentity },
    DynamicName { name: String },
    ExternalName { name: String },
    Closure(Closure),
}

#[derive(Debug)]
pub(crate) enum CallableParseError {
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
        function_name: String,
        function: String,
    },
}

impl MappedCallable {
    pub(crate) fn builtin_identity(&self) -> Option<BuiltinCatalogIdentity> {
        match self {
            Self::Builtin { identity } => Some(*identity),
            Self::DynamicName { .. } | Self::ExternalName { .. } | Self::Closure(_) => None,
        }
    }
}
