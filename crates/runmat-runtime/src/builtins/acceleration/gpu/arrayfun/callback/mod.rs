mod invoke;
mod parse;

use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Closure;

#[derive(Clone)]
pub(super) enum Callable {
    Builtin { identity: BuiltinCatalogIdentity },
    DynamicName { name: String },
    ExternalName { name: String },
    Closure(Closure),
}

impl Callable {
    pub(super) fn builtin_identity(&self) -> Option<BuiltinCatalogIdentity> {
        match self {
            Self::Builtin { identity } => Some(*identity),
            Self::DynamicName { .. } | Self::ExternalName { .. } | Self::Closure(_) => None,
        }
    }
}
