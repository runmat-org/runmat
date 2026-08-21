use super::{BuiltinId, DefPath, FunctionId, MethodId, QualifiedName, SymbolName};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub enum CallableIdentity {
    BoundFunction(FunctionId),
    ExternalFunction {
        function: FunctionId,
        display_name: String,
    },
    Builtin(BuiltinId),
    Imported(DefPath),
    Method(MethodId),
    AnonymousFunction(FunctionId),
    DynamicName(SymbolName),
    ExternalName(QualifiedName),
}

impl CallableIdentity {
    pub fn display_name(&self) -> Option<String> {
        match self {
            Self::BoundFunction(_) | Self::AnonymousFunction(_) => None,
            Self::ExternalFunction { display_name, .. } => {
                (!display_name.is_empty()).then_some(display_name.clone())
            }
            Self::Builtin(id) => (!id.0.is_empty()).then_some(id.0.clone()),
            Self::Imported(path) => path.module.display_name(),
            Self::Method(id) => (!id.0.is_empty()).then_some(id.0.clone()),
            Self::DynamicName(name) => (!name.0.is_empty()).then_some(name.0.clone()),
            Self::ExternalName(name) => name.display_name(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize)]
pub enum CallableFallbackPolicy {
    None,
    RuntimeNameResolution,
    ObjectDispatch,
    ExternalBoundary,
}

impl CallableFallbackPolicy {
    fn is_well_formed_external_name(name: &QualifiedName) -> bool {
        name.0.len() > 1 && name.0.iter().all(|segment| !segment.0.trim().is_empty())
    }

    fn is_well_formed_imported_path(path: &DefPath) -> bool {
        let Some(last_item) = path.item.last() else {
            return false;
        };
        let Some(last_module_segment) = path.module.0.last() else {
            return false;
        };
        path.module.display_name().is_some()
            && !last_item.display_name().trim().is_empty()
            && last_module_segment.0 == last_item.display_name()
    }

    pub fn allows_runtime_name_resolution(self) -> bool {
        matches!(self, Self::RuntimeNameResolution)
    }

    pub fn allows_semantic_name_resolution_for(self, identity: &CallableIdentity) -> bool {
        match identity {
            CallableIdentity::DynamicName(SymbolName(name))
            | CallableIdentity::Method(MethodId(name)) => {
                self.allows_runtime_name_resolution() && !name.trim().is_empty()
            }
            CallableIdentity::Imported(path) => {
                self.allows_runtime_name_resolution() && Self::is_well_formed_imported_path(path)
            }
            CallableIdentity::ExternalName(name) => {
                matches!(self, Self::ExternalBoundary) && Self::is_well_formed_external_name(name)
            }
            _ => false,
        }
    }

    pub fn allows_vm_name_fallback_for(self, identity: &CallableIdentity) -> bool {
        match identity {
            CallableIdentity::DynamicName(SymbolName(name)) => {
                self.allows_runtime_name_resolution() && !name.trim().is_empty()
            }
            CallableIdentity::Imported(path) => {
                self.allows_runtime_name_resolution() && Self::is_well_formed_imported_path(path)
            }
            CallableIdentity::ExternalName(name) => {
                matches!(self, Self::ExternalBoundary) && Self::is_well_formed_external_name(name)
            }
            _ => false,
        }
    }

    pub fn resolution_name_for(self, identity: &CallableIdentity) -> Option<String> {
        if !self.allows_semantic_name_resolution_for(identity) {
            return None;
        }
        match identity {
            CallableIdentity::DynamicName(SymbolName(name))
            | CallableIdentity::Method(MethodId(name)) => {
                let trimmed = name.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            CallableIdentity::Imported(path) => path.module.display_name(),
            CallableIdentity::ExternalName(name) if Self::is_well_formed_external_name(name) => {
                name.display_name()
            }
            _ => None,
        }
    }

    pub fn vm_fallback_name_for(self, identity: &CallableIdentity) -> Option<String> {
        if !self.allows_vm_name_fallback_for(identity) {
            return None;
        }
        match identity {
            CallableIdentity::DynamicName(SymbolName(name)) => {
                let trimmed = name.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            CallableIdentity::Imported(path) => path.module.display_name(),
            CallableIdentity::ExternalName(name) if Self::is_well_formed_external_name(name) => {
                name.display_name()
            }
            _ => None,
        }
    }

    pub fn supports_vm_static_call(self) -> bool {
        matches!(self, Self::RuntimeNameResolution | Self::ExternalBoundary)
    }

    pub fn supports_vm_method_or_member_call(self) -> bool {
        matches!(self, Self::RuntimeNameResolution | Self::ObjectDispatch)
    }

    pub fn allows_object_dispatch(self) -> bool {
        matches!(self, Self::ObjectDispatch)
    }

    pub fn post_object_dispatch(self) -> Self {
        match self {
            Self::ObjectDispatch => Self::RuntimeNameResolution,
            other => other,
        }
    }
}
