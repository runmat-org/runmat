use super::{ObjectAccessContext, ObjectProtocol};
use crate::call::closures::method_access_permitted_for_class;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, ClassIdentity, MemberAccess, MethodName,
};
use runmat_value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectProtocolCallingConvention {
    StandardSubstruct,
    DirectArguments,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedObjectMethod {
    pub registry_generation: u64,
    pub class: ClassIdentity,
    pub declaring_class: ClassIdentity,
    pub method: MethodName,
    pub callable: CallableIdentity,
    pub fallback: CallableFallbackPolicy,
    pub convention: ObjectProtocolCallingConvention,
    pub access: MemberAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolResolution {
    DefaultIndexing,
    Method(ResolvedObjectMethod),
}

pub fn resolved_method_is_current(method: &ResolvedObjectMethod) -> bool {
    crate::class_registry::lookup_bound_method(&method.class, &method.method).is_some_and(|bound| {
        bound.registry_generation == method.registry_generation
            && bound.declaring_class == method.declaring_class
            && bound.callable == method.callable
            && bound.convention
                == match method.convention {
                    ObjectProtocolCallingConvention::StandardSubstruct => {
                        crate::class_registry::RuntimeMethodCallingConvention::StandardSubstruct
                    }
                    ObjectProtocolCallingConvention::DirectArguments => {
                        crate::class_registry::RuntimeMethodCallingConvention::Direct
                    }
                }
    })
}

pub fn resolve_object_protocol(
    base: &Value,
    protocol: ObjectProtocol,
    access: &ObjectAccessContext,
) -> Result<ProtocolResolution, RuntimeError> {
    let Some(class) = crate::object::indexing::class_name_from_base(base).cloned() else {
        return Ok(ProtocolResolution::DefaultIndexing);
    };
    let method_name = protocol.method();
    if let Some(bound) = crate::class_registry::lookup_bound_method(&class, &method_name) {
        if access.active_protocol == Some(protocol) {
            if let Some(active) = access.active_method.as_ref().filter(|active| {
                active.declaring_class == bound.declaring_class
                    && active.method == bound.declaration.name
            }) {
                if active.registry_generation != bound.registry_generation
                    || active.callable != bound.callable
                {
                    return Err(semantic_error(
                        "StaleObjectMethodFrame",
                        "the active object protocol binding changed; rebuild or reload the class",
                    ));
                }
                return Ok(ProtocolResolution::DefaultIndexing);
            }
        }
        let method = &bound.declaration;
        if method.is_static {
            return Err(semantic_error(
                "MethodStaticOnInstance",
                format!("Method '{}' is static", method_name.0),
            ));
        }
        if method.is_abstract {
            return Err(semantic_error(
                "MethodAbstract",
                format!("Method '{}' is abstract", method_name.0),
            ));
        }
        if !method_access_permitted_for_class(
            &bound.declaring_class,
            &method.access,
            access.caller_class.as_ref(),
        ) {
            let (identifier, visibility) = match method.access {
                MemberAccess::Protected => ("MethodProtected", "protected"),
                _ => ("MethodPrivate", "private"),
            };
            return Err(semantic_error(
                identifier,
                format!("Method '{}' is {visibility}", method_name.0),
            ));
        }
        let convention = match bound.convention {
            crate::class_registry::RuntimeMethodCallingConvention::StandardSubstruct => {
                ObjectProtocolCallingConvention::StandardSubstruct
            }
            crate::class_registry::RuntimeMethodCallingConvention::Direct
                if !protocol.uses_standard_substruct() =>
            {
                ObjectProtocolCallingConvention::DirectArguments
            }
            crate::class_registry::RuntimeMethodCallingConvention::Direct => {
                return Err(semantic_error(
                    "InvalidObjectProtocolBinding",
                    format!(
                        "Method '{}' has an invalid object protocol binding",
                        method_name.0
                    ),
                ));
            }
        };
        return Ok(ProtocolResolution::Method(ResolvedObjectMethod {
            registry_generation: bound.registry_generation,
            class,
            declaring_class: bound.declaring_class,
            method: method_name,
            callable: bound.callable,
            fallback: bound.fallback,
            convention,
            access: method.access,
        }));
    }

    Ok(ProtocolResolution::DefaultIndexing)
}

/// Resolve an ordinary declared instance method to the same immutable typed
/// binding used by protocol dispatch. Absence is distinct from an invalid,
/// inaccessible, static, or abstract declaration.
pub fn resolve_declared_object_method(
    base: &Value,
    method_name: &MethodName,
    access: &ObjectAccessContext,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    let Some(class) = crate::object::indexing::class_name_from_base(base).cloned() else {
        return Ok(None);
    };
    let Some(bound) = crate::class_registry::lookup_bound_method(&class, method_name) else {
        return Ok(None);
    };
    let method = &bound.declaration;
    if method.is_static {
        return Err(semantic_error(
            "MethodStaticOnInstance",
            format!("Method '{}' is static", method_name.0),
        ));
    }
    if method.is_abstract {
        return Err(semantic_error(
            "MethodAbstract",
            format!("Method '{}' is abstract", method_name.0),
        ));
    }
    if !method_access_permitted_for_class(
        &bound.declaring_class,
        &method.access,
        access.caller_class.as_ref(),
    ) {
        let (identifier, visibility) = match method.access {
            MemberAccess::Protected => ("MethodProtected", "protected"),
            _ => ("MethodPrivate", "private"),
        };
        return Err(semantic_error(
            identifier,
            format!("Method '{}' is {visibility}", method_name.0),
        ));
    }
    Ok(Some(ResolvedObjectMethod {
        registry_generation: bound.registry_generation,
        class,
        declaring_class: bound.declaring_class,
        method: method_name.clone(),
        callable: bound.callable,
        fallback: bound.fallback,
        convention: ObjectProtocolCallingConvention::DirectArguments,
        access: method.access,
    }))
}

#[cfg(test)]
#[path = "resolution/tests.rs"]
mod tests;
