use super::{ObjectAccessContext, ObjectProtocol};
use crate::call::closures::method_access_permitted_for_class;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, ClassIdentity, MemberAccess, MethodName,
};
use runmat_value::SelectedObjectClassFacts;
use runmat_value::Value;
use std::rc::Rc;

mod intra_class;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectProtocolCallingConvention {
    StandardSubstruct,
    DirectArguments,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedObjectMethod {
    pub registry_generation: Option<crate::class_registry::ClassRegistryRevision>,
    pub class: ClassIdentity,
    pub declaring_class: ClassIdentity,
    pub method: MethodName,
    pub target: ResolvedObjectMethodTarget,
    pub fallback: CallableFallbackPolicy,
    pub convention: ObjectProtocolCallingConvention,
    pub access: MemberAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedObjectMethodTarget {
    Named(CallableIdentity),
    /// Registered linked body selected by its exact portable/local issuance,
    /// not by the method's textual declaration name.
    RegisteredBound(runmat_value::BoundFunctionBinding),
    /// Receiver storage is not issuance authority. The selected VM invoker
    /// checks this owner and lease against the actual body before execution.
    Selected(
        Rc<SelectedObjectClassFacts>,
        runmat_types::ClassMethodOwner,
        runmat_value::BoundFunctionBinding,
    ),
}

/// A resolved object override, or default indexing when no override applies.
/// The method stays inline; protocol resolution adds no heap allocation.
pub type ProtocolResolution = Option<ResolvedObjectMethod>;

pub fn resolved_method_is_current(method: &ResolvedObjectMethod) -> bool {
    if matches!(method.target, ResolvedObjectMethodTarget::Selected(..)) {
        return true;
    }
    crate::class_registry::lookup_bound_method(&method.class, &method.method).is_some_and(|bound| {
        Some(bound.registry_generation) == method.registry_generation
            && bound.declaring_class == method.declaring_class
            && method.target
                == bound
                    .bound_function
                    .map(ResolvedObjectMethodTarget::RegisteredBound)
                    .unwrap_or(ResolvedObjectMethodTarget::Named(bound.callable))
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

pub(crate) fn selected_receiver_facts(
    base: &Value,
) -> Result<Option<Rc<SelectedObjectClassFacts>>, RuntimeError> {
    match base {
        Value::Object(object) => {
            if object.selected_class().is_some() && object.has_native_payload() {
                return Err(semantic_error(
                    "SelectedObjectReceiverInvalid",
                    "selected receiver also carries native object authority",
                ));
            }
            if object
                .selected_class()
                .is_some_and(|facts| facts.is_handle_class())
            {
                return Err(semantic_error(
                    "SelectedObjectReceiverInvalid",
                    "selected handle class is not a handle receiver",
                ));
            }
            Ok(object.selected_class().cloned())
        }
        Value::HandleObject(handle) if handle.valid => {
            runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(object) if object.class_name() == &handle.class_name => {
                    if object.selected_class().is_some() && object.has_native_payload() {
                        return Err(semantic_error(
                            "SelectedObjectReceiverInvalid",
                            "selected handle target also carries native object authority",
                        ));
                    }
                    if object
                        .selected_class()
                        .is_some_and(|facts| !facts.is_handle_class())
                    {
                        return Err(semantic_error(
                            "SelectedObjectReceiverInvalid",
                            "selected value class is not a handle receiver",
                        ));
                    }
                    Ok(object.selected_class().cloned())
                }
                _ => Err(semantic_error(
                    "SelectedObjectReceiverInvalid",
                    "selected handle target is not its class object",
                )),
            })
            .map_err(|_| {
                semantic_error(
                    "SelectedObjectReceiverInvalid",
                    "selected handle target is unavailable",
                )
            })?
        }
        Value::HandleObject(_) => Err(semantic_error(
            "SelectedObjectReceiverInvalid",
            "object handle is invalid",
        )),
        Value::ObjectArray(array) => {
            if array.data().is_empty() {
                return match (
                    array.selected_provenance(),
                    array.has_unselected_provenance(),
                ) {
                    ([], _) => Ok(None),
                    ([facts], false) => Ok(Some(Rc::clone(facts))),
                    _ => Err(semantic_error(
                        "SelectedObjectReceiverInvalid",
                        "empty object array has ambiguous retained class products",
                    )),
                };
            }
            let mut selected: Option<Option<Rc<SelectedObjectClassFacts>>> = None;
            for element in array.data() {
                let facts = selected_receiver_facts(element)?;
                match (&selected, facts) {
                    (None, facts) => selected = Some(facts),
                    (Some(Some(previous)), Some(facts)) if Rc::ptr_eq(previous, &facts) => {}
                    (Some(None), None) => {}
                    _ => {
                        return Err(semantic_error(
                            "SelectedObjectReceiverInvalid",
                            "object array mixes distinct selected class products",
                        ))
                    }
                }
            }
            Ok(selected.flatten())
        }
        _ => Ok(None),
    }
}

/// A selected receiver's absent method cannot be filled by a later ambient
/// same-named class publication. Stored properties remain a separate route.
pub fn selected_receiver_has_authority(base: &Value) -> Result<bool, RuntimeError> {
    Ok(selected_receiver_facts(base)?.is_some())
}

fn selected_method(
    facts: &Rc<SelectedObjectClassFacts>,
    method_name: &MethodName,
    access: &ObjectAccessContext,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    selected_method_kind(facts, method_name, access, DeclaredMethodKind::Instance)
}

#[derive(Clone, Copy)]
enum DeclaredMethodKind {
    Instance,
    Static,
}

impl DeclaredMethodKind {
    fn validate(self, is_static: bool, name: &MethodName) -> Result<(), RuntimeError> {
        match (self, is_static) {
            (Self::Instance, false) | (Self::Static, true) => Ok(()),
            (Self::Instance, true) => Err(semantic_error(
                "MethodStaticOnInstance",
                format!("Method '{name}' is static"),
            )),
            (Self::Static, false) => Err(semantic_error(
                "MethodInstanceOnClass",
                format!("Method '{name}' is not static"),
            )),
        }
    }
}

fn selected_method_kind(
    facts: &Rc<SelectedObjectClassFacts>,
    method_name: &MethodName,
    access: &ObjectAccessContext,
    kind: DeclaredMethodKind,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    let Some(method) = facts.method(method_name) else {
        return Ok(None);
    };
    kind.validate(method.owner.is_static, method_name)?;
    if method.is_abstract {
        return Err(semantic_error(
            "MethodAbstract",
            format!("Method '{}' is abstract", method_name.0),
        ));
    }
    let projection = crate::class_registry::checked_receiver_facts(facts)?;
    if !projection.receiver_member_accessible(
        &method.owner.declaring_class,
        method.access,
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
    let binding = facts.method_binding(method_name).ok_or_else(|| {
        semantic_error(
            "SelectedObjectMethodMissing",
            "selected class method has no admitted body",
        )
    })?;
    Ok(Some(ResolvedObjectMethod {
        registry_generation: None,
        class: facts.class().clone(),
        declaring_class: method.owner.declaring_class.clone(),
        method: method_name.clone(),
        target: ResolvedObjectMethodTarget::Selected(
            Rc::clone(facts),
            method.owner.clone(),
            binding,
        ),
        fallback: CallableFallbackPolicy::None,
        convention: if method.standard_substruct {
            ObjectProtocolCallingConvention::StandardSubstruct
        } else {
            ObjectProtocolCallingConvention::DirectArguments
        },
        access: method.access,
    }))
}

pub fn resolve_object_protocol(
    base: &Value,
    protocol: ObjectProtocol,
    access: &ObjectAccessContext,
) -> Result<ProtocolResolution, RuntimeError> {
    let Some(class) = crate::object::indexing::class_name_from_base(base).cloned() else {
        return Ok(None);
    };
    let method_name = protocol.method();
    let selected = selected_receiver_facts(base)?;
    if intra_class::uses_builtin_indexing(&class, protocol, access, selected.as_deref())? {
        return Ok(None);
    }
    if let Some(facts) = selected {
        let Some(method) = selected_method(&facts, &method_name, access)? else {
            return Ok(None);
        };
        if access.active_protocol == Some(protocol)
            && access.selected_active_owner.as_ref().is_some_and(|active| {
                matches!(&method.target, ResolvedObjectMethodTarget::Selected(_, owner, _) if active == owner)
            })
        {
            return Ok(None);
        }
        if protocol.uses_standard_substruct()
            && method.convention != ObjectProtocolCallingConvention::StandardSubstruct
        {
            return Err(semantic_error(
                "InvalidObjectProtocolBinding",
                "selected object protocol has an invalid calling convention",
            ));
        }
        return Ok(Some(method));
    }
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
                return Ok(None);
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
        return Ok(Some(ResolvedObjectMethod {
            registry_generation: Some(bound.registry_generation),
            class,
            declaring_class: bound.declaring_class,
            method: method_name,
            target: bound
                .bound_function
                .map(ResolvedObjectMethodTarget::RegisteredBound)
                .unwrap_or(ResolvedObjectMethodTarget::Named(bound.callable)),
            fallback: bound.fallback,
            convention,
            access: method.access,
        }));
    }

    Ok(None)
}

/// Resolve an ordinary declared instance method to the same immutable typed
/// binding used by protocol dispatch. Absence is distinct from an invalid,
/// inaccessible, static, or abstract declaration.
pub fn resolve_declared_object_method(
    base: &Value,
    method_name: &MethodName,
    access: &ObjectAccessContext,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    resolve_declared_object_method_kind(base, method_name, access, DeclaredMethodKind::Instance)
}

/// Resolve a static factory using its prototype's retained class product.
/// A same-named current class cannot replace an absent retained method.
pub(crate) fn resolve_declared_object_static_method(
    base: &Value,
    method_name: &MethodName,
    access: &ObjectAccessContext,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    resolve_declared_object_method_kind(base, method_name, access, DeclaredMethodKind::Static)
}

fn resolve_declared_object_method_kind(
    base: &Value,
    method_name: &MethodName,
    access: &ObjectAccessContext,
    kind: DeclaredMethodKind,
) -> Result<Option<ResolvedObjectMethod>, RuntimeError> {
    let Some(class) = crate::object::indexing::class_name_from_base(base).cloned() else {
        return Ok(None);
    };
    if let Some(facts) = selected_receiver_facts(base)? {
        return selected_method_kind(&facts, method_name, access, kind);
    }
    let Some(bound) = crate::class_registry::lookup_bound_method(&class, method_name) else {
        return Ok(None);
    };
    let method = &bound.declaration;
    kind.validate(method.is_static, method_name)?;
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
        registry_generation: Some(bound.registry_generation),
        class,
        declaring_class: bound.declaring_class,
        method: method_name.clone(),
        target: bound
            .bound_function
            .map(ResolvedObjectMethodTarget::RegisteredBound)
            .unwrap_or(ResolvedObjectMethodTarget::Named(bound.callable)),
        fallback: bound.fallback,
        convention: ObjectProtocolCallingConvention::DirectArguments,
        access: method.access,
    }))
}

#[cfg(test)]
#[path = "resolution/tests.rs"]
mod tests;
