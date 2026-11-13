mod context;
mod invocation;
mod path_execution;
mod receiver;
mod resolution;

#[cfg(test)]
mod tests;

pub use invocation::*;
pub use path_execution::*;
pub use receiver::*;
pub(crate) use resolution::selected_receiver_facts as selected_graph_receiver_facts;
pub use resolution::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectProtocol {
    Subsref,
    Subsasgn,
    NumArgumentsFromSubscript,
    End,
}

/// Default work follows an already selected or explicitly bypassed indexing
/// protocol. It still performs ordinary visibility and accessor admission.
#[derive(Clone, Copy)]
pub(crate) enum ProtocolDispatch {
    Resolve,
    Default,
}

impl ProtocolDispatch {
    pub(crate) fn resolve(
        self,
        base: &runmat_value::Value,
        protocol: ObjectProtocol,
        access: &ObjectAccessContext,
    ) -> Result<ProtocolResolution, crate::RuntimeError> {
        match self {
            Self::Resolve => resolve_object_protocol(base, protocol, access),
            Self::Default => Ok(None),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectAccessContext {
    pub caller_class: Option<runmat_types::ClassIdentity>,
    pub caller_method: Option<runmat_types::MethodName>,
    pub active_protocol: Option<ObjectProtocol>,
    pub active_method: Option<ActiveObjectMethodIdentity>,
    pub selected_active_owner: Option<runmat_types::ClassMethodOwner>,
    pub selected_active_binding: Option<runmat_value::BoundFunctionBinding>,
    pub selected_method_dispatch: crate::context::SelectedMethodDispatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveObjectMethodIdentity {
    pub declaring_class: runmat_types::ClassIdentity,
    pub method: runmat_types::MethodName,
    pub callable: runmat_types::CallableIdentity,
    pub registry_generation: crate::class_registry::ClassRegistryRevision,
}

impl ObjectAccessContext {
    fn selected_method_frame_is_admitted(
        owner: &runmat_types::ClassMethodOwner,
        facts: &runmat_value::SelectedObjectClassFacts,
    ) -> bool {
        match owner.role {
            runmat_types::ClassMethodRole::Ordinary => facts
                .method(&owner.method)
                .is_some_and(|method| &method.owner == owner),
            runmat_types::ClassMethodRole::Constructor => {
                facts.ancestry().contains(&owner.declaring_class) && !owner.is_static
            }
        }
    }

    pub fn from_selected_method_frame(
        owner: &runmat_types::ClassMethodOwner,
        facts: &runmat_value::SelectedObjectClassFacts,
    ) -> Result<Self, crate::RuntimeError> {
        if !Self::selected_method_frame_is_admitted(owner, facts) {
            return Err(crate::runtime_error::semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                "selected method frame differs from its receiver's class product",
            ));
        }
        let selected_active_binding = match owner.role {
            runmat_types::ClassMethodRole::Constructor => Some(facts.constructor().clone()),
            runmat_types::ClassMethodRole::Ordinary => facts.method_binding(&owner.method),
        };
        if selected_active_binding.is_none() {
            return Err(crate::runtime_error::semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                "selected method frame has no admitted body",
            ));
        }
        Ok(Self {
            caller_class: Some(owner.declaring_class.clone()),
            caller_method: Some(owner.method.clone()),
            active_protocol: (!owner.is_static)
                .then(|| ObjectProtocol::from_method(&owner.method))
                .flatten(),
            active_method: None,
            selected_active_owner: Some(owner.clone()),
            selected_active_binding,
            selected_method_dispatch: crate::context::SelectedMethodDispatch::Virtual,
        })
    }

    pub fn from_selected_super_method_frame(
        owner: &runmat_types::ClassMethodOwner,
        facts: &runmat_value::SelectedObjectClassFacts,
    ) -> Result<Self, crate::RuntimeError> {
        let method = facts
            .method_by_owner(owner)
            .filter(|method| {
                owner.role == runmat_types::ClassMethodRole::Ordinary
                    && !owner.is_static
                    && !method.is_abstract
                    && method.binding.is_some()
            })
            .ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "ObjectMethodFrameIdentityMismatch",
                    "superclass method frame has no exact receiver-owned body",
                )
            })?;
        Ok(Self {
            caller_class: Some(owner.declaring_class.clone()),
            caller_method: Some(owner.method.clone()),
            active_protocol: ObjectProtocol::from_method(&owner.method),
            selected_active_owner: Some(owner.clone()),
            selected_active_binding: method.binding.clone(),
            selected_method_dispatch: crate::context::SelectedMethodDispatch::Superclass,
            ..Default::default()
        })
    }

    pub fn from_method_frame(
        owner: Option<&runmat_types::ClassMethodOwner>,
        executable: &runmat_types::CallableIdentity,
    ) -> Result<Self, crate::RuntimeError> {
        let Some(owner) = owner else {
            return Ok(Self::default());
        };
        let bound =
            crate::class_registry::lookup_bound_method(&owner.declaring_class, &owner.method)
                .ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "StaleObjectMethodFrame",
                        format!(
                            "registered method '{}.{}' is unavailable; rebuild or reload the class",
                            owner.declaring_class, owner.method.0
                        ),
                    )
                })?;
        if bound.declaring_class != owner.declaring_class || &bound.callable != executable {
            return Err(crate::runtime_error::semantic_error(
                "ObjectMethodFrameIdentityMismatch",
                format!(
                    "executable identity does not match registered method '{}.{}': executing {executable:?}, registered {:?}",
                    owner.declaring_class, owner.method.0, bound.callable
                ),
            ));
        }
        let active_method = Some(ActiveObjectMethodIdentity {
            declaring_class: bound.declaring_class.clone(),
            method: bound.declaration.name.clone(),
            callable: bound.callable.clone(),
            registry_generation: bound.registry_generation,
        });
        Ok(Self {
            caller_class: Some(owner.declaring_class.clone()),
            caller_method: Some(owner.method.clone()),
            active_protocol: (!owner.is_static)
                .then(|| ObjectProtocol::from_method(&owner.method))
                .flatten(),
            active_method,
            selected_active_owner: None,
            selected_active_binding: None,
            selected_method_dispatch: crate::context::SelectedMethodDispatch::Virtual,
        })
    }

    pub fn from_legacy_function_name(name: Option<&str>) -> Self {
        let registered = name.and_then(crate::class_registry::caller_method_for_function);
        let caller_class = registered
            .as_ref()
            .map(|(class, _)| class.clone())
            .or_else(|| crate::call::closures::caller_class_for_function(name));
        let caller_method = registered.map(|(_, method)| method);
        let active_method = caller_class
            .as_ref()
            .zip(caller_method.as_ref())
            .and_then(|(class, method)| crate::class_registry::lookup_bound_method(class, method))
            .map(|bound| ActiveObjectMethodIdentity {
                declaring_class: bound.declaring_class,
                method: bound.declaration.name,
                callable: bound.callable,
                registry_generation: bound.registry_generation,
            });
        Self {
            caller_class,
            active_protocol: caller_method.as_ref().and_then(ObjectProtocol::from_method),
            caller_method,
            active_method,
            selected_active_owner: None,
            selected_active_binding: None,
            selected_method_dispatch: crate::context::SelectedMethodDispatch::Virtual,
        }
    }
}

impl ObjectProtocol {
    pub fn from_method(method: &runmat_types::MethodName) -> Option<Self> {
        if crate::OBJECT_SUBSREF_METHOD.is(method) {
            Some(Self::Subsref)
        } else if crate::OBJECT_SUBSASGN_METHOD.is(method) {
            Some(Self::Subsasgn)
        } else if crate::OBJECT_NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.is(method) {
            Some(Self::NumArgumentsFromSubscript)
        } else if crate::OBJECT_END_METHOD.is(method) {
            Some(Self::End)
        } else {
            None
        }
    }

    pub fn method(self) -> runmat_types::MethodName {
        match self {
            Self::Subsref => crate::OBJECT_SUBSREF_METHOD.owned(),
            Self::Subsasgn => crate::OBJECT_SUBSASGN_METHOD.owned(),
            Self::NumArgumentsFromSubscript => {
                crate::OBJECT_NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.owned()
            }
            Self::End => crate::OBJECT_END_METHOD.owned(),
        }
    }

    pub const fn uses_standard_substruct(self) -> bool {
        matches!(self, Self::Subsref | Self::Subsasgn)
    }
}
