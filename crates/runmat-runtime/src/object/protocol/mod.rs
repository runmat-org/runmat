mod invocation;
mod path_execution;
mod receiver;
mod resolution;

#[cfg(test)]
mod tests;

pub use invocation::*;
pub use path_execution::*;
pub use receiver::*;
pub use resolution::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectProtocol {
    Subsref,
    Subsasgn,
    NumArgumentsFromSubscript,
    End,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectAccessContext {
    pub caller_class: Option<runmat_types::ClassIdentity>,
    pub caller_method: Option<runmat_types::MethodName>,
    pub active_protocol: Option<ObjectProtocol>,
    pub active_method: Option<ActiveObjectMethodIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveObjectMethodIdentity {
    pub declaring_class: runmat_types::ClassIdentity,
    pub method: runmat_types::MethodName,
    pub callable: runmat_types::CallableIdentity,
    pub registry_generation: u64,
}

impl ObjectAccessContext {
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
                    "executable identity does not match registered method '{}.{}'",
                    owner.declaring_class, owner.method.0
                ),
            ));
        }
        let active_method = (!owner.is_static).then(|| ActiveObjectMethodIdentity {
            declaring_class: bound.declaring_class.clone(),
            method: bound.declaration.name.clone(),
            callable: bound.callable.clone(),
            registry_generation: bound.registry_generation,
        });
        Ok(Self {
            caller_class: Some(owner.declaring_class.clone()),
            caller_method: Some(owner.method.clone()),
            active_protocol: active_method
                .as_ref()
                .and_then(|_| ObjectProtocol::from_method(&owner.method)),
            active_method,
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
