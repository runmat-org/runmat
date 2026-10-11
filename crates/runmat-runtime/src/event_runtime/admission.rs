use super::{event_descriptor_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_types::{ClassIdentity, EventDeclaration, MemberName};

#[derive(Clone, Copy)]
pub(super) enum EventOperation {
    Listen,
    Notify,
}

impl EventOperation {
    fn builtin(self) -> &'static str {
        match self {
            Self::Listen => "addlistener",
            Self::Notify => "notify",
        }
    }

    fn access_kind(self) -> crate::class_registry::EventAccessKind {
        match self {
            Self::Listen => crate::class_registry::EventAccessKind::Listen,
            Self::Notify => crate::class_registry::EventAccessKind::Notify,
        }
    }

    fn unknown_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Listen => &runmat_builtins::ADDLISTENER_ERROR_EVENT_UNKNOWN,
            Self::Notify => &runmat_builtins::NOTIFY_ERROR_EVENT_UNKNOWN,
        }
    }

    fn access_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Listen => &runmat_builtins::ADDLISTENER_ERROR_EVENT_ACCESS,
            Self::Notify => &runmat_builtins::NOTIFY_ERROR_EVENT_ACCESS,
        }
    }

    fn changed_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Listen => &runmat_builtins::ADDLISTENER_ERROR_EVENT_CHANGED,
            Self::Notify => &runmat_builtins::NOTIFY_ERROR_EVENT_CHANGED,
        }
    }
}

pub(super) struct EventAdmission {
    receiver_class: ClassIdentity,
    name: MemberName,
    declaration: EventDeclaration,
    owner: ClassIdentity,
    caller_class: Option<ClassIdentity>,
    generation: u64,
    operation: EventOperation,
}

impl EventAdmission {
    pub(super) fn new(
        receiver_class: &ClassIdentity,
        event_name: &str,
        operation: EventOperation,
    ) -> Result<Self, RuntimeError> {
        let name = MemberName::from(event_name);
        let (declaration, owner) = crate::class_registry::lookup_event(receiver_class, &name)
            .ok_or_else(|| {
                event_descriptor_error(
                    operation.builtin(),
                    operation.unknown_descriptor(),
                    format!("{}: unknown event '{}'", operation.builtin(), event_name),
                )
            })?;
        let caller_class = crate::dispatcher::class_access_context();
        if !crate::class_registry::event_accessible_from(
            &declaration,
            &owner,
            caller_class.as_ref(),
            operation.access_kind(),
        ) {
            return Err(event_descriptor_error(
                operation.builtin(),
                operation.access_descriptor(),
                format!(
                    "{}: event '{}' is not accessible",
                    operation.builtin(),
                    event_name
                ),
            ));
        }
        Ok(Self {
            receiver_class: receiver_class.clone(),
            name,
            declaration,
            owner,
            caller_class,
            generation: crate::class_registry::class_registry_generation(),
            operation,
        })
    }

    pub(super) fn recheck(&self, target: &runmat_value::HandleRef) -> Result<(), RuntimeError> {
        if !crate::is_handle_valid(target) || target.class_name != self.receiver_class {
            let descriptor = match self.operation {
                EventOperation::Listen => &runmat_builtins::ADDLISTENER_ERROR_TARGET,
                EventOperation::Notify => &runmat_builtins::NOTIFY_ERROR_TARGET,
            };
            return Err(event_descriptor_error(
                self.operation.builtin(),
                descriptor,
                format!(
                    "{}: event target is no longer valid",
                    self.operation.builtin()
                ),
            ));
        }
        if crate::class_registry::class_registry_generation() == self.generation {
            return Ok(());
        }
        let Some((current, owner)) =
            crate::class_registry::lookup_event(&self.receiver_class, &self.name)
        else {
            return Err(event_descriptor_error(
                self.operation.builtin(),
                self.operation.unknown_descriptor(),
                format!(
                    "{}: event '{}' is no longer declared",
                    self.operation.builtin(),
                    self.name
                ),
            ));
        };
        if !crate::class_registry::event_accessible_from(
            &current,
            &owner,
            self.caller_class.as_ref(),
            self.operation.access_kind(),
        ) {
            return Err(event_descriptor_error(
                self.operation.builtin(),
                self.operation.access_descriptor(),
                format!(
                    "{}: event '{}' is no longer accessible",
                    self.operation.builtin(),
                    self.name
                ),
            ));
        }
        if current != self.declaration || owner != self.owner {
            return Err(event_descriptor_error(
                self.operation.builtin(),
                self.operation.changed_descriptor(),
                format!(
                    "{}: event '{}' changed during dispatch",
                    self.operation.builtin(),
                    self.name
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use runmat_types::{standard, EventAccess, EventAttributes};
    use runmat_value::Value;

    fn event(name: &str, listen_access: EventAccess) -> EventDeclaration {
        EventDeclaration {
            name: MemberName::from(name),
            attributes: EventAttributes {
                listen_access,
                ..EventAttributes::default()
            },
            declared_attributes: Vec::new(),
            span: Default::default(),
        }
    }

    #[test]
    fn admission_resolves_inherited_events_and_rechecks_republication() {
        let parent = ClassIdentity::from("RuntimeEventAdmissionParentTest");
        let child = ClassIdentity::from("RuntimeEventAdmissionChildTest");
        let mut parent_events = std::collections::HashMap::new();
        parent_events.insert(
            MemberName::from("Changed"),
            event("Changed", EventAccess::Public),
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent.clone(),
            parent: Some(standard::HANDLE.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events: parent_events,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: child.clone(),
            parent: Some(parent.clone()),
            properties: Default::default(),
            methods: Default::default(),
            events: Default::default(),
        });
        assert!(EventAdmission::new(&child, "Missing", EventOperation::Notify).is_err());
        let admission = EventAdmission::new(&child, "Changed", EventOperation::Notify)
            .expect("inherited event");
        let target = runmat_value::HandleRef {
            class_name: child.clone(),
            target: runmat_gc::gc_allocate(Value::Object(runmat_value::ObjectInstance::new(child)))
                .expect("event target"),
            valid: true,
        };
        admission.recheck(&target).expect("unchanged event");
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent,
            parent: Some(standard::HANDLE.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events: Default::default(),
        });
        assert!(admission.recheck(&target).is_err());
    }

    #[test]
    fn private_listen_access_is_checked_before_registration() {
        let owner = ClassIdentity::from("RuntimeEventPrivateOwnerTest");
        let mut events = std::collections::HashMap::new();
        events.insert(
            MemberName::from("Secret"),
            event("Secret", EventAccess::Private),
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: owner.clone(),
            parent: Some(standard::HANDLE.owned()),
            properties: Default::default(),
            methods: Default::default(),
            events,
        });
        assert!(EventAdmission::new(&owner, "Secret", EventOperation::Listen).is_err());
        let _scope = crate::dispatcher::push_class_access_context(Some(owner.clone()));
        EventAdmission::new(&owner, "Secret", EventOperation::Listen)
            .expect("declaring class has listen access");
    }
}
