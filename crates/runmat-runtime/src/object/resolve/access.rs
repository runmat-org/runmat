use super::*;
use std::borrow::Cow;

#[derive(Clone, Copy)]
pub(super) struct MemberCaller<'a> {
    name: Option<&'a str>,
    frame: Option<&'a crate::object::protocol::ObjectAccessContext>,
}
impl<'a> MemberCaller<'a> {
    pub(super) fn named(name: Option<&'a str>) -> Self {
        Self { name, frame: None }
    }
    pub(super) fn frame(
        name: Option<&'a str>,
        frame: &'a crate::object::protocol::ObjectAccessContext,
    ) -> Self {
        Self {
            name,
            frame: Some(frame),
        }
    }
    pub(super) fn protocol_access(self) -> Cow<'a, crate::object::protocol::ObjectAccessContext> {
        match self.frame {
            Some(frame) => Cow::Borrowed(frame),
            None => Cow::Owned(
                crate::object::protocol::ObjectAccessContext::from_legacy_function_name(self.name),
            ),
        }
    }
    pub(super) fn class(self) -> Option<ClassIdentity> {
        match self.frame {
            Some(frame) => frame.caller_class.clone(),
            None => caller_class_for_function(self.name),
        }
    }
}

pub(super) fn require_single_rooted_sequence(
    sequence: RootedValueSequence,
) -> Result<RootedValue, RuntimeError> {
    sequence.try_into_single()
}

pub(super) fn mex(identifier: &str, message: &str) -> RuntimeError {
    crate::runtime_error::semantic_error(identifier, message)
}

pub(super) fn caller_class_for_function(
    caller_function_name: Option<&str>,
) -> Option<ClassIdentity> {
    caller_function_name
        .and_then(crate::class_registry::caller_method_for_function)
        .map(|(class, _)| class)
}

pub(super) fn access_permitted(
    owner: &ClassIdentity,
    access: &runmat_types::MemberAccess,
    caller: MemberCaller<'_>,
) -> bool {
    match access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => caller.class().as_ref() == Some(owner),
        runmat_types::MemberAccess::Protected => caller.class().is_some_and(|caller_class| {
            crate::class_registry::is_class_or_subclass(&caller_class, owner)
        }),
    }
}
