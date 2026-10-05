use super::*;

pub(super) fn mex(identifier: &str, message: &str) -> RuntimeError {
    crate::runtime_error::semantic_error(identifier, message)
}

pub(super) fn require_single_sequence(
    sequence: runmat_value::ValueSequence,
) -> Result<Value, RuntimeError> {
    let mut values = sequence.resolve(
        runmat_types::SequenceUse::RequireSingle,
        crate::sequence::SequenceResolutionContext::default(),
    )?;
    Ok(values.remove(0))
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
    caller_function_name: Option<&str>,
) -> bool {
    match access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => {
            caller_class_for_function(caller_function_name).as_ref() == Some(owner)
        }
        runmat_types::MemberAccess::Protected => caller_class_for_function(caller_function_name)
            .is_some_and(|caller_class| {
                crate::class_registry::is_class_or_subclass(&caller_class, owner)
            }),
    }
}
