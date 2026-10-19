use super::*;
use crate::{context::RuntimeContext, execution::RuntimeExecutionService};

#[test]
fn persistent_revision_covers_every_canonical_mutation_without_aba() {
    let mut state = SessionVariableState::default();
    let initial = state.persistent_revision.clone();
    state.store_persistent_named("f", "x", Value::Num(1.0));
    let named = state.persistent_revision.clone();
    assert_ne!(named, initial);
    state.store_persistent_slot("f", 0, Value::Num(2.0));
    let slot = state.persistent_revision.clone();
    assert_ne!(slot, named);
    state.update_persistent_slot("f", 0, &Value::Num(3.0));
    let updated = state.persistent_revision.clone();
    assert_ne!(updated, slot);
    state.update_persistent_slot("f", 1, &Value::Num(4.0));
    assert_eq!(state.persistent_revision, updated);
    state = SessionVariableState::default();
    assert_ne!(state.persistent_revision, initial);
    assert_ne!(state.persistent_revision, updated);
}

#[test]
fn selected_persistent_authority_is_independent_of_ambient_context() {
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let first_authority = first.session_variable_authority();
    let second_authority = second.session_variable_authority();
    let first_initial = first_authority.persistent_revision().unwrap();
    let second_initial = second_authority.persistent_revision().unwrap();
    {
        let _ambient_second = second.enter();
        let wrong = first_authority
            .store_persistent_named_with_context(&second, "f", "x", Value::Num(8.0))
            .unwrap_err();
        assert_eq!(wrong.identifier(), Some("RunMat:Workspace:SessionOwner"));
        assert_eq!(
            first_authority.persistent_revision().unwrap(),
            first_initial
        );
        first_authority
            .store_persistent_named_with_context(&first, "f", "x", Value::Num(7.0))
            .unwrap();
        assert_eq!(
            first_authority.persistent_named_value("f", "x").unwrap(),
            Some(Value::Num(7.0))
        );
        assert_eq!(
            second_authority.persistent_named_value("f", "x").unwrap(),
            None
        );
        assert_ne!(
            first_authority.persistent_revision().unwrap(),
            first_initial
        );
        assert_eq!(
            second_authority.persistent_revision().unwrap(),
            second_initial
        );
    }
    drop(first);
    assert!(first_authority.persistent_revision().is_err());
}

#[test]
fn selected_function_clear_invalidates_even_unwritten_binding_without_touching_other_owners() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let authority = runtime.session_variable_authority();
    let first = authority.persistent_invalidation_revision("f").unwrap();
    let other = authority.persistent_invalidation_revision("g").unwrap();
    authority
        .store_persistent_named_and_slot("g", "x", 0, Value::Num(4.0))
        .unwrap();
    authority.clear_persistent_function("f").unwrap();
    assert_ne!(
        authority.persistent_invalidation_revision("f").unwrap(),
        first
    );
    assert_eq!(
        authority.persistent_invalidation_revision("g").unwrap(),
        other
    );
    assert_eq!(
        authority.persistent_named_value("g", "x").unwrap(),
        Some(Value::Num(4.0))
    );
    authority.clear_all_persistents().unwrap();
    assert_ne!(
        authority.persistent_invalidation_revision("g").unwrap(),
        other
    );
    assert_eq!(authority.persistent_named_value("g", "x").unwrap(), None);
}

#[test]
fn unknown_function_clears_do_not_accumulate_invalidation_names() {
    let mut state = SessionVariableState::default();
    let retained = state.persistent_invalidation_revision("active");
    for index in 0..1024 {
        state.clear_persistent_function(&format!("unknown_{index}"));
    }
    assert_eq!(state.persistent_invalidation_leases.len(), 1);
    let next = state.persistent_invalidation_revision("active");
    assert_eq!(retained, next);
    state.clear_persistent_function("active");
    assert_ne!(retained, state.persistent_invalidation_revision("active"));
}
