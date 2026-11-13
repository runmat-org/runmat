use super::*;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn exact_named_fixture_rejects_duplicate_identity() {
    let runtime = super::super::exact_fixture_runtime();
    assert!(install_exact_named_forwarders(&runtime, &[("", 1)]).is_err());
    assert!(install_exact_named_forwarders(&runtime, &[("a", 1), ("a", 2)]).is_err());
    assert!(install_exact_named_forwarders(&runtime, &[("a", 1), ("b", 1)]).is_err());
}

#[test]
fn exact_named_binding_rejects_wrong_and_ended_sessions_before_invocation() {
    let first = super::super::exact_fixture_runtime();
    let first_session = first.session_lease();
    let first_fixtures =
        install_exact_named_forwarders(&first, &[("shared_name", 17), ("dependency_name", 18)])
            .unwrap();
    let binding = first_fixtures.binding("shared_name");
    let owner = first_fixtures.owner(17);
    let dependency_owner = first_fixtures.owner(18);
    assert!(binding.authority.admits_local(&first_session));
    assert!(owner.has_handles());
    assert!(dependency_owner.has_handles());

    let second = super::super::exact_fixture_runtime();
    let _second_fixtures =
        install_exact_named_forwarders(&second, &[("shared_name", 17), ("dependency_name", 18)])
            .unwrap();
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&calls);
    let entry = second.enter();
    let _invoker = super::super::super::install_local_bound_function_invoker(Rc::new(
        move |bound_runtime, _, _, _, _lexical_environment| {
            observed.set(observed.get() + 1);
            crate::execution::admit_executable_sequence_with(
                &bound_runtime.clone(),
                move || async move {
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error())
                },
            )
        },
    ));
    drop(entry);

    let wrong = futures::executor::block_on(
        super::super::super::try_call_bound_function_in_context(&second, binding.clone(), &[], 0),
    )
    .expect("selected bound route")
    .expect_err("another live session must reject the binding");
    assert_eq!(wrong.identifier(), Some("RunMat:UnavailableBoundFunction"));
    assert_eq!(calls.get(), 0);

    drop(first_fixtures);
    drop(first);
    assert!(!binding.authority.admits_local(&first_session));
    let ended = futures::executor::block_on(
        super::super::super::try_call_bound_function_in_context(&second, binding.clone(), &[], 0),
    )
    .expect("selected bound route")
    .expect_err("an ended session must reject the binding");
    assert_eq!(ended.identifier(), Some("RunMat:UnavailableBoundFunction"));
    assert_eq!(calls.get(), 0);
    assert!(owner.has_handles());
    assert!(dependency_owner.has_handles());
    drop(binding);
    assert!(!owner.has_handles());
    assert!(!dependency_owner.has_handles());
}
