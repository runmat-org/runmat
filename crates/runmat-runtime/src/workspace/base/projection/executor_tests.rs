use super::*;
use runmat_value::Tensor;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn live_capture_canonicalizes_mixed_legacy_names_without_publishing() {
    use crate::context::{RuntimeContext, RuntimeWorkspaceService};

    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    context
        .session_variable_authority()
        .store_global_with_context(&context, "target", Value::Num(3.0))
        .unwrap();
    let owner = BaseWorkspaceOwner::new();
    let mut seed = WorkspaceDelta::new();
    seed.assign("m", Value::Num(2.0)).unwrap();
    owner.apply(seed).unwrap();

    let mut projection = owner.begin_projection().unwrap();
    projection.assign("d", Value::Num(4.0)).unwrap();
    projection.ordered_view_mut().unwrap().entries_mut().insert(
        WorkspaceEntryName::Owned("a".into()),
        RootedWorkspaceEntry::new(Value::Num(1.0)).unwrap(),
    );
    projection
        .ordered_view_mut()
        .unwrap()
        .global_bindings_mut()
        .insert(
            "b".into(),
            GlobalBindingTarget::Named(WorkspaceName::unbounded("target")),
        );
    projection
        .ordered_view_mut()
        .unwrap()
        .global_bindings_mut()
        .insert("z".into(), GlobalBindingTarget::LegacySlot(99));
    // The managed pending assignment shadows a stale legacy namespace row.
    projection
        .ordered_view_mut()
        .unwrap()
        .global_bindings_mut()
        .insert("d".into(), GlobalBindingTarget::LegacySlot(99));

    let service =
        super::super::BaseWorkspaceProjectionService::new_for_context(projection, &context);
    let revision = service.revision().unwrap();
    let captured = service
        .capture_workspace_generation_with_context(&context)
        .unwrap()
        .unwrap();
    assert_eq!(
        captured.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        vec!["a", "b", "d", "m"]
    );
    assert_eq!(captured.get("b"), Some(&Value::Num(3.0)));
    assert_eq!(captured.get("z"), None);
    assert_eq!(captured.global_target("d"), None);
    assert_eq!(
        captured.global_bindings().collect::<Vec<_>>(),
        vec![
            (
                "b",
                &GlobalBindingTarget::Named(WorkspaceName::unbounded("target"))
            ),
            ("z", &GlobalBindingTarget::LegacySlot(99)),
        ]
    );
    assert_eq!(service.revision().unwrap(), revision);
    assert!(service.has_pending_changes().unwrap());
    assert_eq!(owner.lookup("d").unwrap(), None);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn repeated_source_stores_retain_only_latest_bindings_before_publication() {
    let owner = BaseWorkspaceOwner::new();
    let mut projection = owner.begin_projection().unwrap();
    for iteration in 1..=64 {
        let value = f64::from(iteration);
        projection
            .assign_from_executor("t", Value::Num(value))
            .unwrap();
        projection
            .assign_from_executor(
                "Z",
                Value::Tensor(Tensor::new(vec![value, -value], vec![2, 1]).unwrap()),
            )
            .unwrap();
        projection
            .assign_from_executor(
                "S",
                Value::Tensor(Tensor::new(vec![value + 1.0, value + 2.0], vec![2, 1]).unwrap()),
            )
            .unwrap();
        assert_eq!(projection.ordered_view().unwrap().pending().len(), 3);
        assert_eq!(projection.len(), 3);
        assert_eq!(projection.lookup("t"), Some(&Value::Num(value)));
    }
    assert_eq!(owner.lookup("t").unwrap(), None);
    let commit = projection.publish().unwrap();
    assert_eq!(commit.change_count(), 3);
    assert_eq!(owner.lookup("t").unwrap(), Some(Value::Num(64.0)));
    let Some(Value::Tensor(noise)) = owner.lookup("Z").unwrap() else {
        panic!("latest noise")
    };
    assert_eq!(noise.as_f64_slice().unwrap(), &[64.0, -64.0]);
    let Some(Value::Tensor(state)) = owner.lookup("S").unwrap() else {
        panic!("latest state")
    };
    assert_eq!(state.as_f64_slice().unwrap(), &[65.0, 66.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn executor_coalescing_preserves_ordered_assignment_remove_and_clear() {
    let owner = BaseWorkspaceOwner::new();
    let mut projection = owner.begin_projection().unwrap();
    projection.assign("x", Value::Num(1.0)).unwrap();
    projection
        .assign_from_executor("x", Value::Num(2.0))
        .unwrap();
    projection
        .assign_from_executor("x", Value::Num(3.0))
        .unwrap();
    assert_eq!(
        projection.ordered_view().unwrap().pending().len(),
        2,
        "explicit ordered value remains"
    );
    projection.assign("x", Value::Num(4.0)).unwrap();
    projection
        .assign_from_executor("x", Value::Num(5.0))
        .unwrap();
    projection
        .assign_from_executor("x", Value::Num(6.0))
        .unwrap();
    assert_eq!(
        projection.ordered_view().unwrap().pending().len(),
        4,
        "dynamic assignment seals the executor suffix"
    );
    projection.remove("x").unwrap();
    projection
        .assign_from_executor("x", Value::Num(7.0))
        .unwrap();
    assert_eq!(
        projection.ordered_view().unwrap().pending().len(),
        6,
        "remove remains ordered"
    );
    projection.clear().unwrap();
    projection
        .assign_from_executor("x", Value::Num(8.0))
        .unwrap();
    projection
        .assign_from_executor("x", Value::Num(9.0))
        .unwrap();
    assert_eq!(
        projection.ordered_view().unwrap().pending().len(),
        8,
        "clear remains ordered"
    );
    let commit = projection.publish().unwrap();
    assert_eq!(commit.change_count(), 2);
    assert_eq!(owner.lookup("x").unwrap(), Some(Value::Num(9.0)));
}
