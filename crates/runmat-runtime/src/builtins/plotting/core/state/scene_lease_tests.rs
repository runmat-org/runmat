use super::*;

#[test]
fn central_mutation_helpers_replace_exact_scene_lease_after_admission() {
    let _guard = lock_plot_test_registry();
    let handle = new_figure_handle();
    let (_, axes_index) = create_axes_for_figure(Some(handle)).expect("create axes");
    let before = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };

    with_figure_mut(handle, |state| state.tag.push_str("changed")).expect("figure mutation");
    let selected = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!before.shares_lease(&selected));
    assert_eq!(before.figure(), handle.as_u32());

    let invalid = with_axes_target_mut(handle, usize::MAX, |_| ());
    assert!(matches!(
        invalid,
        Err(FigureError::InvalidSubplotIndex { .. })
    ));
    let unchanged = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(selected.shares_lease(&unchanged));

    with_axes_target_mut(handle, axes_index, |state| state.tag.push_str(" again"))
        .expect("axes mutation");
    let successor = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!selected.shares_lease(&successor));
    assert_eq!(successor.figure(), handle.as_u32());

    let _ = axis_limits_snapshot();
    let _ = current_hold_enabled();
    let _ = pan_state_snapshot(handle, None).expect("pan query");
    assert!(matches!(
        set_data_aspect_ratio_for_axes(handle, usize::MAX, [1.0; 3], "auto"),
        Err(FigureError::InvalidSubplotIndex { .. })
    ));
    let after_queries = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(successor.shares_lease(&after_queries));

    let (first, second) = {
        let mut registry = registry();
        let first = get_state_mut(&mut registry, handle).scene_lease.clone();
        let second = get_state_mut(&mut registry, handle).scene_lease.clone();
        (first, second)
    };
    assert!(!successor.shares_lease(&first));
    assert!(
        first.shares_lease(&second),
        "one successor per registry transaction"
    );
}

#[test]
fn interaction_mutation_and_figure_retirement_cannot_reuse_scene_identity() {
    let _guard = lock_plot_test_registry();
    let handle = new_figure_handle();
    let original = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    set_zoom_mode_for_figure(handle, ZoomModeCommand::On).expect("zoom state mutation");
    let selected = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!original.shares_lease(&selected));

    clear_figure(Some(handle)).expect("clear figure");
    let cleared = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!selected.shares_lease(&cleared));

    close_figure(Some(handle)).expect("close figure");
    select_figure(handle);
    let reused_number = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!cleared.shares_lease(&reused_number));
}

#[test]
fn vacant_figure_has_one_lease_through_compound_registry_mutation() {
    let _guard = lock_plot_test_registry();
    let mut registry = registry();
    let fresh = registry.next_handle;
    assert!(!registry.figures.contains_key(&fresh));
    let first = get_state_mut(&mut registry, fresh).scene_lease.clone();
    let second = get_state_mut(&mut registry, fresh).scene_lease.clone();
    assert!(first.shares_lease(&second));
    let retired = registry.figures.remove(&fresh);
    drop(registry);
    drop(retired);
}
