use super::*;

#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
#[test]
fn selected_native_scene_staging_defers_legacy_render_and_closes_exact_receiver() {
    use crate::context::{
        GraphicsFlushRequest, GraphicsSceneStaging, RuntimeContext, RuntimeGraphicsFlushOperation,
        RuntimeGraphicsPresentationService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Selected {
        staged: RefCell<Vec<GraphicsReceiverRef>>,
        retired: RefCell<Vec<GraphicsReceiverRef>>,
    }
    impl RuntimeGraphicsPresentationService for Selected {
        fn flush(
            &self,
            _: GraphicsFlushRequest,
        ) -> std::pin::Pin<Box<dyn RuntimeGraphicsFlushOperation>> {
            panic!("staging is not presentation")
        }
        fn stage_scene(
            &self,
            receiver: GraphicsReceiverRef,
        ) -> Result<GraphicsSceneStaging, RuntimeError> {
            assert!(graphics_receiver_valid(&receiver));
            self.staged.borrow_mut().push(receiver);
            Ok(GraphicsSceneStaging::DeferredToFlush)
        }
        fn retire_scene(&self, receiver: &GraphicsReceiverRef) {
            self.retired.borrow_mut().push(receiver.clone());
        }
    }
    let _guard = lock_plot_test_registry();
    let host = Rc::new(Selected {
        staged: RefCell::new(Vec::new()),
        retired: RefCell::new(Vec::new()),
    });
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_graphics_presentation(host.clone()),
        );
    let _selected = runtime.enter();
    let handle = new_figure_handle();
    let receiver = figure_graphics_receiver(handle).unwrap().unwrap();
    let figure = clone_figure(handle).unwrap();
    present_figure_update_with_options("plot", handle, figure, false, false).unwrap();
    assert!(host.staged.borrow()[0].shares_lease(&receiver));
    close_figure(Some(handle)).unwrap();
    assert!(host.retired.borrow()[0].shares_lease(&receiver));
    assert!(!graphics_receiver_valid(&receiver));
}

#[test]
fn implicit_axis_target_read_and_single_axis_updates_keep_one_current_owner() {
    let _guard = lock_plot_test_registry();
    reset_plot_state();
    let handle = current_figure_handle();
    assert!(!figure_handle_exists(handle));
    assert_eq!(current_axes_target_before_mutation(), (handle, 0));
    assert!(!figure_handle_exists(handle));

    set_current_x_limits(Some((1.0, 2.0)));
    assert!(figure_handle_exists(handle));
    set_current_y_limits(Some((3.0, 4.0)));
    let metadata = axes_metadata_snapshot(handle, 0).expect("current axes");
    assert_eq!(metadata.x_limits, Some((1.0, 2.0)));
    assert_eq!(metadata.y_limits, Some((3.0, 4.0)));
}

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
    #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
    {
        let receiver = figure_graphics_receiver(handle).unwrap().unwrap();
        assert!(!figure_scene_is_current(&receiver, &before));
        assert!(figure_scene_is_current(&receiver, &selected));
    }
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

    let _ = axes_metadata_snapshot(handle, axes_index).expect("axes query");
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

#[test]
fn selecting_axes_rotates_scene_but_future_style_cycle_does_not() {
    let _guard = lock_plot_test_registry();
    let handle = new_figure_handle();
    let (_, other_axes) = create_axes_for_figure(Some(handle)).expect("create first axes");
    create_axes_for_figure(Some(handle)).expect("create second axes");
    let before = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };

    select_axes_for_figure(handle, other_axes).expect("select axes");
    let selected = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(!before.shares_lease(&selected));

    // These counters and orders choose the next plot's style; they do not
    // change the already-captured Figure or its active axes.
    set_line_style_order_for_axes(other_axes, &[LineStyle::Solid]);
    let _ = next_line_style_for_axes(other_axes);
    let _ = line_color_for_target_axes_series_index(handle, other_axes, 0);
    let _ = next_line_color_for_axes(other_axes);
    let after_future_style = {
        let registry = registry();
        registry.figures[&handle].scene_lease.clone()
    };
    assert!(selected.shares_lease(&after_future_style));
}

#[test]
fn color_order_axes_setter_preserves_missing_figure_creation_before_invalid_axes() {
    let _guard = lock_plot_test_registry();
    let missing = {
        let registry = registry();
        let missing = registry.next_handle;
        assert!(!registry.figures.contains_key(&missing));
        missing
    };

    let error = set_color_order_for_axes(missing, usize::MAX, &[])
        .expect_err("invalid axes must still reject");
    assert!(matches!(error, FigureError::InvalidSubplotIndex { .. }));
    let created = {
        let registry = registry();
        registry.figures[&missing].scene_lease.clone()
    };
    assert_eq!(created.figure(), missing.as_u32());

    set_color_order_for_axes(missing, 0, &[Vec4::ONE]).expect("default axes accepts color order");
    let changed = {
        let registry = registry();
        registry.figures[&missing].scene_lease.clone()
    };
    assert!(!created.shares_lease(&changed));
}
