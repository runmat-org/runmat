use super::super::super::set_target::PropertyPresentation;
use super::*;

#[test]
fn async_get_preserves_host_numeric_source_and_unpolled_set_has_no_effect() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let before = x_data(handle);
    let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    let args = [
        Value::String("XData".into()),
        Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("replacement")),
    ];
    let future = set_properties_async(plot, &args, "set");
    drop(future);
    assert_eq!(x_data(handle), before);
}

#[test]
fn stale_snapshot_refuses_atomic_property_update() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let (child, _, revision) = plot_child_element_snapshot(handle).expect("snapshot");
    let PlotChildHandleState::Line(state) = child else {
        panic!("expected line child");
    };
    let figure = state.figure;
    let plot_index = state.plot_index;
    let receiver = crate::builtins::plotting::state::plot_child_graphics_receiver(handle as u64)
        .unwrap()
        .unwrap();
    let selected =
        crate::builtins::plotting::state::GraphicsPropertySelection::capture(receiver).unwrap();
    let authority = crate::builtins::plotting::state::PropertyMutationTarget::selected(selected);
    crate::builtins::plotting::state::update_plot_element(figure, plot_index, |_| {})
        .expect("intervening mutation");
    let mut called = false;
    let error = crate::builtins::plotting::state::update_plot_element_if_current(
        handle,
        figure,
        plot_index,
        revision,
        |_| called = true,
    )
    .expect_err("stale revision");
    assert!(matches!(
        error,
        crate::builtins::plotting::state::FigureError::StalePlotSnapshot
    ));
    assert!(!called);
    let mut selected_called = false;
    crate::builtins::plotting::state::update_plot_element_for_property(
        &authority,
        figure,
        plot_index,
        |_| selected_called = true,
    )
    .expect_err("an intervening ordinary mutation retires the selected scene");
    assert!(!selected_called);
}

#[test]
fn invalid_async_set_does_not_mutate_host_line() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let before = x_data(handle);
    let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    let args = [
        Value::String("XData".into()),
        Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("mismatch")),
    ];
    let error = block_on(set_properties_async(plot, &args, "set")).expect_err("mismatched lengths");
    assert!(error.message.contains("same number of elements"));
    assert_eq!(x_data(handle), before);
}

#[test]
fn invalid_coordinate_input_is_rejected_by_host_preflight() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let before = x_data(handle);
    let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    let args = [
        Value::String("XData".into()),
        Value::String("invalid".into()),
    ];
    let error = block_on(set_properties_async(plot, &args, "set"))
        .expect_err("host-invalid coordinate must fail");
    assert!(error.message.contains("XData must be numeric"));
    assert_eq!(x_data(handle), before);
}

#[test]
fn metadata_queries_and_style_only_updates_do_not_require_coordinate_materialization() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let (child, plot, _) = plot_child_element_snapshot(handle).expect("snapshot");
    let (PlotChildHandleState::Line(state), PlotElement::Line(line)) = (child, plot) else {
        panic!("expected line");
    };
    assert!(matches!(
        line_property_without_data(&state, Some("Type"), "get", &line).expect("metadata"),
        SourcePropertyRead::Ready(Value::String(value)) if value == "line"
    ));
    assert!(matches!(
        line_property_without_data(&state, Some("XData"), "get", &line)
            .expect("known coordinate property"),
        SourcePropertyRead::RequiresCoordinates
    ));
    assert!(line_property_without_data(&state, Some("NotAProperty"), "get", &line).is_err());

    let before = x_data(handle);
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    block_on(set_properties_async(
        target,
        &[Value::String("LineWidth".into()), Value::Num(3.0)],
        "set",
    ))
    .expect("style update");
    assert_eq!(x_data(handle), before);
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    assert_eq!(
        block_on(get_properties_async(target, Some("LineWidth"), "get")).expect("style"),
        Value::Num(3.0)
    );
    for make in [host_line as fn() -> f64, host_scatter, host_line3] {
        let id = make();
        let receiver = crate::builtins::plotting::state::plot_child_graphics_receiver(id as u64)
            .unwrap()
            .unwrap();
        let object = crate::builtins::plotting::state::graphics_receiver_object(receiver.clone())
            .unwrap()
            .unwrap();
        let Value::HandleObject(handle) = &object else {
            panic!("exact chart object")
        };
        assert!(crate::is_handle_valid(handle));
        let original = x_data(id);
        assert_eq!(
            block_on(crate::builtins::plotting::get::get_builtin_async(vec![
                object.clone(),
                Value::String("XData".into())
            ]))
            .unwrap(),
            original
        );
        block_on(crate::builtins::plotting::set::set_builtin_async(vec![
            object.clone(),
            Value::String("LineWidth".into()),
            Value::Num(4.0),
        ]))
        .unwrap();
        assert_eq!(
            block_on(crate::builtins::plotting::get::get_builtin_async(vec![
                object.clone(),
                Value::String("LineWidth".into())
            ]))
            .unwrap(),
            Value::Num(4.0)
        );
        let replacement = Tensor::new(vec![8.0, 9.0], vec![1, 2]).unwrap();
        block_on(crate::builtins::plotting::set::set_builtin_async(vec![
            object.clone(),
            Value::String("XData".into()),
            Value::Tensor(replacement.clone()),
        ]))
        .unwrap();
        assert_eq!(
            block_on(crate::builtins::plotting::get::get_builtin_async(vec![
                object.clone(),
                Value::String("XData".into())
            ]))
            .unwrap(),
            Value::Tensor(replacement)
        );
        let selected =
            crate::builtins::plotting::state::GraphicsPropertySelection::capture(receiver).unwrap();
        crate::builtins::plotting::state::close_figure(Some(selected.figure_handle())).unwrap();
        assert!(
            block_on(crate::builtins::plotting::get::get_builtin_async(vec![
                object,
                Value::String("Type".into())
            ]))
            .is_err()
        );
    }
}

#[test]
fn selected_figure_outcome_retains_exact_scene_and_axes_mutation_is_public() {
    use crate::builtins::plotting::state;
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let id = host_line();
    let (child, _, _) = plot_child_element_snapshot(id).unwrap();
    let figure = child.figure_axes().0;
    let (_, other_axes) = state::create_axes_for_figure(Some(figure)).unwrap();
    state::set_text_properties_for_axes_for_property(
        &state::PropertyMutationTarget::alias(),
        figure,
        other_axes,
        state::PlotObjectKind::Title,
        Some("x".repeat(4096)),
        None,
    )
    .unwrap();
    state::select_axes_for_figure(figure, 0).unwrap();
    let receiver = state::figure_graphics_receiver(figure).unwrap().unwrap();
    let selection = state::GraphicsPropertySelection::capture(receiver.clone()).unwrap();
    use super::super::super::figure_plan::PreparedFigureProperties;
    let original_axes = state::axes_graphics_receiver(figure, 0).unwrap().unwrap();
    let original_axes_object = state::graphics_receiver_object(original_axes.clone())
        .unwrap()
        .unwrap();
    let admitted = PreparedFigureProperties::new(
        figure,
        &[
            Value::String("Name".into()),
            Value::String("retained".into()),
            Value::String("CurrentAxes".into()),
            original_axes_object,
        ],
        "set",
    )
    .unwrap();
    let proposal = admitted.proposal(&selection, "set").unwrap();
    let budget = std::num::NonZeroUsize::new(proposal.retained_byte_upper_bound()).unwrap();
    let before = state::current_figure_revision(figure);
    for edits in [
        vec![
            Value::String("Name".into()),
            Value::String("y".repeat(4096)),
            Value::String("Name".into()),
            Value::String("retained".into()),
        ],
        vec![
            Value::String("CurrentAxes".into()),
            Value::Num(state::encode_axes_handle(figure, other_axes)),
            Value::String("CurrentAxes".into()),
            Value::Num(state::encode_axes_handle(figure, 0)),
        ],
    ] {
        let rejected = PreparedFigureProperties::new(figure, &edits, "set").unwrap();
        let proposal = rejected.proposal(&selection, "set").unwrap();
        assert!(matches!(
            proposal.validate_retained_budget(budget),
            Err(state::FigurePersistenceSnapshotError::BudgetExceeded { .. })
        ));
        assert_eq!(state::current_figure_revision(figure), before);
        selection.ensure_current().unwrap();
    }
    proposal.validate_retained_budget(budget).unwrap();
    let authority = state::PropertyMutationTarget::selected(selection);
    let outcome = admitted.apply(&authority, figure, "set").unwrap();
    assert!(state::graphics_receiver_valid(&original_axes));
    let PropertySetOutcome::Presentation(PropertyPresentation::Selected {
        selection,
        figure: retained,
    }) = outcome
    else {
        panic!("selected Figure requires real presentation")
    };
    assert!(selection.figure_receiver().shares_lease(&receiver));
    selection.ensure_current().unwrap();
    assert_eq!(retained.name.as_deref(), Some("retained"));
    let actual =
        crate::context::RuntimeGraphicsScene::from_property(&selection, &retained).unwrap();
    assert!(actual.retained_byte_upper_bound().unwrap() <= budget.get());
    let axes = state::axes_graphics_receiver(figure, 0).unwrap().unwrap();
    let object = state::graphics_receiver_object(axes).unwrap().unwrap();
    block_on(crate::builtins::plotting::set::set_builtin_async(vec![
        object.clone(),
        Value::String("Grid".into()),
        Value::Bool(true),
        Value::String("Box".into()),
        Value::Bool(false),
    ]))
    .unwrap();
    assert_eq!(
        block_on(crate::builtins::plotting::get::get_builtin_async(vec![
            object,
            Value::String("Grid".into())
        ]))
        .unwrap(),
        Value::Bool(true)
    );
    assert!(selection.ensure_current().is_err());
    assert_eq!(retained.name.as_deref(), Some("retained"));
    state::close_figure(Some(figure)).unwrap();
    assert!(!state::graphics_receiver_valid(&receiver));
}
