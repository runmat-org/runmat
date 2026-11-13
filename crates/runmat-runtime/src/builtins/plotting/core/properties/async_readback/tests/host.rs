use super::super::super::set_target::PropertyPresentation;
use super::*;

#[test]
fn scatter3_properties_preserve_sources_and_reject_partial_resize() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let vector = |values: Vec<f64>| Value::Tensor(Tensor::new(values, vec![1, 2]).unwrap());
    let id = block_on(crate::builtins::plotting::scatter3::scatter3_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            vector(vec![1.0, 2.0]),
            vector(vec![3.0, 4.0]),
            vector(vec![5.0, 6.0]),
            vector(vec![36.0, 49.0]),
            vector(vec![0.0, 1.0]),
        ]),
    ))
    .expect("scatter3");
    let read = |name: &str| {
        let handle = resolve_plot_handle(&Value::Num(id), "get").unwrap();
        block_on(get_properties_async(handle, Some(name), "get")).unwrap()
    };
    assert_eq!(read("CData"), vector(vec![0.0, 1.0]));
    let assert_sizes = |actual: Value, expected: &[f64]| {
        let Value::Tensor(actual) = actual else {
            panic!("SizeData vector")
        };
        let values = crate::builtins::common::tensor::tensor_values_f64(&actual);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
        }
    };
    assert_sizes(read("SizeData"), &[36.0, 49.0]);

    let replacement = |values: Vec<f64>| Value::Tensor(Tensor::new(values, vec![1, 3]).unwrap());
    // A 1x3 CData row is a uniform RGB color in the constructor contract;
    // Nx1 denotes three scalar colors for the resized three-point plot.
    let scalar_colors = Value::Tensor(Tensor::new(vec![0.1, 0.2, 0.3], vec![3, 1]).unwrap());
    let target = resolve_plot_handle(&Value::Num(id), "set").unwrap();
    block_on(set_properties_async(
        target,
        &[
            Value::String("XData".into()),
            replacement(vec![7.0, 8.0, 9.0]),
            Value::String("YData".into()),
            replacement(vec![10.0, 11.0, 12.0]),
            Value::String("ZData".into()),
            replacement(vec![13.0, 14.0, 15.0]),
            Value::String("CData".into()),
            scalar_colors,
            Value::String("SizeData".into()),
            replacement(vec![16.0, 25.0, 36.0]),
        ],
        "set",
    ))
    .expect("atomic scatter3 resize");
    assert_eq!(read("XData"), replacement(vec![7.0, 8.0, 9.0]));
    assert_eq!(read("CData"), replacement(vec![0.1, 0.2, 0.3]));
    assert_sizes(read("SizeData"), &[16.0, 25.0, 36.0]);

    let target = resolve_plot_handle(&Value::Num(id), "set").unwrap();
    block_on(set_properties_async(
        target,
        &[
            Value::String("CData".into()),
            Value::Tensor(Tensor::new(vec![0.2, 0.4, 0.6], vec![1, 3]).unwrap()),
        ],
        "set",
    ))
    .expect("uniform RGB CData");
    let Value::Tensor(rgb) = read("CData") else {
        panic!("RGB CData matrix")
    };
    assert_eq!(rgb.shape(), &[3, 3]);
    let expected = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6];
    for (actual, expected) in crate::builtins::common::tensor::tensor_values_f64(&rgb)
        .iter()
        .zip(expected)
    {
        assert!((actual - expected).abs() < 1e-6);
    }

    let before = [
        read("XData"),
        read("YData"),
        read("ZData"),
        read("CData"),
        read("SizeData"),
    ];
    let target = resolve_plot_handle(&Value::Num(id), "set").unwrap();
    assert!(block_on(set_properties_async(
        target,
        &[
            Value::String("XData".into()),
            replacement(vec![1.0, 2.0, 3.0]),
            Value::String("YData".into()),
            replacement(vec![4.0, 5.0, 6.0]),
            Value::String("ZData".into()),
            replacement(vec![7.0, 8.0, 9.0]),
            Value::String("CData".into()),
            vector(vec![0.0, 1.0]),
        ],
        "set",
    ))
    .is_err());
    assert_eq!(
        before,
        [
            read("XData"),
            read("YData"),
            read("ZData"),
            read("CData"),
            read("SizeData")
        ]
    );
    let target = resolve_plot_handle(&Value::Num(id), "set").unwrap();
    assert!(block_on(set_properties_async(
        target,
        &[Value::String("NotAProperty".into()), Value::Num(1.0)],
        "set",
    ))
    .is_err());
    assert_eq!(
        before,
        [
            read("XData"),
            read("YData"),
            read("ZData"),
            read("CData"),
            read("SizeData")
        ]
    );
}

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
            block_on(crate::builtins::plotting::get::get_builtin_async(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    object.clone(),
                    Value::String("XData".into())
                ])
            ))
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
            block_on(crate::builtins::plotting::get::get_builtin_async(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    object.clone(),
                    Value::String("LineWidth".into())
                ])
            ))
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
            block_on(crate::builtins::plotting::get::get_builtin_async(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    object.clone(),
                    Value::String("XData".into())
                ])
            ))
            .unwrap(),
            Value::Tensor(replacement)
        );
        let selected =
            crate::builtins::plotting::state::GraphicsPropertySelection::capture(receiver).unwrap();
        crate::builtins::plotting::state::close_figure(Some(selected.figure_handle())).unwrap();
        assert!(block_on(crate::builtins::plotting::get::get_builtin_async(
            crate::builtins::common::test_support::builtin_arguments(vec![
                object,
                Value::String("Type".into())
            ])
        ))
        .is_err());
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
        crate::call::FunctionArgumentView::from_slice(&[
            Value::String("Name".into()),
            Value::String("retained".into()),
            Value::String("CurrentAxes".into()),
            original_axes_object,
        ]),
        "set",
    )
    .unwrap();
    let proposal = admitted.proposal(&selection, "set").unwrap();
    let budget = std::num::NonZeroUsize::new(proposal.retained_byte_upper_bound()).unwrap();
    let before = state::current_figure_revision(figure);
    for edits in [
        vec![
            Value::String("Name".into()),
            Value::String("y".repeat(4096).into()),
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
        let rejected = PreparedFigureProperties::new(
            figure,
            crate::call::FunctionArgumentView::from_slice(&edits),
            "set",
        )
        .unwrap();
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
        block_on(crate::builtins::plotting::get::get_builtin_async(
            crate::builtins::common::test_support::builtin_arguments(vec![
                object,
                Value::String("Grid".into())
            ])
        ))
        .unwrap(),
        Value::Bool(true)
    );
    assert!(selection.ensure_current().is_err());
    assert_eq!(retained.name.as_deref(), Some("retained"));
    state::close_figure(Some(figure)).unwrap();
    assert!(!state::graphics_receiver_valid(&receiver));
}
