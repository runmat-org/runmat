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
}
