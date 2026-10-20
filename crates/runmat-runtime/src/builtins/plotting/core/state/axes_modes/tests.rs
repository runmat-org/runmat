use super::*;
use runmat_plot::plots::LinePlot;
use runmat_value::Value;

#[test]
fn manual_three_dimensional_limits_capture_the_shared_display_box() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.add_line3_plot(
        runmat_plot::plots::Line3Plot::new(vec![0., 0.1], vec![0., 0.1], vec![0., 0.]).unwrap(),
    );
    figure.set_axis_equal(true);
    figure.set_axes_y_direction(0, YDirection::Reverse);
    let expected = runmat_plot::plots::figure::display_bounds_3d(
        figure.data_bounds_for_axes(0),
        figure.axes_metadata(0),
    );
    let handle = import_figure(figure);
    lock_axes_limits(handle, 0).unwrap();
    let metadata = axes_metadata_snapshot(handle, 0).unwrap();
    assert_eq!(
        metadata.x_limits,
        Some((expected.min.x as f64, expected.max.x as f64))
    );
    assert_eq!(
        metadata.y_limits,
        Some((expected.min.y as f64, expected.max.y as f64))
    );
    assert_eq!(
        metadata.z_limits,
        Some((expected.min.z as f64, expected.max.z as f64))
    );
    assert_eq!(metadata.y_direction, YDirection::Reverse);
}

#[test]
fn direction_property_and_axis_tokens_share_state_without_reordering_limits() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let axes = current_axes_state();
    set_axis_limits(Some((1., 4.)), Some((2., 8.)));
    crate::builtins::plotting::cmds::axis_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("ij".into())]),
    )
    .unwrap();
    let meta = axes_metadata_snapshot(axes.handle, axes.active_index).unwrap();
    assert_eq!(meta.y_limits, Some((2., 8.)));
    assert_eq!(meta.y_direction, YDirection::Reverse);
    let target =
        crate::builtins::plotting::properties::PlotHandle::Axes(axes.handle, axes.active_index);
    crate::builtins::plotting::properties::set_properties(
        target,
        &[Value::String("YDir".into()), Value::String("normal".into())],
        "set",
    )
    .unwrap();
    assert_eq!(
        axes_metadata_snapshot(axes.handle, axes.active_index)
            .unwrap()
            .y_direction,
        YDirection::Normal
    );
}

#[test]
fn manual_limits_remain_frozen_when_data_is_appended() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.add_line_plot(LinePlot::new(vec![1., 4.], vec![2., 8.]).unwrap());
    let handle = import_figure(figure);
    lock_axes_limits(handle, 0).unwrap();
    let before = axes_metadata_snapshot(handle, 0).unwrap();
    assert_eq!(before.x_limits, Some((1., 4.)));
    assert_eq!(before.y_limits, Some((2., 8.)));
    assert_eq!(before.z_limits, None);
    set_hold(HoldMode::On);
    render_active_plot(
        "manual-test",
        PlotRenderOptions::default(),
        |figure, axes| {
            figure.add_line_plot_on_axes(
                LinePlot::new(vec![-100., 100.], vec![-200., 200.]).unwrap(),
                axes,
            );
            Ok(())
        },
    )
    .unwrap();
    let after = axes_metadata_snapshot(handle, 0).unwrap();
    assert_eq!(before.x_limits, after.x_limits);
    assert_eq!(before.y_limits, after.y_limits);
}
