use super::*;
use runmat_plot::plots::LinePlot;
use runmat_value::Value;

#[test]
fn direction_property_and_axis_tokens_share_state_without_reordering_limits() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let axes = current_axes_state();
    set_axis_limits(Some((1., 4.)), Some((2., 8.)));
    crate::builtins::plotting::cmds::axis_builtin(vec![Value::String("ij".into())]).unwrap();
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
