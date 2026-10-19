use super::*;
use crate::builtins::plotting::properties::{get_properties, resolve_plot_handle};
use runmat_plot::plots::{LinePlot, NumericPlotData};
use runmat_value::Value;

fn line(y: f64) -> LinePlot {
    LinePlot::new(vec![0., 1.], vec![y, y + 1.]).unwrap()
}

#[test]
fn grid_reclaims_overlay_before_indices_are_compacted() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    let overlay = figure.ensure_overlay_axes(0);
    let removed_index = figure.add_line_plot_on_axes(line(10.), overlay);
    let survivor_index = figure.add_line_plot_on_axes(line(20.), 0);
    let handle = import_figure(figure);
    let removed = register_line_handle(handle, overlay, removed_index);
    let survivor = register_line_handle(handle, 0, survivor_index);
    let bar = ensure_colorbar_for_axes(handle, overlay).unwrap();
    configure_subplot(1, 2, 1).unwrap();
    assert!(plot_child_handle_snapshot(removed).is_err());
    assert!(plot_child_handle_snapshot(bar).is_err());
    assert_eq!(
        plot_child_handle_snapshot(survivor).unwrap().plot_index(),
        Some(0)
    );
    assert_eq!(
        axes_metadata_snapshot(handle, 1).unwrap().overlay_parent,
        None
    );
    assert_eq!(current_axes_state().active_index, 1);
}

#[test]
fn targeted_clear_rebases_other_axes_handles_without_selection_drift() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let removed_index = figure.add_line_plot_on_axes(line(10.), 0);
    let survivor_index = figure.add_line_plot_on_axes(line(20.), 1);
    figure.set_active_axes_index(1);
    let handle = import_figure(figure);
    select_axes_for_figure(handle, 1).unwrap();
    let removed = register_line_handle(handle, 0, removed_index);
    let survivor = register_line_handle(handle, 1, survivor_index);
    let removed_receiver = plot_child_graphics_receiver(removed as u64)
        .unwrap()
        .expect("removed child");
    let survivor_receiver = plot_child_graphics_receiver(survivor as u64)
        .unwrap()
        .expect("surviving child");
    let bar = ensure_colorbar_for_axes(handle, 0).unwrap();
    clear_axes_target(handle, 0, false).unwrap();
    assert_eq!(current_axes_state().active_index, 1);
    assert_eq!(current_figure_handle(), handle);
    assert!(plot_child_handle_snapshot(removed).is_err());
    assert!(!graphics_receiver_valid(&removed_receiver));
    assert!(graphics_receiver_valid(&survivor_receiver));
    assert!(survivor_receiver.shares_lease(
        &plot_child_graphics_receiver(survivor as u64)
            .unwrap()
            .expect("same remapped child")
    ));
    assert_eq!(
        plot_child_handle_snapshot(survivor).unwrap().plot_index(),
        Some(0)
    );
    let target = resolve_plot_handle(&Value::Num(survivor), "get").unwrap();
    let Value::Tensor(y) = get_properties(target, Some("YData"), "get").unwrap() else {
        panic!("YData");
    };
    assert_eq!(y.materialize_f64(), vec![20., 21.]);
    assert_eq!(colorbar_handle_for_axes(handle, 0), Some(bar));
}

#[test]
fn reset_preserves_structural_axes_position_units_and_live_colorbar() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    figure.set_axes_position(0, [0.2, 0.3, 0.4, 0.5]);
    figure.set_axes_units(0, "pixels");
    figure.set_axes_limits(0, Some((10., 20.)), Some((30., 40.)));
    figure.set_axes_grid_enabled(0, false);
    let handle = import_figure(figure);
    select_axes_for_figure(handle, 1).unwrap();
    let bar = ensure_colorbar_for_axes(handle, 0).unwrap();
    clear_axes_target(handle, 0, true).unwrap();
    let meta = axes_metadata_snapshot(handle, 0).unwrap();
    assert_eq!(meta.position, [0.2, 0.3, 0.4, 0.5]);
    assert_eq!(meta.units, "pixels");
    assert_eq!(meta.x_limits, None);
    assert_eq!(meta.y_limits, None);
    assert!(meta.grid_enabled);
    assert!(meta.colorbar_enabled);
    assert_eq!(colorbar_handle_for_axes(handle, 0), Some(bar));
    assert_eq!(current_axes_state().active_index, 1);
    assert_eq!(clone_figure(handle).unwrap().axes_count(), 2);
}

#[test]
fn composite_indices_rebase_without_reconstructing_metadata() {
    let figure = FigureHandle::default();
    let mut stacked = PlotChildHandleState::StackedPlot(StackedPlotHandleState {
        figure,
        axes_indices: vec![1, 2],
        line_plot_indices: vec![2, 3],
        line_group_counts: vec![1, 1],
        line_labels: vec!["a".into(), "b".into()],
        x_data: vec![1.],
        y_data: vec![vec![2.], vec![3.]],
        x_source: None,
        y_sources: vec![],
        display_variables: vec!["first".into(), "second".into()],
        source_table: None,
        x_variable: vec![],
        combine_matching_names: false,
        x_label: "x label".into(),
        title: "retained".into(),
        appearance: Default::default(),
        visible: true,
        grid_visible: false,
        x_limits: Some((1., 4.)),
    });
    assert!(remap_indices(&mut stacked, &[None, None, Some(0), Some(1)]));
    let PlotChildHandleState::StackedPlot(stacked) = stacked else {
        unreachable!()
    };
    assert_eq!(stacked.line_plot_indices, vec![0, 1]);
    assert_eq!(stacked.axes_indices, vec![1, 2]);
    assert_eq!(stacked.line_group_counts, vec![1, 1]);
    assert_eq!(stacked.title, "retained");
    assert_eq!(stacked.y_data, vec![vec![2.], vec![3.]]);
    let mut text = PlotChildHandleState::TextScatter(TextScatterHandleState {
        figure,
        axes_index: 2,
        annotation_indices: vec![5, 6],
        marker_plot_index: Some(3),
        is_3d: false,
        x_data: NumericPlotData::from_f64(vec![1.], vec![1, 1]).unwrap(),
        y_data: NumericPlotData::from_f64(vec![2.], vec![1, 1]).unwrap(),
        z_data: None,
        text_data: vec!["keep".into()],
        text_density_percentage: 60.,
        max_text_length: 40,
        marker_color: TextScatterMarkerColor::Auto,
        marker_size: 6.,
        color_data: None,
        colors: vec![],
        visible: true,
        base_style: Default::default(),
    });
    assert!(remap_indices(&mut text, &[None, None, Some(0), Some(1)]));
    let PlotChildHandleState::TextScatter(text) = text else {
        unreachable!()
    };
    assert_eq!(text.marker_plot_index, Some(1));
    assert_eq!(text.annotation_indices, vec![5, 6]);
    assert_eq!(text.text_data, vec!["keep"]);
}

#[test]
fn replacement_builder_rebases_other_axes_before_appending_new_plot() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let first = figure.add_line_plot_on_axes(line(10.), 0);
    let second = figure.add_line_plot_on_axes(line(20.), 1);
    let handle = import_figure(figure);
    let removed = register_line_handle(handle, 0, first);
    let retained = register_line_handle(handle, 1, second);
    render_active_plot(
        "clear-test",
        PlotRenderOptions::default(),
        |figure, axes| {
            figure.add_line_plot_on_axes(line(30.), axes);
            Ok(())
        },
    )
    .unwrap();
    assert!(plot_child_handle_snapshot(removed).is_err());
    assert_eq!(
        plot_child_handle_snapshot(retained).unwrap().plot_index(),
        Some(0)
    );
}
