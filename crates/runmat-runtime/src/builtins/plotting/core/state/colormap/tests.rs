use super::*;

#[test]
fn figure_palette_updates_direct_image_limits_for_each_numeric_storage() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    for axes in 0..2 {
        let surface = runmat_plot::plots::SurfacePlot::new(
            vec![1., 2.],
            vec![1., 2.],
            vec![vec![1., 2.], vec![3., 4.]],
        )
        .unwrap();
        figure.add_surface_plot_on_axes(surface, axes);
    }
    let handle = import_figure(figure);
    register_image_handle(
        handle,
        0,
        0,
        Some(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap()),
        "direct",
    );
    let integer =
        Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1; 4]), vec![2, 2]).unwrap();
    register_image_handle(handle, 1, 1, Some(integer), "direct");
    set_colormap_for_figure_with_length(handle, ColorMap::Hot, 9).unwrap();
    let figure = clone_figure(handle).unwrap();
    for (index, integer) in [(0, false), (1, true)] {
        let Some(PlotElement::Surface(surface)) = figure.plots().nth(index) else {
            panic!("image");
        };
        assert_eq!(
            surface.color_limits,
            Some(crate::builtins::plotting::image::direct_colormap_limits(
                integer, 9
            ))
        );
    }
}

#[test]
fn figure_palette_and_axes_override_preserve_selection_and_reset_inheritance() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    let target = import_figure(figure);
    select_axes_for_figure(target, 1).unwrap();
    let other = import_figure(Figure::new());
    set_colormap_for_figure_with_length(target, ColorMap::Hot, 9).unwrap();
    set_colormap_for_axes_with_length(target, 0, ColorMap::Jet, 3).unwrap();
    assert_eq!(current_figure_handle(), other);
    assert_eq!(clone_figure(target).unwrap().active_axes_index, 1);
    assert_eq!(colormap_length_for_axes(target, 0), 3);
    assert_eq!(colormap_length_for_axes(target, 1), 9);
    let (map, length) = figure_colormap_snapshot(target).unwrap();
    assert!(matches!(map, ColorMap::Hot));
    assert_eq!(length, 9);
    clear_axes_target(target, 0, true).unwrap();
    assert_eq!(colormap_length_for_axes(target, 0), 9);
    assert!(matches!(
        axes_metadata_snapshot(target, 0).unwrap().colormap,
        ColorMap::Hot
    ));
    assert_eq!(current_figure_handle(), other);
}

#[test]
fn invalid_palette_target_does_not_create_or_select_a_figure() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    let current = current_axes_state();
    let invalid = FigureHandle::from(900_001);
    assert!(set_colormap_for_figure_with_length(invalid, ColorMap::Hot, 9).is_err());
    assert!(clone_figure(invalid).is_none());
    assert_eq!(current_figure_handle(), current.handle);
    assert!(set_colormap_for_axes_with_length(current.handle, 50, ColorMap::Hot, 9).is_err());
    assert_eq!(colormap_length_for_axes(current.handle, 0), 256);
}
