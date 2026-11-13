use super::super::test_support::setup;
use super::*;
use runmat_value::Value;

#[test]
fn shading_changes_existing_surfaces_only_on_current_axes() {
    let _guard = setup();
    use crate::builtins::plotting::state::*;
    use runmat_plot::plots::surface::ShadingMode;
    use runmat_plot::plots::{Figure, PlotElement, SurfacePlot};
    reset_plot_state();
    let mut figure = Figure::new();
    figure.set_subplot_grid(1, 2);
    for axes in 0..2 {
        figure.add_surface_plot_on_axes(
            SurfacePlot::new(vec![1., 2.], vec![1., 2.], vec![vec![1., 2.], vec![3., 4.]])
                .unwrap()
                .with_shading(ShadingMode::Flat),
            axes,
        );
    }
    let handle = import_figure(figure);
    select_axes_for_figure(handle, 1).unwrap();
    shading_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::String("interp".into())],
    ))
    .unwrap();
    let figure = clone_figure(handle).unwrap();
    for (index, plot) in figure.plots().enumerate() {
        let PlotElement::Surface(surface) = plot else {
            panic!("surface");
        };
        assert_eq!(
            surface.shading_mode,
            if index == 0 {
                ShadingMode::Flat
            } else {
                ShadingMode::Smooth
            }
        );
    }
}

#[test]
fn extra_arguments_do_not_create_figure() {
    let _guard = setup();
    crate::builtins::plotting::state::reset_plot_state();
    assert!(
        shading_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::String("flat".into()), Value::String("extra".into())]
        ))
        .is_err()
    );
    assert!(crate::builtins::plotting::state::current_figure_handle_if_exists().is_none());
}
