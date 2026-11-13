//! Lifecycle selection checks do not open windows or initialize a renderer.
use super::native::{render_in_mode, PlottingMode};
use super::FigureHandle;
use runmat_plot::plots::{Figure, LinePlot};
use std::sync::atomic::Ordering;

#[test]
fn hidden_figures_close_without_automatic_export_in_every_mode() {
    for mode in [
        PlottingMode::Auto,
        PlottingMode::Interactive,
        PlottingMode::Static,
    ] {
        for populated in [false, true] {
            let handle = FigureHandle::from(0xF286);
            let registration = runmat_plot::gui::lifecycle::register_handle(handle.as_u32())
                .expect("owned test window registration");
            let signal = registration.signal().expect("close signal");
            let mut figure = Figure::new();
            figure.set_name("name authored before visibility");
            figure.set_visible(false);
            figure.set_subplot_grid(1, 2);
            if populated {
                figure.add_line_plot(LinePlot::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap());
            }
            let result = render_in_mode(handle, figure, mode).expect("hidden lifecycle update");
            assert_eq!(result, format!("Figure {} is hidden", handle.as_u32()));
            assert!(
                signal.load(Ordering::SeqCst),
                "the actual owned window must close"
            );
            drop(registration);
        }
    }
}

#[test]
fn empty_static_figures_defer_automatic_export_during_construction() {
    let handle = FigureHandle::from(0xF287);
    for subplot in [false, true] {
        let mut figure = Figure::new();
        figure.set_name("visible metadata-only figure");
        if subplot {
            figure.set_subplot_grid(1, 2);
        }
        let result = render_in_mode(handle, figure, PlottingMode::Static)
            .expect("blank construction must not attempt a static export");
        assert_eq!(result, format!("Figure {} updated", handle.as_u32()));
    }
}
