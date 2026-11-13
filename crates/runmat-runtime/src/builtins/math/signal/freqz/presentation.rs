//! Zero-output frequency-response presentation. Numerical output forms never enter here.
use super::*;
use crate::builtins::plotting::state::{
    append_active_plot, prepare_plotyy_axes, presentation_unavailable, register_line_handle,
    with_render_cause, PlotRenderOptions,
};
use runmat_plot::plots::LinePlot;

pub(super) fn plot_response(eval: &FreqzEvaluation, sampled: bool) -> BuiltinResult<()> {
    // Construct both plots before touching figure state. An allocation or invalid
    // plot then cannot publish a partial response.
    let magnitude: Vec<f64> = eval
        .h
        .iter()
        .map(|value| 20.0 * value.norm().log10())
        .collect();
    let phase: Vec<f64> = eval.h.iter().map(|value| value.arg()).collect();
    let magnitude = LinePlot::new(eval.w.clone(), magnitude)
        .map_err(|cause| freqz_error_with_detail(&FREQZ_ERROR_PLOT, cause))?
        .with_label("Magnitude (dB)");
    let phase = LinePlot::new(eval.w.clone(), phase)
        .map_err(|cause| freqz_error_with_detail(&FREQZ_ERROR_PLOT, cause))?
        .with_label("Phase (rad)");

    let x_label = if sampled {
        "Frequency (Hz)"
    } else {
        "Frequency (rad/sample)"
    };
    let opts = PlotRenderOptions {
        title: "Frequency Response",
        x_label,
        y_label: "Magnitude (dB)",
        grid: true,
        axis_equal: false,
    };
    let (figure, left_axes, right_axes, _, _) = prepare_plotyy_axes()
        .map_err(|cause| freqz_error_with_detail(&FREQZ_ERROR_PLOT, cause.to_string()))?;
    let indices = std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(2)));
    let published = std::rc::Rc::clone(&indices);
    let result = append_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        debug_assert_eq!(axes, left_axes);
        figure.set_axes_xlabel(right_axes, x_label);
        figure.set_axes_ylabel(right_axes, "Phase (rad)");
        published.borrow_mut().push((
            left_axes,
            figure.add_line_plot_on_axes(magnitude, left_axes),
        ));
        published
            .borrow_mut()
            .push((right_axes, figure.add_line_plot_on_axes(phase, right_axes)));
        Ok(())
    });
    for &(axes, index) in indices.borrow().iter() {
        register_line_handle(figure, axes, index);
    }
    match result {
        Ok(_) => Ok(()),
        Err(error) if presentation_unavailable(&error) => Ok(()),
        Err(error) => Err(with_render_cause(
            freqz_error_with_detail(&FREQZ_ERROR_PLOT, error.message()),
            error,
        )),
    }
}
