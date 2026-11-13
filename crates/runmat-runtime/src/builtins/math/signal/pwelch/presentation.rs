//! Pre-effect Welch line preparation and figure publication.
use super::*;
use crate::builtins::plotting::state::{
    current_figure_handle, presentation_unavailable, register_line_handle, render_active_plot,
    select_axes_for_figure, with_render_cause, PlotRenderOptions,
};
use runmat_plot::plots::line::LinePlot;

pub(super) struct PreparedWelchPlot {
    plots: Vec<LinePlot>,
    parent: Option<(crate::builtins::plotting::state::FigureHandle, usize)>,
    scale: SpectrumScale,
    sampled: bool,
}

pub(super) async fn prepare_pwelch_plot(
    evaluation: &PwelchEvaluation,
    options: &PwelchOptions,
    parent: Option<(crate::builtins::plotting::state::FigureHandle, usize)>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PreparedWelchPlot> {
    let rows = evaluation.f.len();
    if evaluation.pxx.len() != checked_output_elements(rows, evaluation.columns)? {
        return Err(pwelch_error(&PWELCH_ERROR_INTERNAL));
    }
    let mut plots = Vec::new();
    plots
        .try_reserve_exact(evaluation.columns)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    for column in 0..evaluation.columns {
        let mut frequency = Vec::new();
        frequency.try_reserve_exact(rows).map_err(|cause| {
            window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause)
        })?;
        frequency.extend_from_slice(&evaluation.f);
        let mut decibels = Vec::new();
        decibels.try_reserve_exact(rows).map_err(|cause| {
            window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause)
        })?;
        let start = column * rows;
        for power in &evaluation.pxx[start..start + rows] {
            decibels.push(10.0 * power.log10());
        }
        cooperation.completed(rows as u64).await?;
        let plot = LinePlot::new(frequency, decibels)
            .map_err(|cause| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, cause))?
            .with_label(format!("Channel {}", column + 1));
        plots.push(plot);
    }

    Ok(PreparedWelchPlot {
        plots,
        parent,
        scale: options.scale,
        sampled: options.fs.is_some(),
    })
}

pub(super) fn publish_pwelch_plot(prepared: PreparedWelchPlot) -> BuiltinResult<()> {
    let PreparedWelchPlot {
        plots,
        parent,
        scale,
        sampled,
    } = prepared;
    if let Some((figure, axes)) = parent {
        select_axes_for_figure(figure, axes).map_err(|cause| {
            pwelch_control_error(
                &PWELCH_ERROR_INVALID_PARENT,
                crate::builtins::plotting::plotting_error_with_source(
                    BUILTIN_NAME,
                    format!("pwelch: invalid Parent target: {cause}"),
                    cause,
                ),
            )
        })?;
    }
    let figure = current_figure_handle();
    let opts = PlotRenderOptions {
        title: match scale {
            SpectrumScale::Psd => "Welch Power Spectral Density Estimate",
            SpectrumScale::Power => "Welch Power Spectrum",
        },
        x_label: if sampled {
            "Frequency (Hz)"
        } else {
            "Normalized Frequency (rad/sample)"
        },
        y_label: match scale {
            SpectrumScale::Psd => "Power/Frequency (dB)",
            SpectrumScale::Power => "Power (dB)",
        },
        grid: true,
        axis_equal: false,
    };
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(plots.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let handles = std::rc::Rc::new(std::cell::RefCell::new(handles));
    let captured = std::rc::Rc::clone(&handles);
    let presentation = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        for plot in plots {
            let index = figure.add_line_plot_on_axes(plot, axes);
            captured.borrow_mut().push((axes, index));
        }
        Ok(())
    });
    for &(axes, index) in handles.borrow().iter() {
        register_line_handle(figure, axes, index);
    }
    match presentation {
        Ok(_) => Ok(()),
        Err(error) if presentation_unavailable(&error) => Ok(()),
        Err(error) => Err(with_render_cause(
            pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, error.message()),
            error,
        )),
    }
}
