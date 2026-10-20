//! MATLAB-compatible `animatedline` builtin.

use runmat_builtins::{ANIMATEDLINE_DESCRIPTOR, ANIMATEDLINE_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{Line3Plot, LinePlot};
use runmat_value::{Tensor, Value};

use super::op_common::line_inputs::NumericInput;
use super::op_common::{apply_axes_target, AxesTarget};
use super::plot::build_line_plot_for_builtin;
use super::plot3::build_line3_plot_for_builtin;
use super::plotting_error;
use super::properties::{
    animated_line_maximum_from_value, resolve_plot_handle, set_properties, PlotHandle,
};
use super::state::{
    append_active_plot, current_axes_state, current_figure_handle,
    line_color_for_target_axes_series_index, register_animated_line_handle, PlotChildHandleState,
    PlotRenderOptions,
};
use super::style::{parse_line_style_args, value_as_string, LineAppearance, LineStyleParseOptions};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "animatedline";
const DEFAULT_MAXIMUM_NUM_POINTS: usize = 1_000_000;

#[runtime_builtin(
    name = "animatedline",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::animatedline"
)]
pub async fn animatedline_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ANIMATEDLINE_DESCRIPTOR,
        &ANIMATEDLINE_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    let resident_inputs: Vec<_> = args
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let result = construct_animatedline(args).await;
    let cleanup =
        crate::builtins::common::gpu_helpers::clear_input_residency(&resident_inputs, BUILTIN_NAME);
    crate::runtime_error::complete_with_cleanup(result, cleanup)
}

async fn construct_animatedline(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (leading_axes, args) = split_leading_axes_handle_allow_empty(args)
        .await
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?;
    let parsed = parse_animatedline_args(leading_axes, args)
        .await
        .map_err(|error| contextual_error(error, "RunMat:animatedline:Internal"))?;
    apply_axes_target(parsed.axes_target, BUILTIN_NAME)
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?;

    let figure_handle = current_figure_handle();
    let plot_index = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index);
    let axes_index = std::rc::Rc::new(std::cell::RefCell::new(None));
    let axes_index_slot = std::rc::Rc::clone(&axes_index);
    let mut object = Some(parsed.object);
    let render_result = append_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions::default(),
        move |figure, active_axes| {
            *axes_index_slot.borrow_mut() = Some(active_axes);
            let object = object
                .take()
                .expect("animatedline plot object consumed exactly once");
            let (index, is_3d) = match object {
                AnimatedLineObject::TwoD(plot) => {
                    (figure.add_line_plot_on_axes(plot, active_axes), false)
                }
                AnimatedLineObject::ThreeD(plot) => {
                    figure.set_axes_z_limits(active_axes, None);
                    (figure.add_line3_plot_on_axes(plot, active_axes), true)
                }
            };
            *plot_index_slot.borrow_mut() = Some((index, is_3d));
            Ok(())
        },
    );

    let axes_index = axes_index.borrow().unwrap_or(0);
    let (plot_index, is_3d) = plot_index.borrow().unwrap_or((0, false));
    let handle = register_animated_line_handle(
        figure_handle,
        axes_index,
        plot_index,
        is_3d,
        parsed.maximum_num_points,
    );

    if !parsed.property_pairs.is_empty() {
        let state = PlotChildHandleState::AnimatedLine(super::state::AnimatedLineHandleState {
            figure: figure_handle,
            axes_index,
            plot_index,
            is_3d,
            maximum_num_points: parsed.maximum_num_points,
        });
        set_properties(
            PlotHandle::PlotChild(handle, Box::new(state)),
            &parsed.property_pairs,
            BUILTIN_NAME,
        )
        .map_err(|error| contextual_error(error, "RunMat:animatedline:InvalidArgument"))?;
    }

    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(contextual_error(err, "RunMat:animatedline:Internal"));
        }
    }
    Ok(Value::Num(handle))
}

async fn split_leading_axes_handle_allow_empty(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(AxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(mut first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if matches!(&first, Value::GpuTensor(handle) if handle.shape().iter().try_fold(1usize, |n, d| n.checked_mul(*d)) == Some(1))
    {
        // A scalar may be an axes identity; resolving it needs its value, not
        // a shape-derived guess. Nonscalar coordinates retain their own path.
        first = materialize_control(first).await?;
    }
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, BUILTIN_NAME) {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}

fn animatedline_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    contextual_error(
        plotting_error(BUILTIN_NAME, detail.as_ref().to_string()),
        "RunMat:animatedline:InvalidArgument",
    )
}

#[cfg(test)]
#[path = "animatedline/tests.rs"]
mod tests;

fn contextual_error(mut error: crate::RuntimeError, identifier: &str) -> crate::RuntimeError {
    if error.identifier.is_none() {
        error.identifier = Some(identifier.into());
    }
    error.context = error.context.with_builtin(BUILTIN_NAME);
    error
}

#[path = "animatedline/arguments.rs"]
mod arguments;
#[cfg(test)]
use arguments::maximum_from_value;
use arguments::{materialize_control, parse_animatedline_args, AnimatedLineObject};
