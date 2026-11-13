//! Shared reference-line service used by the refline and lsline public identities.
mod admission;
mod errors;
mod fit;
mod plot;
#[cfg(test)]
mod provider_tests;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::apply_axes_target;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    append_active_plot, axes_metadata_snapshot, clone_figure, current_axes_state,
    register_line_handle, FigureHandle, PlotRenderOptions,
};
use crate::builtins::plotting::style::{
    marker_metadata_from_appearance, parse_line_style_arguments, value_as_f64,
    LineStyleParseOptions,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use admission::{
    coefficient_count, normalize_refline_coefficients, parse_refline_coefficients,
    split_optional_axes, ReflineLineSpec, ReflinePlan,
};
use errors::ReferenceLineRuntimeErrors;
use fit::least_squares_coefficients;
use plot::{least_squares_specs, line_from_spec, refline_x_span};
use runmat_builtins::{
    BuiltinErrorDescriptor, ReferenceLineOwner, REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
    REFLINE_INTEGER_COEFFICIENT_EXTENSION, REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
};
use runmat_plot::plots::{LinePlot, LineStyle, PlotElement};
use runmat_value::{Tensor, Value};

const NAME: &str = "refline";

pub(super) async fn run_reference_lines(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    owner: ReferenceLineOwner,
) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes(args)?;
    let coefficient_count = coefficient_count(args.view(), owner)?;
    let style = parse_line_style_arguments(
        args.view().range(coefficient_count..)?,
        &LineStyleParseOptions::generic(owner.name()),
    )
    .map_err(|err| {
        let message = err.message.clone();
        owner.invalid_with_source(message, err)
    })?;
    let coefficients =
        normalize_refline_coefficients(args.fork(), runtime.as_ref(), owner, coefficient_count)
            .await?;
    let (plan, remaining) = parse_refline_coefficients(coefficients.view(), owner)?;
    debug_assert!(remaining.is_empty());
    apply_axes_target(target, owner.name()).map_err(|err| {
        if matches!(owner, ReferenceLineOwner::Refline) && err.identifier().is_some() {
            return err;
        }
        let message = err.message.clone();
        owner.internal_with_source(message, err)
    })?;
    let axes = current_axes_state();
    let figure_handle = axes.handle;

    let specs = match plan {
        ReflinePlan::Explicit { slope, intercept } => vec![ReflineLineSpec {
            slope,
            intercept,
            x_span: refline_x_span(figure_handle, axes.active_index, owner)?,
        }],
        ReflinePlan::LeastSquares => {
            least_squares_specs(figure_handle, axes.active_index, owner).await?
        }
    };
    if specs.is_empty() {
        return Ok(Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).map_err(
            |err| owner.internal(format!("{}: {err}", owner.name())),
        )?));
    }

    let lines = specs
        .iter()
        .map(|spec| line_from_spec(*spec, &style, owner))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut lines = Some(lines);

    let plot_indices_slot = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_out = std::rc::Rc::clone(&plot_indices_slot);
    let render_result = append_active_plot(
        owner.name(),
        PlotRenderOptions::default(),
        move |figure, axes| {
            let lines = lines.take().ok_or_else(|| {
                owner.internal(format!("{}: lines already rendered", owner.name()))
            })?;
            let mut indices = plot_indices_out.borrow_mut();
            for line in lines {
                let index = figure.add_line_plot_on_axes(line, axes);
                indices.push((axes, index));
            }
            Ok(())
        },
    );

    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(err);
        }
    }

    let handles = plot_indices_slot
        .borrow()
        .iter()
        .map(|(axes_index, plot_index)| {
            register_line_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect::<Vec<_>>();
    if handles.is_empty() {
        return Err(owner.internal(format!("{}: line was not registered", owner.name())));
    }
    if handles.len() == 1 {
        return Ok(Value::Num(handles[0]));
    }

    let len = handles.len();
    Ok(Value::Tensor(Tensor::new(handles, vec![len, 1]).map_err(
        |err| owner.internal(format!("{}: {err}", owner.name())),
    )?))
}
