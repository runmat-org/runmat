//! MATLAB-compatible `linkaxes` builtin.

use runmat_builtins::{LINKAXES_DESCRIPTOR, LINKAXES_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::Value;

use super::plotting_error;
use super::state::{
    axes_metadata_snapshot, decode_axes_handle, figure_handle_exists, link_axes, FigureHandle,
    LinkAxesMode,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::op_common::handles::numeric_handle_from_integer;
use crate::builtins::plotting::style::value_as_string;

type AxesTarget = (FigureHandle, usize);
type ParsedLinkAxesArgs = (Vec<AxesTarget>, Option<LinkAxesMode>);

#[runmat_macros::runtime_builtin(
    name = "linkaxes",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::linkaxes"
)]
pub fn linkaxes_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &LINKAXES_DESCRIPTOR,
        &LINKAXES_ERROR_TOO_MANY_OUTPUTS,
        "linkaxes",
    )?;
    let (handles, mode) = parse_linkaxes_args(args)
        .map_err(|error| contextual_error(error, "RunMat:linkaxes:InvalidArgument"))?;
    link_axes(handles, mode).map_err(|err| {
        contextual_error(
            crate::builtins::plotting::plotting_error_with_source(
                "linkaxes",
                format!("linkaxes: {err}"),
                err,
            ),
            "RunMat:linkaxes:Internal",
        )
    })?;
    Ok(Value::output_list(vec![]))
}

fn parse_linkaxes_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<ParsedLinkAxesArgs> {
    match args.len() {
        1 => Ok((axes_handles_from_value(&args[0])?, Some(LinkAxesMode::XY))),
        2 => Ok((
            axes_handles_from_value(&args[0])?,
            link_mode_from_value(&args[1])?,
        )),
        _ => Err(plotting_error(
            "linkaxes",
            "linkaxes: expected axes handles and optional option",
        )),
    }
}

fn link_mode_from_value(value: &Value) -> crate::BuiltinResult<Option<LinkAxesMode>> {
    let option = value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "linkaxes",
                source.to_string(),
                source,
            )
        })?
        .ok_or_else(|| plotting_error("linkaxes", "linkaxes: option must be a string"))?;
    match option.trim().to_ascii_lowercase().as_str() {
        "x" => Ok(Some(LinkAxesMode::X)),
        "y" => Ok(Some(LinkAxesMode::Y)),
        "xy" => Ok(Some(LinkAxesMode::XY)),
        "off" => Ok(None),
        _ => Err(plotting_error(
            "linkaxes",
            "linkaxes: option must be 'x', 'y', 'xy', or 'off'",
        )),
    }
}

fn axes_handles_from_value(value: &Value) -> crate::BuiltinResult<Vec<(FigureHandle, usize)>> {
    let raw = match value {
        Value::Num(handle) => vec![*handle],
        Value::Int(handle) => vec![numeric_handle_from_integer(handle).ok_or_else(|| {
            plotting_error("linkaxes", "linkaxes: axes must be valid axes handles")
        })?],
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                (0..storage.len())
                    .map(|index| {
                        storage
                            .value_at(index)
                            .and_then(|value| numeric_handle_from_integer(&value))
                            .ok_or_else(|| {
                                plotting_error(
                                    "linkaxes",
                                    "linkaxes: axes must be valid axes handles",
                                )
                            })
                    })
                    .collect::<crate::BuiltinResult<Vec<_>>>()?
            } else {
                tensor_utils::tensor_values_f64(tensor)
            }
        }
        _ => {
            return Err(plotting_error(
                "linkaxes",
                "linkaxes: axes must be an axes handle or numeric array of axes handles",
            ))
        }
    };

    let mut axes = Vec::with_capacity(raw.len());
    for handle in raw {
        let (figure, axes_index) = decode_axes_handle(handle)
            .map_err(|_| plotting_error("linkaxes", "linkaxes: axes must be valid axes handles"))?;
        if !figure_handle_exists(figure) {
            return Err(plotting_error("linkaxes", "linkaxes: invalid axes handle"));
        }
        axes_metadata_snapshot(figure, axes_index).map_err(|err| {
            crate::builtins::plotting::plotting_error_with_source(
                "linkaxes",
                format!("linkaxes: {err}"),
                err,
            )
        })?;
        axes.push((figure, axes_index));
    }
    Ok(axes)
}

fn contextual_error(mut error: crate::RuntimeError, identifier: &str) -> crate::RuntimeError {
    if error.identifier.is_none() {
        error.identifier = Some(identifier.into());
    }
    error.context = error.context.with_builtin("linkaxes");
    error
}
#[cfg(test)]
#[path = "linkaxes/tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) fn linkaxes_builtin_source(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    linkaxes_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?)
}
