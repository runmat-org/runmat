//! MATLAB-compatible `daspect` builtin.

use runmat_builtins::{DASPECT_ERROR_TOO_MANY_OUTPUTS, DASPECT_NUMERIC_AXES_ALIAS_EXTENSION};
use runmat_value::{Tensor, Value};

use super::op_common::axes_target::AxesTarget;
use super::properties::{data_aspect_ratio_mode_from_value, resolve_plot_handle, PlotHandle};
use super::state::{
    data_aspect_ratio_snapshot, data_aspect_ratio_snapshot_for_axes, set_data_aspect_ratio,
    set_data_aspect_ratio_for_axes,
};
use super::{plotting_error, plotting_error_with_source};
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "daspect";

#[runmat_macros::runtime_builtin(
    name = "daspect",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::daspect"
)]
pub fn daspect_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if requested_outputs() > 1 {
        return Err(
            crate::build_runtime_error(DASPECT_ERROR_TOO_MANY_OUTPUTS.message)
                .with_builtin(BUILTIN_NAME)
                .with_identifier("RunMat:daspect:TooManyOutputs")
                .build(),
        );
    }
    daspect_state(args).map_err(|mut error| {
        if error.identifier.is_none() {
            error.identifier = Some("RunMat:daspect:InvalidArgument".into());
        }
        error.context = error.context.with_builtin(BUILTIN_NAME);
        error
    })
}

fn daspect_state(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (target, args) = split_optional_axes_target(args)?;
    match args.len() {
        0 => query_ratio(target),
        1 => {
            let value = &args[0];
            if let Some(text) = value_as_string(value).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "daspect",
                    source.to_string(),
                    source,
                )
            })? {
                match text.trim().to_ascii_lowercase().as_str() {
                    "mode" => return query_mode(target),
                    "auto" | "manual" => {
                        ensure_setter_has_no_output()?;
                        let mode = data_aspect_ratio_mode_from_value(value, BUILTIN_NAME)?;
                        let ratio = current_ratio(target)?;
                        set_ratio_and_mode(target, ratio, mode)?;
                        return Ok(Value::output_list(Vec::new()));
                    }
                    _ => {}
                }
            }
            ensure_setter_has_no_output()?;
            let ratio = ratio_from_value(value)?;
            set_ratio_and_mode(target, ratio, "manual")?;
            Ok(Value::output_list(Vec::new()))
        }
        _ => Err(daspect_err("expected zero or one data-aspect argument")),
    }
}

fn requested_outputs() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn ensure_setter_has_no_output() -> BuiltinResult<()> {
    if requested_outputs() == 0 {
        Ok(())
    } else {
        Err(daspect_err("setter forms do not return a value"))
    }
}

fn split_optional_axes_target(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(AxesTarget, crate::call::BuiltinArguments)> {
    if let Some(first) = args.first() {
        if let Some(alias) = exact_numeric_axes_alias(first)? {
            if let Ok(PlotHandle::Axes(handle, axes_index)) =
                resolve_plot_handle(&Value::Num(alias), BUILTIN_NAME)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &DASPECT_NUMERIC_AXES_ALIAS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                return Ok((Some((handle, axes_index)), args.into_range(1..)?));
            }
        }
    }
    Ok((None, args))
}

fn exact_numeric_axes_alias(value: &Value) -> BuiltinResult<Option<f64>> {
    let candidate = match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => super::op_common::handles::numeric_handle_from_integer(value),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                super::op_common::handles::numeric_handle_from_integer(
                    &storage.value_at(0).expect("scalar integer storage"),
                )
            } else {
                Some(crate::builtins::common::tensor::tensor_value_f64(tensor, 0))
            }
        }
        _ => return Ok(None),
    };
    let value =
        candidate.ok_or_else(|| daspect_err("axes handle must be exactly representable"))?;
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return Err(daspect_err(
            "axes handle alias must be a positive integer scalar",
        ));
    }
    Ok(Some(value))
}

fn ratio_from_value(value: &Value) -> BuiltinResult<[f64; 3]> {
    let Value::Tensor(tensor) = value else {
        return Err(daspect_err(
            "DataAspectRatio must be a host row or column numeric vector",
        ));
    };
    let is_vector = matches!(tensor.shape(), [3] | [1, 3] | [3, 1]);
    if !is_vector {
        return Err(daspect_err(
            "DataAspectRatio must be a 3-element row or column vector",
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        let mut out = [0.0; 3];
        for (index, value) in storage.exact_values().iter().enumerate() {
            if value.try_to_u64().is_none_or(|value| value == 0) {
                return Err(daspect_err("DataAspectRatio values must be positive"));
            }
            out[index] = value.to_f64();
        }
        return Ok(out);
    }
    let values = crate::builtins::common::tensor::tensor_values_f64(tensor);
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(daspect_err(
            "DataAspectRatio values must be positive and finite",
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn query_ratio(target: AxesTarget) -> BuiltinResult<Value> {
    Ok(ratio_value(current_ratio(target)?))
}

fn query_mode(target: AxesTarget) -> BuiltinResult<Value> {
    let (_, mode) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(Value::string(mode))
}

fn current_ratio(target: AxesTarget) -> BuiltinResult<[f64; 3]> {
    let (ratio, _) = match target {
        Some((handle, axes_index)) => data_aspect_ratio_snapshot_for_axes(handle, axes_index)
            .map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })?,
        None => data_aspect_ratio_snapshot(),
    };
    Ok(ratio)
}

fn set_ratio_and_mode(target: AxesTarget, ratio: [f64; 3], mode: &str) -> BuiltinResult<()> {
    match target {
        Some((handle, axes_index)) => {
            set_data_aspect_ratio_for_axes(handle, axes_index, ratio, mode).map_err(|err| {
                plotting_error_with_source(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}"), err)
            })
        }
        None => {
            set_data_aspect_ratio(ratio, mode);
            Ok(())
        }
    }
}

fn ratio_value(ratio: [f64; 3]) -> Value {
    Value::Tensor(Tensor::new(ratio.to_vec(), vec![1, 3]).expect("aspect-ratio row vector"))
}

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) if chars.rows == 1 => {
            Some(runmat_value::decode_utf16_code_units(chars.data())?)
        }
        _ => None,
    })
}

fn daspect_err(detail: impl AsRef<str>) -> crate::RuntimeError {
    plotting_error(BUILTIN_NAME, detail.as_ref().to_string())
}

#[cfg(test)]
#[path = "daspect/tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) fn daspect_builtin_source(args: Vec<Value>) -> BuiltinResult<Value> {
    daspect_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?)
}
