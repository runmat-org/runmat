//! Shared admitted color-limit controls for clim and its caxis spelling.

use super::limits::{
    limit_state_error, limit_value, parse_limit_command, prepare_limit_controls,
    validate_limit_control_count, LimitCommand,
};
use crate::builtins::plotting::state::{
    axis_displayed_color_limits_for_axes_for_property,
    set_axis_color_limits_preserving_selection_for_property,
};
use runmat_builtins::CLIM_INTEGER_LIMITS_EXTENSION;
use runmat_value::Value;

pub fn apply_color_limits(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let (target, controls) = prepare_limit_controls(runtime, builtin, args.view())?;
    validate_limit_control_count(builtin, controls)?;
    if let Some(Value::Tensor(tensor)) = controls.first() {
        if matches!(
            tensor.numeric_dtype(),
            runmat_value::NumericDType::I8
                | runmat_value::NumericDType::I16
                | runmat_value::NumericDType::I32
                | runmat_value::NumericDType::I64
                | runmat_value::NumericDType::U8
                | runmat_value::NumericDType::U16
                | runmat_value::NumericDType::U32
                | runmat_value::NumericDType::U64
        ) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLIM_INTEGER_LIMITS_EXTENSION,
                builtin,
            )?;
        }
    }
    if controls.len() == 1
        && crate::builtins::plotting::style::value_as_string(&controls[0])
            .map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    builtin,
                    source.to_string(),
                    source,
                )
            })?
            .is_some_and(|mode| mode.trim().eq_ignore_ascii_case("tight"))
    {
        return Err(crate::builtins::plotting::plotting_error(
            builtin,
            format!("{builtin}: unsupported mode `tight`"),
        ));
    }
    let command = parse_limit_command(builtin, controls, true)?;
    if matches!(command, LimitCommand::Freeze | LimitCommand::Set(_)) {
        super::axes_output::validate_setter_output(builtin)?;
    }
    let target = target.resolve(builtin)?;
    let (handle, index) = target.coordinates();
    let property = target.authority();
    match command {
        LimitCommand::Query => {
            let limits = axis_displayed_color_limits_for_axes_for_property(property, handle, index)
                .map_err(|err| limit_state_error(builtin, err))?;
            Ok(limit_value(limits))
        }
        LimitCommand::QueryMode | LimitCommand::QueryMethod | LimitCommand::SetMethod(_) => {
            Err(crate::builtins::plotting::plotting_error(
                builtin,
                format!("{builtin}: axis-limit mode or method is unsupported for color limits"),
            ))
        }
        LimitCommand::Freeze => {
            let limits = axis_displayed_color_limits_for_axes_for_property(property, handle, index)
                .map_err(|err| limit_state_error(builtin, err))?
                .ok_or_else(|| {
                    crate::builtins::plotting::plotting_error(
                        builtin,
                        format!("{builtin}: automatic color extent is unavailable for manual mode"),
                    )
                })?;
            set_axis_color_limits_preserving_selection_for_property(
                property,
                handle,
                index,
                Some(limits),
            )
            .map_err(|err| limit_state_error(builtin, err))?;
            Ok(limit_value(Some(limits)))
        }
        LimitCommand::Set(limits) => {
            set_axis_color_limits_preserving_selection_for_property(
                property, handle, index, limits,
            )
            .map_err(|err| limit_state_error(builtin, err))?;
            Ok(limit_value(limits))
        }
    }
}
