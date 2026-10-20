//! Coefficient form, class, shape and compatibility admission.
use super::*;

pub(super) fn coefficient_count(
    args: crate::call::FunctionArgumentView<'_>,
    owner: ReferenceLineOwner,
) -> BuiltinResult<usize> {
    if matches!(owner, ReferenceLineOwner::Lsline) {
        return if args.is_empty() {
            Ok(0)
        } else {
            Err(owner.invalid("lsline: expected zero inputs or one axes handle"))
        };
    }
    let Some(first) = args.first() else {
        return Ok(0);
    };
    let first_len = coefficient_element_count(first, owner)?;
    if matches!(first, Value::Tensor(_) | Value::GpuTensor(_)) && !matches!(first_len, 1 | 2) {
        return Err(owner.invalid("refline: coefficients must be scalar or a two-element vector"));
    }
    if first_len == 2 {
        Ok(1)
    } else if args.len() >= 2 {
        let second = &args[1];
        if matches!(second, Value::Tensor(_) | Value::GpuTensor(_))
            && coefficient_element_count(second, owner)? != 1
        {
            return Err(owner.invalid("refline: intercept must be scalar"));
        }
        Ok(2)
    } else {
        Err(owner
            .invalid("refline: expected coefficients as [slope intercept] or slope, intercept"))
    }
}

fn coefficient_element_count(value: &Value, owner: ReferenceLineOwner) -> BuiltinResult<usize> {
    if let Value::GpuTensor(handle) = value {
        if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
            return Err(owner.invalid("refline: coefficients must be real numeric values"));
        }
    }
    Ok(match value {
        Value::Tensor(tensor) => tensor_utils::tensor_element_len(tensor),
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| {
                owner.invalid(format!("{}: coefficient shape overflows", owner.name()))
            })?,
        _ => 1,
    })
}

pub(super) async fn normalize_refline_coefficients(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    owner: ReferenceLineOwner,
    coefficient_count: usize,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64, value_has_logical_class, value_has_native_integer_class,
    };
    // Compatibility admission retains its precedence, and every host-visible
    // coefficient is then checked before provider work begins.
    for value in args.iter().take(coefficient_count) {
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
        }
        if value_has_logical_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
        }
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &REFLINE_INTEGER_COEFFICIENT_EXTENSION,
                owner.name(),
            )?;
            if !matches!(value, Value::GpuTensor(_)) && !native_integer_value_is_exact_f64(value) {
                return Err(owner.invalid(format!(
                    "{}: integer coefficients must be exactly representable as double",
                    owner.name()
                )));
            }
        }
    }
    preflight_host_coefficients(args.view(), coefficient_count, owner)?;
    let coefficients = args.into_range(..coefficient_count)?;
    if !coefficients
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(coefficients);
    }
    coefficients
        .try_map_preserving_roots_async(coefficient_count, runtime, async |index| {
            let value = if matches!(coefficients[index], Value::GpuTensor(_)) {
                crate::dispatcher::gather_if_needed_async(&coefficients[index])
                    .await
                    .map_err(|err| {
                        let message = format!("{}: {err}", owner.name());
                        owner.invalid_with_source(message, err)
                    })?
            } else {
                coefficients[index].clone()
            };
            if value_has_native_integer_class(&value) && !native_integer_value_is_exact_f64(&value)
            {
                return Err(owner.invalid(format!(
                    "{}: integer coefficients must be exactly representable as double",
                    owner.name()
                )));
            }
            Ok(value)
        })
        .await
}

fn preflight_host_coefficients(
    args: crate::call::FunctionArgumentView<'_>,
    coefficient_count: usize,
    owner: ReferenceLineOwner,
) -> BuiltinResult<()> {
    if coefficient_count == 0 {
        return Ok(());
    }
    if !matches!(args[0], Value::GpuTensor(_)) && coefficient_pair(&args[0], owner)?.is_none() {
        finite_scalar(&args[0], "slope", owner)?;
    }
    if coefficient_count == 2 && !matches!(args[1], Value::GpuTensor(_)) {
        finite_scalar(&args[1], "intercept", owner)?;
    }
    Ok(())
}

pub(super) type AxesTarget = Option<(FigureHandle, usize)>;

#[derive(Clone, Copy)]
pub(super) enum ReflinePlan {
    Explicit { slope: f64, intercept: f64 },
    LeastSquares,
}

#[derive(Clone, Copy)]
pub(super) struct ReflineLineSpec {
    pub(super) slope: f64,
    pub(super) intercept: f64,
    pub(super) x_span: (f64, f64),
}

pub(super) fn split_optional_axes(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(AxesTarget, crate::call::BuiltinArguments)> {
    let Some(first) = args.first() else {
        return Ok((None, args));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(first, NAME) {
        return Ok((Some((handle, axes_index)), args.into_range(1..)?));
    }
    Ok((None, args))
}

pub(super) fn parse_refline_coefficients<'a>(
    args: crate::call::FunctionArgumentView<'a>,
    owner: ReferenceLineOwner,
) -> BuiltinResult<(ReflinePlan, crate::call::FunctionArgumentView<'a>)> {
    match args.split_first() {
        None => Ok((ReflinePlan::LeastSquares, args)),
        Some((coeffs, rest)) => {
            if let Some(pair) = coefficient_pair(coeffs, owner)? {
                return Ok((
                    ReflinePlan::Explicit {
                        slope: pair.0,
                        intercept: pair.1,
                    },
                    rest,
                ));
            }
            if rest.is_empty() {
                return Err(owner.invalid(format!(
                    "{}: expected coefficients as [slope intercept] or slope, intercept",
                    owner.name()
                )));
            }
            let slope = finite_scalar(coeffs, "slope", owner)?;
            let intercept = finite_scalar(&rest[0], "intercept", owner)?;
            Ok((ReflinePlan::Explicit { slope, intercept }, rest.range(1..)?))
        }
    }
}

fn coefficient_pair(value: &Value, owner: ReferenceLineOwner) -> BuiltinResult<Option<(f64, f64)>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor_utils::tensor_element_len(tensor) != 2 {
                return Ok(None);
            }
            let slope = tensor_utils::tensor_value_f64(tensor, 0);
            let intercept = tensor_utils::tensor_value_f64(tensor, 1);
            if !slope.is_finite() || !intercept.is_finite() {
                return Err(owner.invalid(format!(
                    "{}: coefficients must contain finite slope and intercept",
                    owner.name()
                )));
            }
            Ok(Some((slope, intercept)))
        }
        _ => Ok(None),
    }
}

fn finite_scalar(value: &Value, name: &str, owner: ReferenceLineOwner) -> BuiltinResult<f64> {
    let value = value_as_f64(value)
        .ok_or_else(|| owner.invalid(format!("{}: {name} must be numeric", owner.name())))?;
    if !value.is_finite() {
        return Err(owner.invalid(format!("{}: {name} must be finite", owner.name())));
    }
    Ok(value)
}
