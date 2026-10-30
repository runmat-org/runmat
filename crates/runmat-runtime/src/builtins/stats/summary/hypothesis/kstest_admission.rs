//! Value-role, compatibility, output-count, and provider-boundary admission.
use super::*;
use std::borrow::Cow;
pub(super) fn is_kstest_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn kstest_cdf_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Cell(cell) if cell.data().len() == 1 => {
            kstest_cdf_contains_typed_integer(&cell.data()[0])
        }
        _ => is_kstest_typed_integer(value),
    }
}

pub(super) fn validate_kstest_cdf_role(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Tensor(_)
        | Value::String(_)
        | Value::CharArray(_)
        | Value::FunctionHandle(_) => Ok(()),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(kstest_invalid_argument(
                    "kstest: CDF must be a real numeric matrix",
                ));
            }
            let shape = handle.shape();
            if shape.len() != 2 || shape[1] != 2 || shape[0] < 2 {
                return Err(kstest_invalid_argument(
                    "kstest: CDF table must be an n-by-2 numeric matrix",
                ));
            }
            Ok(())
        }
        Value::Cell(cell) if cell.data().len() == 1 => validate_kstest_cdf_role(&cell.data()[0]),
        other => Err(kstest_invalid_argument(format!(
            "kstest: CDF must be a two-column numeric matrix or normcdf handle; probability distribution objects are not supported yet, got {other:?}"
        ))),
    }
}

pub(super) fn validate_kstest_alpha_role(value: &Value) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(array) => array.data.len() == 1,
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(kstest_invalid_argument(
                    "kstest: Alpha must be real numeric",
                ));
            }
            handle
                .shape()
                .iter()
                .try_fold(1usize, |len, dim| len.checked_mul(*dim))
                == Some(1)
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(kstest_invalid_argument("kstest: Alpha must be numeric"))
    }
}

pub(super) fn preflight_kstest_sample_metadata(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(kstest_invalid_argument(
                    "kstest: x must be a real numeric vector",
                ));
            }
            handle.shape()
        }
        _ => return Ok(()),
    };
    if !is_kstest_vector_shape(shape) {
        return Err(kstest_invalid_argument("kstest: x must be a vector"));
    }
    if shape.contains(&0) {
        return Err(kstest_invalid_argument(
            "kstest: x must contain at least one non-NaN value",
        ));
    }
    Ok(())
}

pub(super) fn ensure_kstest_sample_extension(x: &Value) -> BuiltinResult<()> {
    if is_kstest_typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KSTEST_INTEGER_SAMPLE_EXTENSION,
            KSTEST_NAME,
        )?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn ensure_kstest_extensions(x: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let plan = KstestOptionPlan::admit(crate::builtins::common::test_support::builtin_arguments(
        rest.to_vec(),
    ))?;
    ensure_kstest_sample_extension(x)?;
    plan.ensure_extensions(x)?;
    plan.preflight_known().map(|_| ())
}
pub(super) fn ensure_exact_kstest_integer_tensor(value: &Tensor, role: &str) -> BuiltinResult<()> {
    if let Some(storage) = value.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                return Err(kstest_invalid_argument(format!(
                    "kstest: integer {role} values must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn kstest_outputs_for_count(
    out_count: usize,
    eval: KstestEvaluation,
) -> BuiltinResult<Value> {
    match out_count {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(Value::output_list(vec![eval.h])),
        2 => Ok(Value::output_list(vec![eval.h, eval.p])),
        3 => Ok(Value::output_list(vec![eval.h, eval.p, eval.ksstat])),
        4 => Ok(Value::output_list(vec![
            eval.h,
            eval.p,
            eval.ksstat,
            eval.cv,
        ])),
        _ => Err(kstest_invalid_argument(
            "kstest: too many output arguments; maximum is 4",
        )),
    }
}

pub(super) async fn kstest_gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|cause| {
        let mut owner = kstest_invalid_argument(format!("kstest: {cause}"));
        owner.gpu_gather_retry = cause.gpu_gather_retry;
        owner.span = cause.span;
        owner.source = Some(Box::new(cause));
        owner
    })
}
pub(super) fn kstest_scalar_text<'a>(
    value: &'a Value,
    context: &str,
) -> BuiltinResult<Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(Cow::Borrowed(text.as_str())),
        Value::CharArray(array) if array.rows == 1 => Ok(Cow::Owned(
            crate::builtins::stats::decode_utf16_with_error(array.data(), || {
                kstest_invalid_argument(format!("kstest: {context} is not valid UTF-16 text"))
            })?,
        )),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(Cow::Borrowed).ok_or_else(|| {
                kstest_invalid_argument(format!("kstest: {context} must be nonmissing text"))
            })
        }
        Value::Cell(cell) if cell.data().len() == 1 => kstest_scalar_text(&cell.data()[0], context),
        other => Err(kstest_invalid_argument(format!(
            "kstest: {context} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn kstest_scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_values_f64(tensor).first().copied()
        }
        _ => None,
    }
}
