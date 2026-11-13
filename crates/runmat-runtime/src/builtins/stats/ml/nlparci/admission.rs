use super::*;
use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
use runmat_value::NumericDType;

pub(super) use runmat_builtins::NlparciSource as Method;

pub(super) struct Admitted {
    pub(super) method: Method,
    pub(super) source: Value,
    pub(super) alpha: Option<Value>,
}

pub(super) fn dimensions(value: &Value) -> BuiltinResult<(usize, usize, NumericDType)> {
    let (shape, dtype): (&[usize], NumericDType) = match value {
        Value::Num(_) => (&[1, 1], NumericDType::F64),
        Value::Tensor(tensor) => (tensor.shape(), tensor.numeric_dtype()),
        Value::GpuTensor(handle)
            if handle.physical().storage() == GpuTensorStorage::Real
                && !handle.view().is_logical() =>
        {
            let dtype = match runmat_accelerate_api::handle_precision(handle) {
                Some(ProviderPrecision::F32) => NumericDType::F32,
                Some(ProviderPrecision::F64) => NumericDType::F64,
                None => return Err(invalid("resident input has no real floating precision")),
            };
            (handle.shape(), dtype)
        }
        _ => return Err(invalid("inputs must be real single or double arrays")),
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(invalid("inputs must be real single or double arrays"));
    }
    let (rows, cols) = match shape {
        [rows] => (*rows, 1),
        [rows, cols] => (*rows, *cols),
        _ => {
            return Err(invalid(
                "inputs must be scalar, vector, or two-dimensional matrix",
            ))
        }
    };
    rows.checked_mul(cols)
        .ok_or_else(|| invalid("input dimensions overflow"))?;
    Ok((rows, cols, dtype))
}

pub(super) fn admit(
    beta: &Value,
    residual: &Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Admitted> {
    if !matches!(rest.len(), 2 | 4) {
        return Err(invalid("expected beta, residuals, and a Jacobian or Covar name-value pair, optionally followed by Alpha"));
    }
    let method = crate::builtins::common::validation::value_text_scalar_string(&rest[0])
        .as_deref()
        .and_then(Method::parse)
        .ok_or_else(|| invalid("the first option must be Jacobian or Covar"))?;
    if rest.len() == 4
        && !crate::builtins::common::validation::value_text_scalar_string(&rest[2])
            .is_some_and(|name| name.eq_ignore_ascii_case("Alpha"))
    {
        return Err(invalid("the final option must be Alpha"));
    }
    let (beta_rows, beta_cols, _) = dimensions(beta)?;
    let coefficients = beta_rows
        .checked_mul(beta_cols)
        .ok_or_else(|| invalid("coefficient dimensions overflow"))?;
    if coefficients == 0 || beta_rows != 1 && beta_cols != 1 {
        return Err(invalid("beta must be a nonempty real floating vector"));
    }
    coefficients
        .checked_mul(2)
        .ok_or_else(|| invalid("confidence interval dimensions overflow"))?;
    let (r_rows, r_cols, _) = dimensions(residual)?;
    let observations = r_rows
        .checked_mul(r_cols)
        .ok_or_else(|| invalid("residual dimensions overflow"))?;
    if observations == 0 || r_rows != 1 && r_cols != 1 {
        return Err(invalid("residuals must be a nonempty real floating vector"));
    }
    let (source_rows, source_cols, _) = dimensions(&rest[1])?;
    match method {
        Method::Jacobian if source_rows != observations || source_cols != coefficients => {
            return Err(invalid(
                "Jacobian must have one row per residual and one column per coefficient",
            ));
        }
        Method::Covariance if source_rows != coefficients || source_cols != coefficients => {
            return Err(invalid(
                "Covar must be a square matrix with one row per coefficient",
            ));
        }
        _ => {}
    }
    if rest.len() == 4 {
        let (rows, cols, _) = dimensions(&rest[3])?;
        if rows != 1 || cols != 1 {
            return Err(invalid("Alpha must be a real floating scalar"));
        }
    }
    let mut values = rest.into_iter();
    let _method_name = values.next();
    let source = values.next().expect("admitted source pair");
    let _alpha_name = values.next();
    let alpha = values.next();
    Ok(Admitted {
        method,
        source,
        alpha,
    })
}
