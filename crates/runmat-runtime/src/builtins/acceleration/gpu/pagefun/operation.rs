use super::*;
#[derive(Clone, Copy)]
pub(super) enum PageOperation {
    Mtimes,
}

impl PageOperation {
    pub(super) fn from_callable(value: Value) -> BuiltinResult<Self> {
        use std::borrow::Cow;
        let raw = match &value {
            Value::FunctionHandle(func) => Cow::Borrowed(func.name()),
            Value::ExternalFunctionHandle(func) => Cow::Borrowed(func.name()),
            Value::BoundFunctionHandle(handle) => Cow::Borrowed(handle.name.as_str()),
            Value::String(runmat_value::StringElement::Text(s)) => Cow::Borrowed(s.as_str()),
            Value::StringArray(sa) => {
                if sa.data().len() != 1 {
                    return Err(pagefun_error_with_detail(
                        &PAGEFUN_ERROR_INVALID_CALLABLE,
                        "function string array must contain exactly one element",
                    ));
                }
                Cow::Borrowed(sa.data()[0].as_str().ok_or_else(|| {
                    pagefun_error_with_detail(
                        &PAGEFUN_ERROR_INVALID_CALLABLE,
                        "function string array must contain exactly one nonmissing element",
                    )
                })?)
            }
            Value::CharArray(chars) => {
                if chars.rows != 1 {
                    return Err(pagefun_error_with_detail(
                        &PAGEFUN_ERROR_INVALID_CALLABLE,
                        "function char array must be a single row character vector",
                    ));
                }
                Cow::Owned(runmat_value::decode_utf16_code_units(chars.data()).map_err(
                    |source| {
                        let mut error = pagefun_error_with_detail(
                            &PAGEFUN_ERROR_INVALID_CALLABLE,
                            source.to_string(),
                        );
                        error.source = Some(Box::new(source));
                        error
                    },
                )?)
            }
            other => {
                return Err(pagefun_error_with_detail(
                    &PAGEFUN_ERROR_INVALID_CALLABLE,
                    format!("unsupported function handle type {}", other.type_name()),
                ))
            }
        };
        let trimmed = raw.trim();
        let lowered = trimmed.trim_start_matches('@').to_ascii_lowercase();
        match lowered.as_str() {
            "mtimes" => Ok(Self::Mtimes),
            _ => Err(pagefun_error_with_detail(
                &PAGEFUN_ERROR_UNSUPPORTED_OPERATION,
                format!("unsupported function '{trimmed}'; currently only @mtimes is implemented"),
            )),
        }
    }

    pub(super) fn validate_arity(&self, arg_count: usize) -> BuiltinResult<()> {
        match self {
            Self::Mtimes => {
                if arg_count != 2 {
                    return Err(pagefun_error(&PAGEFUN_ERROR_INVALID_ARITY));
                }
                Ok(())
            }
        }
    }

    pub(super) fn validate_shapes(&self, inputs: &[PreparedInput]) -> BuiltinResult<()> {
        match self {
            Self::Mtimes => {
                let lhs = &inputs[0];
                let rhs = &inputs[1];
                if lhs.rows() * lhs.cols() != 1
                    && rhs.rows() * rhs.cols() != 1
                    && lhs.cols() != rhs.rows()
                {
                    return Err(pagefun_error_with_detail(
                        &PAGEFUN_ERROR_MATRIX_DIM_MISMATCH,
                        format!(
                            "inner matrix dimensions must agree ({}x{} * {}x{})",
                            lhs.rows(),
                            lhs.cols(),
                            rhs.rows(),
                            rhs.cols()
                        ),
                    ));
                }
                Ok(())
            }
        }
    }

    pub(super) async fn evaluate(&self, args: &[Value]) -> crate::BuiltinResult<Value> {
        match self {
            Self::Mtimes => crate::builtins::math::linalg::ops::matrix_arithmetic::product::evaluate(&args[0], &args[1]).await.map_err(pagefun_operational_error)?.map_err(|error| {
                use crate::builtins::math::linalg::ops::matrix_arithmetic::product::ProductError;
                match error {
                    ProductError::InvalidInput(message) => pagefun_error_with_detail(&PAGEFUN_ERROR_INVALID_INPUT, message),
                    ProductError::Internal(message) => pagefun_internal_error(message),
                }
            }),
        }
    }

    pub(super) fn output_kind(&self, inputs: &[PreparedInput]) -> OutputKind {
        match self {
            Self::Mtimes => {
                if inputs.iter().any(|input| input.is_complex()) {
                    OutputKind::Complex
                } else {
                    OutputKind::Real
                }
            }
        }
    }

    pub(super) fn output_matrix_shape(
        &self,
        inputs: &[PreparedInput],
        kind: OutputKind,
    ) -> BuiltinResult<(usize, usize)> {
        match self {
            Self::Mtimes => {
                let lhs = &inputs[0];
                let rhs = &inputs[1];
                let (rows, cols) = if lhs.rows() * lhs.cols() == 1 {
                    (rhs.rows(), rhs.cols())
                } else if rhs.rows() * rhs.cols() == 1 {
                    (lhs.rows(), lhs.cols())
                } else {
                    (lhs.rows(), rhs.cols())
                };
                match kind {
                    OutputKind::Real | OutputKind::Complex => Ok((rows, cols)),
                }
            }
        }
    }

    pub(super) fn output_numeric_dtype(&self, inputs: &[PreparedInput]) -> NumericDType {
        match self {
            Self::Mtimes => inputs
                .iter()
                .filter_map(|input| input.data.numeric_dtype())
                .find(|dtype| {
                    matches!(
                        dtype,
                        NumericDType::I8
                            | NumericDType::I16
                            | NumericDType::I32
                            | NumericDType::I64
                            | NumericDType::U8
                            | NumericDType::U16
                            | NumericDType::U32
                            | NumericDType::U64
                    )
                })
                .or_else(|| {
                    inputs
                        .iter()
                        .filter_map(|input| input.data.numeric_dtype())
                        .find(|dtype| *dtype == NumericDType::F32)
                })
                .unwrap_or(NumericDType::F64),
        }
    }
}
