//! Moving Average Convergence/Divergence indicator.

use runmat_builtins::{
    BuiltinErrorDescriptor, MACD_ERROR_INTERNAL as ERROR_INTERNAL,
    MACD_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT, MACD_ERROR_OUTPUT_COUNT as ERROR_OUTPUT_COUNT,
    MACD_NONDOUBLE_MATRIX_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerInputAvailability, MACD_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_types::standard;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, selected_row_names, table_from_columns_like, table_height, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "macd";
const FAST_ALPHA: f64 = 0.15;
const SLOW_ALPHA: f64 = 0.075;
const SIGNAL_ALPHA: f64 = 0.20;

#[runtime_builtin(
    name = "macd",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::finance::macd"
)]
async fn macd_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    data: Value,
) -> BuiltinResult<Value> {
    let output_count = crate::output_count::current_output_count();
    match output_count {
        Some(0) => return Ok(Value::output_list(Vec::new())),
        Some(count) if count > 2 => {
            return Err(macd_error(
                ERROR_OUTPUT_COUNT,
                "macd: at most two outputs are supported",
            ));
        }
        _ => {}
    }

    let input = MacdInput::from_value(data).await?;
    let include_signal = output_count == Some(2);
    let eval = input.evaluate(runtime.as_ref(), include_signal)?;
    match output_count {
        Some(1) => Ok(Value::output_list(vec![eval.macd])),
        Some(2) => Ok(Value::output_list(vec![
            eval.macd,
            eval.signal.expect("signal requested"),
        ])),
        None => Ok(eval.macd),
        _ => unreachable!("output count is validated before evaluation"),
    }
}

enum MacdInput {
    Matrix {
        close: Tensor,
    },
    Tabular {
        source: runmat_value::ObjectInstance,
        close: Tensor,
    },
}

struct MacdEval {
    macd: Value,
    signal: Option<Value>,
}

impl MacdInput {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        // Class and shape admission precede the one provider gather. An integer
        // resident payload must be inspected after that gather, not downloaded
        // once for exactness and again for the recurrence.
        let raw_integer =
            crate::builtins::common::validation::value_has_native_integer_class(&value);
        if raw_integer {
            crate::compatibility::ensure_builtin_extension_enabled(
                &MACD_NONDOUBLE_MATRIX_EXTENSION,
                NAME,
            )?;
        }
        let mut resident_declared_double = false;
        if let Value::GpuTensor(handle) = &value {
            let shape = handle.shape();
            if shape.len() != 2 || shape[1] != 4 {
                return Err(macd_invalid(
                    "macd: matrix input must have exactly four columns [High Low Open Close]",
                ));
            }
            resident_declared_double = runmat_accelerate_api::handle_class_identity(handle)
                .is_some_and(|class| class.is(standard::DOUBLE));
            let nondouble_numeric = runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_class_identity(handle)
                    .is_some_and(|class| class.is(standard::SINGLE));
            if nondouble_numeric {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &MACD_NONDOUBLE_MATRIX_EXTENSION,
                    NAME,
                )?;
            }
        }
        if let Value::Tensor(tensor) = &value {
            let shape = tensor_shape_for(tensor);
            if shape.len() != 2 || shape[1] != 4 {
                return Err(macd_invalid(
                    "macd: matrix input must have exactly four columns [High Low Open Close]",
                ));
            }
            if tensor.numeric_dtype() != NumericDType::F64 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &MACD_NONDOUBLE_MATRIX_EXTENSION,
                    NAME,
                )?;
            }
            if raw_integer
                && !crate::builtins::common::validation::native_integer_value_is_exact_f64(&value)
            {
                return Err(macd_invalid(
                    "macd: integer matrix prices must be exactly representable as double",
                ));
            }
        }
        let value = gather_if_needed_async(&value).await.map_err(|mut err| {
            if err.context.builtin.is_none() {
                err.context = err.context.with_builtin(NAME);
            }
            err
        })?;
        match value {
            Value::Tensor(tensor) => {
                let shape = tensor_shape_for(&tensor);
                if shape.len() != 2 || shape[1] != 4 {
                    return Err(macd_invalid(
                        "macd: matrix input must have exactly four columns [High Low Open Close]",
                    ));
                }
                if tensor.numeric_dtype() != NumericDType::F64 && !resident_declared_double {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &MACD_NONDOUBLE_MATRIX_EXTENSION,
                        NAME,
                    )?;
                }
                if tensor.integer_storage().is_some_and(|storage| {
                    storage.exact_values().iter().any(|value| {
                        !crate::builtins::common::validation::integer_is_exact_f64(value)
                    })
                }) {
                    return Err(macd_invalid(
                        "macd: integer matrix prices must be exactly representable as double",
                    ));
                }
                Ok(MacdInput::Matrix {
                    close: close_column_from_matrix(&tensor, &shape)?,
                })
            }
            Value::Object(object) if is_tabular_object(&object) => {
                let close = validate_price_variables(&object)?;
                Ok(MacdInput::Tabular {
                    source: object,
                    close,
                })
            }
            other => Err(macd_invalid(format!(
                "macd: expected M-by-4 matrix, table, or timetable, got {other:?}"
            ))),
        }
    }

    fn evaluate(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
        include_signal: bool,
    ) -> BuiltinResult<MacdEval> {
        match self {
            MacdInput::Matrix { close } => {
                let (macd, signal) = compute_macd(&close, include_signal)?;
                Ok(MacdEval {
                    macd: Value::Tensor(macd),
                    signal: signal.map(Value::Tensor),
                })
            }
            MacdInput::Tabular { source, close } => {
                let (macd, signal) = compute_macd(&close, include_signal)?;
                let rows = (0..close.rows()).collect::<Vec<_>>();
                let row_names = selected_row_names(&source, &rows)?;
                let macd = table_from_columns_like(
                    runtime,
                    &source,
                    vec!["Close".to_string()],
                    vec![Value::Tensor(macd)],
                    row_names.clone(),
                    None,
                )?;
                let signal = signal
                    .map(|signal| {
                        table_from_columns_like(
                            runtime,
                            &source,
                            vec!["Close".to_string()],
                            vec![Value::Tensor(signal)],
                            row_names,
                            None,
                        )
                    })
                    .transpose()?;
                Ok(MacdEval { macd, signal })
            }
        }
    }
}

fn compute_macd(close: &Tensor, include_signal: bool) -> BuiltinResult<(Tensor, Option<Tensor>)> {
    let shape = tensor_shape_for(close);
    let rows = shape
        .first()
        .copied()
        .unwrap_or_else(|| tensor::tensor_element_len(close));
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };
    let close_values = tensor::tensor_values_f64_cow(close);
    let fast = exponential_moving_average(&close_values, rows, cols, FAST_ALPHA);
    let slow = exponential_moving_average(&close_values, rows, cols, SLOW_ALPHA);
    let macd = fast
        .iter()
        .zip(slow.iter())
        .map(|(fast, slow)| fast - slow)
        .collect::<Vec<_>>();
    let signal = if include_signal {
        Some(exponential_moving_average(&macd, rows, cols, SIGNAL_ALPHA))
    } else {
        None
    };
    let dtype = close.numeric_dtype();
    Ok((
        Tensor::new_with_dtype(macd, vec![rows, cols], dtype)
            .map_err(|err| macd_internal(format!("macd: {err}")))?,
        signal
            .map(|signal| {
                Tensor::new_with_dtype(signal, vec![rows, cols], dtype)
                    .map_err(|err| macd_internal(format!("macd: {err}")))
            })
            .transpose()?,
    ))
}

fn exponential_moving_average(data: &[f64], rows: usize, cols: usize, alpha: f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if rows == 0 {
        return out;
    }
    let retention = 1.0 - alpha;
    for col in 0..cols {
        let base = col * rows;
        let mut average = data[base];
        out[base] = average;
        for row in 1..rows {
            let idx = base + row;
            average = alpha * data[idx] + retention * average;
            out[idx] = average;
        }
    }
    out
}

fn close_column_from_matrix(tensor: &Tensor, shape: &[usize]) -> BuiltinResult<Tensor> {
    let rows = shape[0];
    let close = (0..rows)
        .map(|row| tensor::tensor_value_f64(tensor, row + 3 * rows))
        .collect::<Vec<_>>();
    Tensor::new(close, vec![rows, 1]).map_err(|err| macd_internal(format!("macd: {err}")))
}

fn validate_price_variables(object: &runmat_value::ObjectInstance) -> BuiltinResult<Tensor> {
    let variables = table_variables(object)?;
    let height = table_height(object)?;
    let mut close = None;
    for required in ["High", "Low", "Open", "Close"] {
        let Some((_, value)) = variables
            .fields
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(required))
        else {
            return Err(macd_invalid(format!(
                "macd: table input must contain variable '{required}'"
            )));
        };
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
            return Err(macd_invalid(format!(
                "macd: table variable '{required}' integer values must be exactly representable as double"
            )));
        }
        let tensor = tensor_from_numeric_value(value.clone())?;
        let shape = tensor_shape_for(&tensor);
        let rows = shape
            .first()
            .copied()
            .unwrap_or_else(|| tensor::tensor_element_len(&tensor));
        let cols = if shape.len() >= 2 { shape[1] } else { 1 };
        if rows != height || cols != 1 || shape.len() > 2 {
            return Err(macd_invalid(format!(
                "macd: table variable '{required}' must be an M-by-1 numeric vector"
            )));
        }
        if required == "Close" {
            close = Some(tensor);
        }
    }
    close.ok_or_else(|| macd_invalid("macd: table input must contain variable 'Close'"))
}

fn tensor_from_numeric_value(value: Value) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| macd_invalid(format!("macd: {err}")))?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| macd_invalid(format!("macd: {err}")))
}

fn tensor_shape_for(tensor: &Tensor) -> Vec<usize> {
    if tensor.shape().is_empty() {
        tensor::default_shape_for(tensor.shape(), tensor::tensor_element_len(tensor))
    } else {
        tensor.shape().to_vec()
    }
}

fn macd_invalid(message: impl Into<String>) -> RuntimeError {
    macd_error(ERROR_INVALID_INPUT, message)
}

fn macd_internal(message: impl Into<String>) -> RuntimeError {
    macd_error(ERROR_INTERNAL, message)
}

fn macd_error(descriptor: BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let builder = build_runtime_error(message).with_builtin(NAME);
    match descriptor.identifier {
        Some(identifier) => builder.with_identifier(identifier).build(),
        None => builder.build(),
    }
}

#[cfg(test)]
mod tests;
