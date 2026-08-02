//! Moving Average Convergence/Divergence indicator.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, selected_row_names, table_from_columns_like, table_height, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "macd";
const FAST_ALPHA: f64 = 0.15;
const SLOW_ALPHA: f64 = 0.075;
const SIGNAL_ALPHA: f64 = 0.20;

const PARAM_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Price data as an M-by-4 matrix, table, or timetable.",
};

const OUTPUT_MACD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "MACDLine",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Moving Average Convergence/Divergence series.",
};

const OUTPUT_SIGNAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SignalLine",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Nine-period exponential moving average of the MACD line.",
};

const INPUTS: [BuiltinParamDescriptor; 1] = [PARAM_DATA];
const OUTPUTS_MACD: [BuiltinParamDescriptor; 1] = [OUTPUT_MACD];
const OUTPUTS_BOTH: [BuiltinParamDescriptor; 2] = [OUTPUT_MACD, OUTPUT_SIGNAL];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "MACDLine = macd(Data)",
        inputs: &INPUTS,
        outputs: &OUTPUTS_MACD,
    },
    BuiltinSignatureDescriptor {
        label: "[MACDLine, SignalLine] = macd(Data)",
        inputs: &INPUTS,
        outputs: &OUTPUTS_BOTH,
    },
];

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.macd.INVALID_INPUT",
        identifier: Some("RunMat:macd:InvalidInput"),
        when: "Input is not a numeric M-by-4 matrix or table/timetable with High, Low, Open, and Close variables.",
        message: "macd: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.macd.OUTPUT_COUNT",
        identifier: Some("RunMat:macd:OutputCount"),
        when: "More than two output arguments are requested.",
        message: "macd: too many output arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.macd.INTERNAL",
        identifier: Some("RunMat:macd:Internal"),
        when: "Output construction fails.",
        message: "macd: internal error",
    },
];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn macd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) => Type::Tensor {
            shape: shape.as_ref().map(|dims| {
                if dims.len() >= 2 {
                    vec![dims[0], Some(1)]
                } else {
                    dims.clone()
                }
            }),
        },
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "macd",
    category = "finance",
    summary = "Compute the Moving Average Convergence/Divergence indicator.",
    keywords = "macd,finance,technical indicator,ema,moving average",
    type_resolver(macd_type),
    descriptor(crate::builtins::finance::macd::DESCRIPTOR),
    builtin_path = "crate::builtins::finance::macd"
)]
async fn macd_builtin(data: Value) -> BuiltinResult<Value> {
    let output_count = crate::output_count::current_output_count();
    match output_count {
        Some(0) => return Ok(Value::OutputList(Vec::new())),
        Some(count) if count > 2 => {
            return Err(macd_error(
                "RunMat:macd:OutputCount",
                "macd: at most two outputs are supported",
            ));
        }
        _ => {}
    }

    let input = MacdInput::from_value(data).await?;
    let include_signal = output_count == Some(2);
    let eval = input.evaluate(include_signal)?;
    match output_count {
        Some(1) => Ok(Value::OutputList(vec![eval.macd])),
        Some(2) => Ok(Value::OutputList(vec![
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
        source: runmat_builtins::ObjectInstance,
        close: Tensor,
    },
}

struct MacdEval {
    macd: Value,
    signal: Option<Value>,
}

impl MacdInput {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        let value = gather_if_needed_async(&value)
            .await
            .map_err(|err| macd_internal(format!("macd: {err}")))?;
        match value {
            Value::Tensor(tensor) => {
                let shape = tensor_shape_for(&tensor);
                if shape.len() != 2 || shape[1] != 4 {
                    return Err(macd_invalid(
                        "macd: matrix input must have exactly four columns [High Low Open Close]",
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

    fn evaluate(self, include_signal: bool) -> BuiltinResult<MacdEval> {
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
                    &source,
                    vec!["Close".to_string()],
                    vec![Value::Tensor(macd)],
                    row_names.clone(),
                    None,
                )?;
                let signal = signal
                    .map(|signal| {
                        table_from_columns_like(
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

fn validate_price_variables(object: &runmat_builtins::ObjectInstance) -> BuiltinResult<Tensor> {
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
    if tensor.shape.is_empty() {
        tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(tensor))
    } else {
        tensor.shape.clone()
    }
}

fn macd_invalid(message: impl Into<String>) -> RuntimeError {
    macd_error("RunMat:macd:InvalidInput", message)
}

fn macd_internal(message: impl Into<String>) -> RuntimeError {
    macd_error("RunMat:macd:Internal", message)
}

fn macd_error(identifier: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier(identifier)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, NumericDType, NumericScalar, Value};

    use crate::builtins::table::{table_from_columns, table_variables};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(macd_builtin(value))
    }

    fn matrix(rows: usize, columns: [[f64; 4]; 5]) -> Value {
        let mut data = Vec::with_capacity(rows * 4);
        for col in 0..4 {
            for row_values in columns.iter().take(rows) {
                data.push(row_values[col]);
            }
        }
        Value::Tensor(Tensor::new(data, vec![rows, 4]).unwrap())
    }

    fn integer_matrix(rows: usize, columns: [[i16; 4]; 5]) -> Value {
        let mut data = Vec::with_capacity(rows * 4);
        for col in 0..4 {
            for row_values in columns.iter().take(rows) {
                data.push(row_values[col]);
            }
        }
        let mut tensor =
            Tensor::new_integer(IntegerStorage::I16(data), vec![rows, 4]).expect("integer matrix");
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn matrix_input_returns_macd_line_by_default() {
        let out = expect_tensor(
            call(matrix(
                5,
                [
                    [2.0, 0.5, 1.0, 1.0],
                    [3.0, 1.5, 2.0, 2.0],
                    [4.0, 2.5, 3.0, 4.0],
                    [5.0, 3.5, 4.0, 8.0],
                    [6.0, 4.5, 5.0, 16.0],
                ],
            ))
            .unwrap(),
        );
        assert_eq!(out.shape, vec![5, 1]);
        assert_close(out.data[0], 0.0);
        assert_close(out.data[1], 0.075);
        assert_close(out.data[2], 0.283125);
        assert_close(out.data[3], 0.743578125);
        assert_close(out.data[4], 1.697244140625);
    }

    #[test]
    fn matrix_input_reads_typed_integer_storage_exactly_as_double() {
        let out = expect_tensor(
            call(integer_matrix(
                5,
                [
                    [2, 1, 1, 1],
                    [3, 2, 2, 2],
                    [4, 3, 3, 4],
                    [5, 4, 4, 8],
                    [6, 5, 5, 16],
                ],
            ))
            .unwrap(),
        );
        assert_eq!(out.shape, vec![5, 1]);
        assert!(out.integer_storage().is_none());
        assert_close(out.data[0], 0.0);
        assert_close(out.data[1], 0.075);
        assert_close(out.data[2], 0.283125);
        assert_close(out.data[3], 0.743578125);
        assert_close(out.data[4], 1.697244140625);
    }

    #[test]
    fn tensor_shape_for_reads_scalar_typed_integer_storage_without_mirror() {
        let mut input = Tensor::new_integer(IntegerStorage::U8(vec![7]), Vec::new()).unwrap();
        input.data.clear();

        assert_eq!(tensor_shape_for(&input), vec![1, 1]);
    }

    #[test]
    fn two_outputs_return_macd_and_signal() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = call(matrix(
            5,
            [
                [2.0, 0.5, 1.0, 1.0],
                [3.0, 1.5, 2.0, 2.0],
                [4.0, 2.5, 3.0, 4.0],
                [5.0, 3.5, 4.0, 8.0],
                [6.0, 4.5, 5.0, 16.0],
            ],
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected outputs");
        };
        assert_eq!(values.len(), 2);
        let macd = expect_tensor(values[0].clone());
        let signal = expect_tensor(values[1].clone());
        assert_eq!(macd.shape, vec![5, 1]);
        assert_eq!(signal.shape, vec![5, 1]);
        assert_close(signal.data[0], 0.0);
        assert_close(signal.data[1], 0.015);
        assert_close(signal.data[2], 0.068625);
        assert_close(signal.data[3], 0.203615625);
        assert_close(signal.data[4], 0.502341328125);
    }

    #[test]
    fn zero_outputs_return_empty_without_validating_input() {
        let _guard = crate::output_count::push_output_count(Some(0));
        let result = call(Value::String("not data".to_string())).unwrap();
        assert_eq!(result, Value::OutputList(Vec::new()));
    }

    #[test]
    fn rejects_more_than_two_outputs_before_evaluation() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = call(matrix(
            1,
            [[2.0, 0.5, 1.0, 1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:OutputCount"));
    }

    #[test]
    fn empty_matrix_returns_empty_series() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = call(Value::Tensor(Tensor::new(vec![], vec![0, 4]).unwrap())).unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected outputs");
        };
        let macd = expect_tensor(values[0].clone());
        let signal = expect_tensor(values[1].clone());
        assert_eq!(macd.shape, vec![0, 1]);
        assert_eq!(signal.shape, vec![0, 1]);
        assert!(macd.data.is_empty());
        assert!(signal.data.is_empty());
    }

    #[test]
    fn nan_prices_propagate_through_ema_recurrence() {
        let out = expect_tensor(
            call(matrix(
                4,
                [
                    [2.0, 0.5, 1.0, 1.0],
                    [3.0, 1.5, 2.0, f64::NAN],
                    [4.0, 2.5, 3.0, 4.0],
                    [5.0, 3.5, 4.0, 8.0],
                    [0.0; 4],
                ],
            ))
            .unwrap(),
        );
        assert_eq!(out.shape, vec![4, 1]);
        assert_eq!(out.data[0], 0.0);
        assert!(out.data[1].is_nan());
        assert!(out.data[2].is_nan());
        assert!(out.data[3].is_nan());
    }

    #[test]
    fn table_input_is_case_insensitive_and_returns_table_with_close() {
        let table = table_from_columns(
            vec![
                "high".to_string(),
                "LOW".to_string(),
                "Open".to_string(),
                "close".to_string(),
            ],
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap()),
            ],
        )
        .unwrap();
        let out = call(table).unwrap();
        let Value::Object(object) = out else {
            panic!("expected table output");
        };
        assert_eq!(object.class_name, "table");
        let variables = table_variables(&object).unwrap();
        assert_eq!(variables.fields.len(), 1);
        let close = expect_tensor(variables.fields.get("Close").cloned().unwrap());
        assert_eq!(close.shape, vec![3, 1]);
        assert_close(close.data[1], 0.075);
        assert_close(close.data[2], 0.283125);
    }

    #[test]
    fn table_input_preserves_native_single_close_class() {
        let single_column = |values: Vec<f64>| {
            Value::Tensor(
                Tensor::new_with_dtype(values, vec![3, 1], NumericDType::F32)
                    .expect("single table variable"),
            )
        };
        let table = table_from_columns(
            vec![
                "High".to_string(),
                "Low".to_string(),
                "Open".to_string(),
                "Close".to_string(),
            ],
            vec![
                single_column(vec![2.0, 3.0, 4.0]),
                single_column(vec![0.5, 1.5, 2.5]),
                single_column(vec![1.0, 2.0, 3.0]),
                single_column(vec![1.0, 2.0, 4.0]),
            ],
        )
        .expect("single table");

        let Value::Object(object) = call(table).expect("macd") else {
            panic!("expected table output");
        };
        let variables = table_variables(&object).expect("output variables");
        let close = expect_tensor(variables.fields.get("Close").cloned().expect("Close"));
        assert_eq!(close.numeric_dtype(), NumericDType::F32);
        for (index, expected) in [0.0_f32, 0.075, 0.283125].into_iter().enumerate() {
            assert_eq!(
                close.numeric_value_at(index),
                Some(NumericScalar::F32(expected))
            );
        }
    }

    #[test]
    fn rejects_matrix_without_four_columns() {
        let err = call(Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
    }

    #[test]
    fn rejects_higher_rank_matrix_input() {
        let err = call(Value::Tensor(
            Tensor::new((1..=16).map(|value| value as f64).collect(), vec![2, 4, 2]).unwrap(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
    }

    #[test]
    fn rejects_table_missing_required_price_variable() {
        let table = table_from_columns(
            vec!["High".to_string(), "Low".to_string(), "Open".to_string()],
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            ],
        )
        .unwrap();
        let err = call(table).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
    }

    #[test]
    fn rejects_nonnumeric_price_variables() {
        let table = table_from_columns(
            vec![
                "High".to_string(),
                "Low".to_string(),
                "Open".to_string(),
                "Close".to_string(),
            ],
            vec![
                Value::String("not numeric".to_string()),
                Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
            ],
        )
        .unwrap();
        let err = call(table).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
    }

    #[test]
    fn rejects_multicolumn_price_variables() {
        let table = table_from_columns(
            vec![
                "High".to_string(),
                "Low".to_string(),
                "Open".to_string(),
                "Close".to_string(),
            ],
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
            ],
        )
        .unwrap();
        let err = call(table).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
    }
}
