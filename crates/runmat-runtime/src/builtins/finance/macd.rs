//! Moving Average Convergence/Divergence indicator.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, ResolveContext, Tensor, Type, Value,
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

const MACD_NONDOUBLE_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "macd-nondouble-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "raw non-double macd matrix input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MacdNondoubleMatrixExtension"),
};

pub const MACD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MACD_NONDOUBLE_MATRIX_EXTENSION];

const MACD_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Data matrix",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "R2026a documents raw matrix Data as double; RunMat mode admits native integer M-by-4 matrices only when every price is exactly representable at the binary64 EMA boundary.",
    }];

const MACD_INTEGER_TABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "High/Low/Open/Close table variables",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented table and timetable containers impose names and vector shapes without restricting the numeric storage class of their price variables.",
    }];

pub const MACD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[MACDLine, SignalLine] = macd(integer_Data_matrix)",
        inputs: &MACD_INTEGER_MATRIX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The raw-matrix extension is gated before provider access, reads authoritative integer storage, rejects lossy binary64 conversion, and computes host-double column outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[MACDLine, SignalLine] = macd(table_with_integer_prices)",
        inputs: &MACD_INTEGER_TABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer price variables remain exact until a checked binary64 EMA boundary; outputs preserve the table or timetable container and use double Close data.",
    },
];

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
    extensions(crate::builtins::finance::macd::MACD_EXTENSIONS),
    integer_capabilities(crate::builtins::finance::macd::MACD_INTEGER_CAPABILITIES),
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
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            &value,
            &MACD_NONDOUBLE_MATRIX_EXTENSION,
            NAME,
            "matrix price",
        )
        .await?;
        let mut resident_declared_double = false;
        if let Value::GpuTensor(handle) = &value {
            resident_declared_double = runmat_accelerate_api::handle_class_name(handle)
                .is_some_and(|class| class.eq_ignore_ascii_case("double"));
            let nondouble_numeric = runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_class_name(handle)
                    .is_some_and(|class| class.eq_ignore_ascii_case("single"));
            if nondouble_numeric {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &MACD_NONDOUBLE_MATRIX_EXTENSION,
                    NAME,
                )?;
            }
        }
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
                if tensor.numeric_dtype() != NumericDType::F64 && !resident_declared_double {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &MACD_NONDOUBLE_MATRIX_EXTENSION,
                        NAME,
                    )?;
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

    fn call_with_mode(value: Value, extensions_enabled: bool) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
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
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(data), vec![rows, 4]).expect("integer matrix");
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
        assert_close(out.materialize_f64()[0], 0.0);
        assert_close(out.materialize_f64()[1], 0.075);
        assert_close(out.materialize_f64()[2], 0.283125);
        assert_close(out.materialize_f64()[3], 0.743578125);
        assert_close(out.materialize_f64()[4], 1.697244140625);
    }

    #[test]
    fn matrix_input_reads_typed_integer_storage_exactly_as_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
        assert_close(out.materialize_f64()[0], 0.0);
        assert_close(out.materialize_f64()[1], 0.075);
        assert_close(out.materialize_f64()[2], 0.283125);
        assert_close(out.materialize_f64()[3], 0.743578125);
        assert_close(out.materialize_f64()[4], 1.697244140625);
    }

    #[test]
    fn raw_nondouble_matrix_follows_compatibility_mode() {
        let integer = || integer_matrix(1, [[2, 1, 1, 1], [0; 4], [0; 4], [0; 4], [0; 4]]);
        let error = call_with_mode(integer(), false)
            .expect_err("MATLAB mode rejects raw integer macd matrix");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
        );
        call_with_mode(integer(), true).expect("RunMat mode accepts raw integer macd matrix");

        let single = Tensor::from_f32(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("single matrix");
        let error = call_with_mode(Value::Tensor(single), false)
            .expect_err("MATLAB mode rejects raw single macd matrix");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
        );
    }

    #[test]
    fn integer_prices_reject_before_a_lossy_binary64_boundary() {
        let wide = (1_u64 << 53) + 1;
        let matrix = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![wide; 4]), vec![1, 4])
                .expect("wide matrix"),
        );
        let error = call_with_mode(matrix, true).expect_err("lossy raw price must reject");
        assert!(error.message().contains("exactly representable"));

        let table = table_from_columns(
            vec![
                "High".to_string(),
                "Low".to_string(),
                "Open".to_string(),
                "Close".to_string(),
            ],
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
                        .expect("wide close"),
                ),
            ],
        )
        .expect("table");
        let error = call(table).expect_err("lossy table price must reject");
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn macd_integer_capabilities_separate_matrix_extension_and_table_data() {
        assert_eq!(MACD_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            MACD_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(
            MACD_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
    }

    #[test]
    fn resident_nondouble_matrix_uses_the_same_compatibility_gate() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let double = Tensor::new(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("matrix");
            let double_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &double)
                    .expect("upload");
            runmat_accelerate_api::set_handle_precision(
                &double_handle,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            runmat_accelerate_api::set_handle_class_name(&double_handle, "double");
            call_with_mode(Value::GpuTensor(double_handle), false)
                .expect("provider precision does not change documented double class");

            let upload = || {
                let matrix =
                    Tensor::from_f32(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("matrix");
                let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &matrix)
                    .expect("upload");
                runmat_accelerate_api::set_handle_precision(
                    &handle,
                    runmat_accelerate_api::ProviderPrecision::F32,
                );
                runmat_accelerate_api::set_handle_class_name(&handle, "single");
                handle
            };
            let error = call_with_mode(Value::GpuTensor(upload()), false)
                .expect_err("MATLAB mode rejects resident single matrix");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
            );
            call_with_mode(Value::GpuTensor(upload()), true)
                .expect("RunMat mode accepts resident single matrix");
        });
    }

    #[test]
    fn tensor_shape_for_reads_scalar_typed_integer_storage_without_mirror() {
        let input = Tensor::new_integer(IntegerStorage::U8(vec![7]), Vec::new()).unwrap();

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
        assert_close(signal.materialize_f64()[0], 0.0);
        assert_close(signal.materialize_f64()[1], 0.015);
        assert_close(signal.materialize_f64()[2], 0.068625);
        assert_close(signal.materialize_f64()[3], 0.203615625);
        assert_close(signal.materialize_f64()[4], 0.502341328125);
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
        assert!(macd.materialize_f64().is_empty());
        assert!(signal.materialize_f64().is_empty());
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
        assert_eq!(out.materialize_f64()[0], 0.0);
        assert!(out.materialize_f64()[1].is_nan());
        assert!(out.materialize_f64()[2].is_nan());
        assert!(out.materialize_f64()[3].is_nan());
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
        assert_close(close.materialize_f64()[1], 0.075);
        assert_close(close.materialize_f64()[2], 0.283125);
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
