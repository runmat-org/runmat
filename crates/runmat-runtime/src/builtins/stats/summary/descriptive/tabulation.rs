//! Frequency tabulation with exact integer labels and typed cell output.
use super::{descriptive_error, gathered, is_real_typed_integer_value};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, IntValue, StringArray, Tensor, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input data array.",
};

const OUTPUT_TABLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tbl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Frequency table.",
}];

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];

const TABULATE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tabulate-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "tabulate with typed-integer input data and exact cell output is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TabulateIntegerDataExtension"),
};

const TABULATE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tabulate-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tabulate with a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TabulateGpuInputExtension"),
};

const TABULATE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TABULATE_INTEGER_DATA_EXTENSION,
    TABULATE_GPU_INPUT_EXTENSION,
];

const TABULATE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input domain is single or double; RunMat mode additionally accepts all eight integer classes and returns a heterogeneous cell table so exact wide values are not coerced through binary64.",
    }];

const TABULATE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tbl = tabulate(integer_X)",
        inputs: &TABULATE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer observations group and sort in exact same-class storage; the cell table preserves integer values in column one and uses double counts and percentages, while positive-gap expansion is bounded to avoid unrepresentable allocation.",
    }];

const TABULATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tbl = tabulate(X)",
    inputs: &INPUTS_X,
    outputs: &OUTPUT_TABLE,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DESCRIPTIVE.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, flags, dimensions, or options are malformed.",
    message: "descriptive statistics: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DESCRIPTIVE.INTERNAL",
    identifier: None,
    when: "Internal tensor conversion or allocation fails.",
    message: "descriptive statistics: internal error",
};

macro_rules! descriptive_descriptor {
    ($name:literal, $signatures:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn descriptive_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

const MAX_EXACT_TABULATE_ROWS: u128 = 1_000_000;

fn compare_same_class_integers(left: &IntValue, right: &IntValue) -> Ordering {
    match (left, right) {
        (IntValue::I8(left), IntValue::I8(right)) => left.cmp(right),
        (IntValue::I16(left), IntValue::I16(right)) => left.cmp(right),
        (IntValue::I32(left), IntValue::I32(right)) => left.cmp(right),
        (IntValue::I64(left), IntValue::I64(right)) => left.cmp(right),
        (IntValue::U8(left), IntValue::U8(right)) => left.cmp(right),
        (IntValue::U16(left), IntValue::U16(right)) => left.cmp(right),
        (IntValue::U32(left), IntValue::U32(right)) => left.cmp(right),
        (IntValue::U64(left), IntValue::U64(right)) => left.cmp(right),
        _ => unreachable!("integer tensor storage is homogeneous"),
    }
}

fn positive_integer_magnitude(value: &IntValue) -> Option<u128> {
    match value {
        IntValue::I8(value) if *value >= 1 => Some(*value as u128),
        IntValue::I16(value) if *value >= 1 => Some(*value as u128),
        IntValue::I32(value) if *value >= 1 => Some(*value as u128),
        IntValue::I64(value) if *value >= 1 => Some(*value as u128),
        IntValue::U8(value) if *value >= 1 => Some(*value as u128),
        IntValue::U16(value) if *value >= 1 => Some(*value as u128),
        IntValue::U32(value) if *value >= 1 => Some(*value as u128),
        IntValue::U64(value) if *value >= 1 => Some(*value as u128),
        _ => None,
    }
}

fn positive_integer_like(prototype: &IntValue, value: u128) -> IntValue {
    match prototype {
        IntValue::I8(_) => IntValue::I8(value as i8),
        IntValue::I16(_) => IntValue::I16(value as i16),
        IntValue::I32(_) => IntValue::I32(value as i32),
        IntValue::I64(_) => IntValue::I64(value as i64),
        IntValue::U8(_) => IntValue::U8(value as u8),
        IntValue::U16(_) => IntValue::U16(value as u16),
        IntValue::U32(_) => IntValue::U32(value as u32),
        IntValue::U64(_) => IntValue::U64(value as u64),
    }
}

fn exact_integer_tabulate(name: &str, mut values: Vec<IntValue>) -> BuiltinResult<Value> {
    let total = values.len();
    if values.is_empty() {
        return CellArray::new(Vec::new(), 0, 3)
            .map(Value::Cell)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}")));
    }
    let positive_max = values
        .iter()
        .map(positive_integer_magnitude)
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.into_iter().max());
    let rows = if let Some(max) = positive_max.filter(|max| *max <= MAX_EXACT_TABULATE_ROWS) {
        let prototype = values[0].clone();
        let counts =
            values
                .into_iter()
                .fold(HashMap::<IntValue, usize>::new(), |mut counts, value| {
                    *counts.entry(value).or_insert(0) += 1;
                    counts
                });
        (1..=max)
            .map(|value| {
                let value = positive_integer_like(&prototype, value);
                let count = counts.get(&value).copied().unwrap_or(0);
                (value, count)
            })
            .collect::<Vec<_>>()
    } else {
        values.sort_by(compare_same_class_integers);
        let mut rows = Vec::new();
        let mut idx = 0usize;
        while idx < values.len() {
            let value = values[idx].clone();
            let mut end = idx + 1;
            while end < values.len() && compare_same_class_integers(&values[end], &value).is_eq() {
                end += 1;
            }
            rows.push((value, end - idx));
            idx = end;
        }
        rows
    };
    let row_count = rows.len();
    let mut data = Vec::with_capacity(row_count * 3);
    for (value, count) in rows {
        data.push(Value::Int(value));
        data.push(Value::Num(count as f64));
        data.push(Value::Num(count as f64 * 100.0 / total as f64));
    }
    CellArray::new(data, row_count, 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn numeric_tabulate(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        return exact_integer_tabulate(name, storage.exact_values());
    }
    let mut values = tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    let n = values.len() as f64;
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
    });
    let positive_integers = !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && *value >= 1.0 && value.fract().abs() <= 1e-12);
    let mut rows = Vec::<(f64, f64)>::new();
    if positive_integers {
        let max_value = values.last().copied().unwrap_or(0.0).round().max(0.0) as usize;
        let mut counts = vec![0usize; max_value + 1];
        for value in values {
            counts[value.round() as usize] += 1;
        }
        rows.extend((1..=max_value).map(|idx| (idx as f64, counts[idx] as f64)));
    } else {
        let mut idx = 0usize;
        while idx < values.len() {
            let value = values[idx];
            let mut end = idx + 1;
            while end < values.len()
                && ((value.is_nan() && values[end].is_nan()) || values[end] == value)
            {
                end += 1;
            }
            rows.push((value, (end - idx) as f64));
            idx = end;
        }
    }
    let rows_len = rows.len();
    let mut data = Vec::with_capacity(rows_len * 3);
    data.extend(rows.iter().map(|(value, _)| *value));
    data.extend(rows.iter().map(|(_, count)| *count));
    data.extend(rows.iter().map(|(_, count)| {
        if n == 0.0 {
            f64::NAN
        } else {
            *count * 100.0 / n
        }
    }));
    Tensor::new(data, vec![rows_len, 3])
        .map(tensor::tensor_into_value)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn logical_tabulate(name: &str, data: Vec<u8>) -> BuiltinResult<Value> {
    let false_count = data.iter().filter(|value| **value == 0).count();
    let true_count = data.len() - false_count;
    let total = data.len() as f64;
    let mut rows = Vec::new();
    if false_count > 0 {
        rows.push((false, false_count));
    }
    if true_count > 0 {
        rows.push((true, true_count));
    }
    let mut cell_data = Vec::with_capacity(rows.len() * 3);
    for (value, count) in &rows {
        cell_data.push(Value::Bool(*value));
        cell_data.push(Value::Num(*count as f64));
        cell_data.push(Value::Num(if total == 0.0 {
            f64::NAN
        } else {
            *count as f64 * 100.0 / total
        }));
    }
    CellArray::new(cell_data, rows.len(), 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

pub(super) fn label_for_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::Num(num) => Some(num.to_string()),
        Value::Int(int) => Some(int.decimal_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn string_tabulate(name: &str, labels: impl IntoIterator<Item = String>) -> BuiltinResult<Value> {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut total = 0usize;
    for label in labels {
        *counts.entry(label).or_insert(0) += 1;
        total += 1;
    }
    let rows = counts.len();
    let mut data = Vec::with_capacity(rows * 3);
    for (label, count) in counts {
        data.push(Value::StringArray(
            StringArray::new(vec![label], vec![1, 1])
                .map_err(|err| descriptive_error(name, format!("{name}: {err}")))?,
        ));
        data.push(Value::Num(count as f64));
        data.push(Value::Num(if total == 0 {
            f64::NAN
        } else {
            count as f64 * 100.0 / total as f64
        }));
    }
    CellArray::new(data, rows, 3)
        .map(Value::Cell)
        .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
}

fn tabulate_value(name: &str, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => numeric_tabulate(name, tensor),
        Value::LogicalArray(logical) => logical_tabulate(name, logical.data.into_vec()),
        Value::Int(value) => exact_integer_tabulate(name, vec![value]),
        Value::Num(_) | Value::Bool(_) => tensor::value_into_tensor_for(name, value)
            .map_err(|err| descriptive_error(name, format!("{name}: {err}")))
            .and_then(|tensor| numeric_tabulate(name, tensor)),
        Value::String(text) => string_tabulate(name, [text]),
        Value::StringArray(array) => string_tabulate(name, array.data),
        Value::CharArray(chars) if chars.rows == 1 => {
            string_tabulate(name, chars.data.into_iter().map(|ch| ch.to_string()))
        }
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for entry in &cell.data {
                let Some(label) = label_for_value(entry) else {
                    return Err(descriptive_error(
                        name,
                        format!("{name}: cell entries must be scalar strings or scalars"),
                    ));
                };
                labels.push(label);
            }
            string_tabulate(name, labels)
        }
        other => Err(descriptive_error(
            name,
            format!("{name}: unsupported input type {other:?}"),
        )),
    }
}

pub mod tabulate {
    use super::*;
    descriptive_descriptor!("tabulate", TABULATE_SIGNATURES);

    #[runtime_builtin(
        name = "tabulate",
        category = "stats/summary",
        summary = "Create a frequency table.",
        keywords = "tabulate,frequency,count,percent,statistics",
        type_resolver(super::descriptive_type),
        descriptor(self::DESCRIPTOR),
        extensions(super::TABULATE_EXTENSIONS),
        integer_capabilities(super::TABULATE_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::descriptive::tabulation::tabulate"
    )]
    pub(crate) async fn tabulate_builtin(value: Value) -> BuiltinResult<Value> {
        if super::is_real_typed_integer_value(&value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABULATE_INTEGER_DATA_EXTENSION,
                "tabulate",
            )?;
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &TABULATE_GPU_INPUT_EXTENSION,
                "tabulate",
            )?;
        }
        let value = super::gathered(value, "tabulate").await?;
        super::tabulate_value("tabulate", value)
    }
}
