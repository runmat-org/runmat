//! Shared MATLAB argument-validation helpers and callable `mustBe*` builtins.

use std::cmp::Ordering;
use std::path::Path;

use runmat_accelerate_api::{handle_integer_type, handle_is_logical};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, SparseTensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::identifiers::is_valid_varname;
use crate::builtins::common::tensor;
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::underlying_type::underlying_type_matches;
use crate::builtins::logical::rel::integer_comparison::integer_f64_order;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

/// MATLAB stores complex integer arrays, but does not support arithmetic on
/// them. Arithmetic builtins use this before selecting floating or provider
/// execution paths so exact integer components are never coerced to `f64`.
pub fn is_typed_complex_integer(value: &Value) -> bool {
    matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
}

/// Reject a value that would otherwise enter a floating complex operation.
pub fn reject_typed_complex_integer(value: &Value, builtin: &str) -> BuiltinResult<()> {
    if is_typed_complex_integer(value) {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

/// Reject operations that would consume the lossy `f64` compatibility view of
/// a typed complex integer tensor. MATLAB permits storage/inspection of these
/// values, but not operations on them.
pub fn reject_typed_complex_integer_tensor(
    tensor: &ComplexTensor,
    builtin: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationAtom {
    Number(f64),
    Integer(IntValue),
    Text(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RangeInclusivity {
    pub lower: bool,
    pub upper: bool,
}

impl RangeInclusivity {
    pub const CLOSED: Self = Self {
        lower: true,
        upper: true,
    };

    pub const OPEN: Self = Self {
        lower: false,
        upper: false,
    };

    pub const OPEN_LEFT: Self = Self {
        lower: false,
        upper: true,
    };

    pub const OPEN_RIGHT: Self = Self {
        lower: true,
        upper: false,
    };
}

const VALUE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to validate.",
};

const EXTRA_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Additional validator-specific argument.",
};

const PREDICATE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Validation result.",
}];

const VALIDATOR_INPUTS: [BuiltinParamDescriptor; 2] = [VALUE_INPUT, EXTRA_INPUT];
const VALIDATOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mustBe*(A, ...)",
    inputs: &VALIDATOR_INPUTS,
    outputs: &[],
}];

const PREDICATE_INPUTS: [BuiltinParamDescriptor; 1] = [VALUE_INPUT];
const ISVARNAME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvarname(S)",
    inputs: &PREDICATE_INPUTS,
    outputs: &PREDICATE_OUTPUT,
}];

const NAMEDARGS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = namedargs2cell(S)",
    inputs: &PREDICATE_INPUTS,
    outputs: &[BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Cell row vector of alternating field names and values.",
    }],
}];

const VALIDATION_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.FAILED",
    identifier: Some("RunMat:validators:ValidationFailed"),
    when: "A value does not satisfy the requested validator.",
    message: "argument validation failed",
};

const VALIDATION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ARGUMENT_VALIDATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:validators:InvalidArgument"),
    when: "A validator receives an unsupported argument count or argument type.",
    message: "invalid argument validation input",
};

const VALIDATION_ERRORS: [BuiltinErrorDescriptor; 2] =
    [VALIDATION_ERROR_FAILED, VALIDATION_ERROR_INVALID_ARGUMENT];

pub const VALIDATOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VALIDATOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VALIDATION_ERRORS,
};

pub const ISVARNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISVARNAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub const NAMEDARGS2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NAMEDARGS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VALIDATION_ERRORS,
};

pub fn validation_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: validation failed")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:ValidationFailed"))
        .build()
}

fn invalid_argument_error(builtin: &str, detail: impl AsRef<str>) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        format!("{builtin}: invalid argument")
    } else {
        format!("{builtin}: {detail}")
    };
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:InvalidArgument"))
        .build()
}

fn pass() -> BuiltinResult<Value> {
    Ok(Value::Num(0.0))
}

fn require_args<'a>(
    builtin: &str,
    args: &'a [Value],
    min: usize,
    max: usize,
) -> BuiltinResult<&'a Value> {
    if args.len() < min || args.len() > max {
        return Err(invalid_argument_error(builtin, "invalid number of inputs").into());
    }
    args.first()
        .ok_or_else(|| invalid_argument_error(builtin, "missing value").into())
}

fn require_arg_count(builtin: &str, args: &[Value], min: usize, max: usize) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        return Err(invalid_argument_error(builtin, "invalid number of inputs").into());
    }
    Ok(())
}

fn require_exact_arg_count(builtin: &str, args: &[Value], expected: usize) -> BuiltinResult<()> {
    require_arg_count(builtin, args, expected, expected)
}

fn check_validator(builtin: &str, ok: bool) -> BuiltinResult<Value> {
    if ok {
        pass()
    } else {
        Err(validation_error(builtin, "value does not satisfy validator").into())
    }
}

pub fn dispatch_validator(builtin: &str, args: Vec<Value>) -> BuiltinResult<Value> {
    let value = require_args(builtin, &args, 1, usize::MAX)?;
    match builtin {
        "mustBeA" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(builtin, must_be_a(value, type_names_arg(&args, 1)?)?)
        }
        "mustBeColumn" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_column(value))
        }
        "mustBeFile" => check_validator(builtin, {
            require_exact_arg_count(builtin, &args, 1)?;
            value_texts(value)?.iter().all(|p| Path::new(p).is_file())
        }),
        "mustBeFinite" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_finite(value))
        }
        "mustBeFloat" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_float(value))
        }
        "mustBeFolder" => check_validator(builtin, {
            require_exact_arg_count(builtin, &args, 1)?;
            value_texts(value)?.iter().all(|p| Path::new(p).is_dir())
        }),
        "mustBeGreaterThan" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_greater_than(value, numeric_arg(&args, 1)?),
            )
        }
        "mustBeGreaterThanOrEqual" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_greater_than_or_equal(value, numeric_arg(&args, 1)?),
            )
        }
        "mustBeInRange" => {
            require_arg_count(builtin, &args, 3, 5)?;
            let lower = numeric_arg(&args, 1)?;
            let upper = numeric_arg(&args, 2)?;
            let inclusivity = range_inclusivity_arg(builtin, &args[3..])?;
            check_validator(builtin, value_is_in_range(value, lower, upper, inclusivity))
        }
        "mustBeInteger" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_integer(value))
        }
        "mustBeLessThan" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(builtin, value_is_less_than(value, numeric_arg(&args, 1)?))
        }
        "mustBeLessThanOrEqual" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_is_less_than_or_equal(value, numeric_arg(&args, 1)?),
            )
        }
        "mustBeMember" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(builtin, value_is_member(value, &args[1])?)
        }
        "mustBeNegative" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_negative(value))
        }
        "mustBeNonempty" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, !value_is_empty(value))
        }
        "mustBeNonmissing" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonmissing(value))
        }
        "mustBeNonNan" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_non_nan(value))
        }
        "mustBeNonnegative" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonnegative(value))
        }
        "mustBeNonpositive" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonpositive(value))
        }
        "mustBeNonsparse" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, !matches!(value, Value::SparseTensor(_)))
        }
        "mustBeNonzero" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonzero(value))
        }
        "mustBeNonzeroLengthText" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_nonzero_length_text(value))
        }
        "mustBeNumeric" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_numeric(value))
        }
        "mustBeNumericOrLogical" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_numeric_or_logical(value))
        }
        "mustBePositive" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_positive(value))
        }
        "mustBeReal" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_real(value))
        }
        "mustBeScalarOrEmpty" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_scalar_or_empty(value))
        }
        "mustBeSparse" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, matches!(value, Value::SparseTensor(_)))
        }
        "mustBeText" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_text(value))
        }
        "mustBeTextScalar" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_text_scalar(value))
        }
        "mustBeUnderlyingType" => {
            require_exact_arg_count(builtin, &args, 2)?;
            check_validator(
                builtin,
                value_underlying_type_matches(value, type_names_arg(&args, 1)?)?,
            )
        }
        "mustBeValidVariableName" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(
                builtin,
                value_texts(value)?
                    .iter()
                    .all(|name| is_valid_varname(name)),
            )
        }
        "mustBeVector" => {
            require_exact_arg_count(builtin, &args, 1)?;
            check_validator(builtin, value_is_vector(value)?)
        }
        "validateFunctionSignaturesJSON" => {
            require_exact_arg_count(builtin, &args, 1)?;
            validate_function_signatures_json(value)?;
            pass()
        }
        _ => Err(invalid_argument_error(builtin, "unknown validator").into()),
    }
}

pub fn value_shape_2d(value: &Value) -> (usize, usize) {
    match value {
        Value::Tensor(t) => (t.rows, t.cols),
        Value::SparseTensor(t) => (t.rows, t.cols),
        Value::ComplexTensor(t) => (t.rows, t.cols),
        Value::LogicalArray(a) => {
            let rows = a.shape.first().copied().unwrap_or(0);
            let cols = a.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        Value::Cell(c) => (c.rows, c.cols),
        Value::CharArray(c) => (c.rows, c.cols),
        Value::StringArray(s) => (s.rows, s.cols),
        Value::GpuTensor(handle) => {
            let rows = handle.shape.first().copied().unwrap_or(1);
            let cols = handle.shape.get(1).copied().unwrap_or(1);
            (rows, cols)
        }
        _ => (1, 1),
    }
}

pub fn value_is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.is_empty(),
        Value::SparseTensor(t) => t.rows == 0 || t.cols == 0,
        Value::ComplexTensor(t) => tensor::complex_tensor_element_len(t) == 0,
        Value::LogicalArray(a) => a.data.is_empty(),
        Value::StringArray(s) => s.data.is_empty(),
        Value::CharArray(c) => c.rows == 0 || c.cols == 0,
        Value::Cell(c) => c.data.is_empty(),
        Value::GpuTensor(handle) => handle.shape.contains(&0),
        _ => false,
    }
}

pub fn value_is_finite(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| v.is_finite()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite()),
        Value::LogicalArray(_) | Value::CharArray(_) => true,
        Value::GpuTensor(_) => true,
        _ => false,
    }
}

pub fn value_is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_) => true,
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

pub fn value_is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Complex(_, _) => true,
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none(),
        Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F64 | NumericDType::F32),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            !handle_is_logical(handle) && handle_integer_type(handle).is_none()
        }
        _ => false,
    }
}

pub fn value_is_numeric_or_logical(value: &Value) -> bool {
    value_is_numeric(value) || matches!(value, Value::Bool(_) | Value::LogicalArray(_))
}

pub fn value_is_text(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) => true,
        Value::CharArray(chars) => chars.rows == 1,
        Value::Cell(cell) => cell.data.iter().all(value_is_text),
        _ => false,
    }
}

pub fn value_is_text_scalar(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(strings) => {
            strings.data.len() == 1 && strings.rows == 1 && strings.cols == 1
        }
        Value::CharArray(chars) => chars.rows == 1,
        _ => false,
    }
}

pub fn value_is_nonzero_length_text(value: &Value) -> bool {
    if !value_is_text(value) {
        return false;
    }
    match value {
        Value::String(s) => !s.is_empty(),
        Value::StringArray(s) => s.data.iter().all(|value| !value.is_empty()),
        Value::CharArray(c) => c.rows == 1 && c.cols > 0,
        Value::Cell(c) => c.data.iter().all(value_is_nonzero_length_text),
        _ => false,
    }
}

pub fn value_is_scalar_or_empty(value: &Value) -> bool {
    let (rows, cols) = value_shape_2d(value);
    (rows == 1 && cols == 1) || rows == 0 || cols == 0
}

pub fn value_is_real(value: &Value) -> bool {
    match value {
        Value::Complex(_, im) => *im == 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => t
            .integer_storage()
            .as_ref()
            .expect("checked integer complex storage")
            .imag
            .exact_values()
            .iter()
            .all(IntValue::is_zero),
        Value::ComplexTensor(t) => t.materialize_f64().iter().all(|(_, im)| *im == 0.0),
        _ => true,
    }
}

pub fn value_is_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Num(v) => v.is_finite() && v.fract() == 0.0,
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::Complex(re, im) => *im == 0.0 && re.is_finite() && re.fract() == 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => value_is_real(value),
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *im == 0.0 && re.is_finite() && re.fract() == 0.0),
        Value::GpuTensor(handle) => {
            handle_is_logical(handle) || handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

pub fn value_is_non_nan(value: &Value) -> bool {
    match value {
        Value::Num(v) => !v.is_nan(),
        Value::Complex(re, im) => !re.is_nan() && !im.is_nan(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t).iter().all(|v| !v.is_nan()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| !v.is_nan()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| !re.is_nan() && !im.is_nan()),
        Value::Cell(c) => c.data.iter().all(value_is_non_nan),
        _ => true,
    }
}

pub fn value_is_nonmissing(value: &Value) -> bool {
    value_is_non_nan(value)
}

pub fn value_is_positive(value: &Value) -> bool {
    if let Some(result) = exact_integer_values_all(value, int_is_positive) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v > 0.0)
}

pub fn value_is_negative(value: &Value) -> bool {
    if let Some(result) = exact_integer_values_all(value, int_is_negative) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v < 0.0)
}

pub fn value_is_nonnegative(value: &Value) -> bool {
    if let Some(result) = exact_integer_values_all(value, int_is_nonnegative) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v >= 0.0)
}

pub fn value_is_nonpositive(value: &Value) -> bool {
    if let Some(result) = exact_integer_values_all(value, int_is_nonpositive) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v <= 0.0)
}

pub fn value_is_nonzero(value: &Value) -> bool {
    match value {
        Value::Complex(re, im) => re.is_finite() && im.is_finite() && (*re != 0.0 || *im != 0.0),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let integer_data = t.integer_storage().expect("checked integer data");
            (0..integer_data.len()).all(|index| integer_data.is_nonzero_at(index).unwrap_or(false))
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite() && (*re != 0.0 || *im != 0.0)),
        _ => {
            if let Some(result) = exact_integer_values_all(value, |integer| !integer.is_zero()) {
                return result;
            }
            numeric_values_all(value, |v| v.is_finite() && v != 0.0)
        }
    }
}

pub fn value_is_greater_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering >= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v >= threshold)
}

pub fn value_is_less_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering <= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v <= threshold)
}

pub fn value_is_greater_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Greater)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v > threshold)
}

pub fn value_is_less_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Less)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v < threshold)
}

pub fn value_is_in_range(
    value: &Value,
    lower: f64,
    upper: f64,
    inclusivity: RangeInclusivity,
) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        let lower_ok = int_f64_matches(integer, lower, |ordering| {
            if inclusivity.lower {
                ordering >= Ordering::Equal
            } else {
                ordering == Ordering::Greater
            }
        });
        let upper_ok = int_f64_matches(integer, upper, |ordering| {
            if inclusivity.upper {
                ordering <= Ordering::Equal
            } else {
                ordering == Ordering::Less
            }
        });
        lower_ok && upper_ok
    }) {
        return result;
    }
    numeric_values_all(value, |v| {
        v.is_finite()
            && if inclusivity.lower {
                v >= lower
            } else {
                v > lower
            }
            && if inclusivity.upper {
                v <= upper
            } else {
                v < upper
            }
    })
}

pub fn value_is_column(value: &Value) -> bool {
    let (_, cols) = value_shape_2d(value);
    cols == 1
}

pub fn value_is_vector(value: &Value) -> Result<bool, RuntimeError> {
    let (rows, cols) = value_shape_2d(value);
    Ok((rows == 1 || cols == 1) && !(rows == 0 && cols > 1) && !(cols == 0 && rows > 1))
}

pub fn value_matches_class(value: &Value, class_name: &str) -> bool {
    let requested = class_name.trim();
    if requested.is_empty() {
        return false;
    }
    match requested.to_ascii_lowercase().as_str() {
        "numeric" => value_is_numeric(value),
        "float" => value_is_float(value),
        "integer" => value_has_native_integer_class(value),
        "logical" => matches!(value, Value::Bool(_) | Value::LogicalArray(_)),
        "char" => matches!(value, Value::CharArray(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_)),
        "sparse" => matches!(value, Value::SparseTensor(_)),
        "double" => {
            matches!(value, Value::Num(_) | Value::Complex(_, _))
                || matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F64)
                || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_none())
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F64)
        }
        "single" => {
            matches!(value, Value::Tensor(t) if t.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == NumericDType::F32)
        }
        "gpuarray" => matches!(value, Value::GpuTensor(_)),
        _ => class_name_for_value(value).eq_ignore_ascii_case(requested),
    }
}

pub fn value_has_native_integer_class(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_some(),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub fn value_has_logical_class(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if handle_is_logical(handle))
}

pub fn native_integer_value_is_exact_f64(value: &Value) -> bool {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::SparseTensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            storage.real.exact_values().iter().all(exact)
                && storage.imag.exact_values().iter().all(exact)
        }),
        Value::Cell(cell) => cell.data.iter().all(native_integer_value_is_exact_f64),
        Value::Struct(value) => value.fields.values().all(native_integer_value_is_exact_f64),
        Value::OutputList(values) => values.iter().all(native_integer_value_is_exact_f64),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle)
            .is_none_or(|element_type| element_type.element_size() <= 4),
        _ => true,
    }
}

/// Check a native integer value against a binary64 boundary without rejecting a
/// resident 64-bit value solely because its contents are not visible in handle
/// metadata. Compatibility admission must run before calling this helper.
pub async fn native_integer_value_is_exact_f64_async(value: &Value) -> Result<bool, RuntimeError> {
    if native_integer_value_is_exact_f64(value) {
        return Ok(true);
    }
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some()) {
        let Value::GpuTensor(handle) = value else {
            unreachable!();
        };
        let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| {
                build_runtime_error(
                    "integer exactness check: no acceleration provider owns the input handle",
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
            })?;
        let expected_type = handle_integer_type(handle).expect("integer type checked above");
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle).is_some()
            || !crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
                handle,
                None,
                Some(expected_type),
                false,
            )
        {
            return Err(build_runtime_error(
                "integer exactness check: input handle has contradictory integer metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let snapshot = crate::builtins::common::gpu_helpers::snapshot_handle_metadata(handle);
        let gathered = provider.download_integer(handle).await.map_err(|error| {
            build_runtime_error(format!(
                "integer exactness check: owner-preserving download failed: {error}"
            ))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        });
        crate::builtins::common::gpu_helpers::restore_handle_metadata(handle, &snapshot);
        let gathered = gathered?;
        if gathered.shape != handle.shape || gathered.data.element_type() != expected_type {
            return Err(build_runtime_error(
                "integer exactness check: provider returned contradictory integer payload metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
        let values_exact = match &gathered.data {
            runmat_accelerate_api::HostIntegerDataOwned::I8(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I16(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I32(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I64(values) => {
                values.iter().all(|value| exact(&IntValue::I64(*value)))
            }
            runmat_accelerate_api::HostIntegerDataOwned::U8(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U16(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U32(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U64(values) => {
                values.iter().all(|value| exact(&IntValue::U64(*value)))
            }
        };
        return Ok(values_exact);
    }
    Ok(false)
}

pub fn must_be_a(value: &Value, class_names: Vec<String>) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| value_matches_class(value, class_name)))
}

pub fn value_underlying_type_matches(
    value: &Value,
    class_names: Vec<String>,
) -> Result<bool, RuntimeError> {
    Ok(class_names
        .iter()
        .any(|class_name| underlying_type_matches(value, class_name)))
}

pub fn value_is_member(value: &Value, set: &Value) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    let allowed = atoms(set)?;
    value_is_member_atoms_inner(&values, &allowed)
}

pub fn value_is_member_atoms(
    value: &Value,
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    let values = atoms(value)?;
    value_is_member_atoms_inner(&values, allowed)
}

fn value_is_member_atoms_inner(
    values: &[ValidationAtom],
    allowed: &[ValidationAtom],
) -> Result<bool, RuntimeError> {
    Ok(values
        .iter()
        .all(|value| allowed.iter().any(|allowed| atom_eq(value, allowed))))
}

pub fn atoms(value: &Value) -> Result<Vec<ValidationAtom>, RuntimeError> {
    match value {
        Value::Num(v) => Ok(vec![ValidationAtom::Number(*v)]),
        Value::Int(v) => Ok(vec![ValidationAtom::Integer(v.clone())]),
        Value::Bool(v) => Ok(vec![ValidationAtom::Bool(*v)]),
        Value::String(s) => Ok(vec![ValidationAtom::Text(s.clone())]),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![ValidationAtom::Text(chars_to_string(c))]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return Ok(storage
                    .exact_values()
                    .into_iter()
                    .map(ValidationAtom::Integer)
                    .collect());
            }
            Ok(tensor::tensor_values_f64_cow(t)
                .iter()
                .copied()
                .map(ValidationAtom::Number)
                .collect())
        }
        Value::SparseTensor(t) => sparse_atoms(t),
        Value::LogicalArray(a) => Ok(a
            .data
            .iter()
            .map(|v| ValidationAtom::Bool(*v != 0))
            .collect()),
        Value::StringArray(s) => Ok(s.data.iter().cloned().map(ValidationAtom::Text).collect()),
        Value::Cell(c) => {
            let mut out = Vec::new();
            for entry in &c.data {
                out.extend(atoms(entry)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument_error(
            "mustBeMember",
            "unsupported member value type",
        )),
    }
}

fn sparse_atoms(t: &SparseTensor) -> Result<Vec<ValidationAtom>, RuntimeError> {
    let numel = t.rows.saturating_mul(t.cols);
    let mut out = Vec::with_capacity(numel.min(t.nnz().saturating_add(1)));
    if let Some(storage) = t.integer_storage() {
        out.extend(
            storage
                .exact_values()
                .into_iter()
                .map(ValidationAtom::Integer),
        );
        if storage.len() < numel {
            let zero = storage
                .zeros_like(1)
                .value_at(0)
                .expect("one-element zero storage");
            out.push(ValidationAtom::Integer(zero));
        }
        return Ok(out);
    }
    let values = t.materialize_f64();
    out.extend(values.iter().copied().map(ValidationAtom::Number));
    if values.len() < numel {
        out.push(ValidationAtom::Number(0.0));
    }
    Ok(out)
}

fn atom_eq(left: &ValidationAtom, right: &ValidationAtom) -> bool {
    match (left, right) {
        (ValidationAtom::Number(a), ValidationAtom::Number(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Integer(b)) => a == b,
        (ValidationAtom::Integer(a), ValidationAtom::Number(b))
        | (ValidationAtom::Number(b), ValidationAtom::Integer(a)) => {
            integer_f64_order(a.clone(), *b) == Some(Ordering::Equal)
        }
        (ValidationAtom::Text(a), ValidationAtom::Text(b)) => a == b,
        (ValidationAtom::Bool(a), ValidationAtom::Bool(b)) => a == b,
        _ => false,
    }
}

fn exact_integer_values_all(
    value: &Value,
    pred: impl Fn(&IntValue) -> bool + Copy,
) -> Option<bool> {
    match value {
        Value::Int(value) => Some(pred(value)),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| integer_storage_all(storage, pred)),
        Value::SparseTensor(tensor) => tensor.integer_storage().map(|storage| {
            let numel = tensor.rows.saturating_mul(tensor.cols);
            integer_storage_all(storage, pred)
                && (storage.len() >= numel || integer_storage_zero_satisfies(storage, pred))
        }),
        Value::ComplexTensor(tensor) => tensor.integer_storage().map(|storage| {
            (0..storage.len()).all(|index| {
                let Some(real) = storage.real.value_at(index) else {
                    return false;
                };
                let Some(imag) = storage.imag.value_at(index) else {
                    return false;
                };
                imag.is_zero() && pred(&real)
            })
        }),
        _ => None,
    }
}

fn integer_storage_all(storage: &IntegerStorage, pred: impl Fn(&IntValue) -> bool + Copy) -> bool {
    (0..storage.len()).all(|index| {
        storage
            .value_at(index)
            .as_ref()
            .is_some_and(|value| pred(value))
    })
}

fn integer_storage_zero_satisfies(
    storage: &IntegerStorage,
    pred: impl Fn(&IntValue) -> bool,
) -> bool {
    storage.zeros_like(1).value_at(0).as_ref().is_some_and(pred)
}

fn int_f64_matches(integer: &IntValue, threshold: f64, pred: impl Fn(Ordering) -> bool) -> bool {
    integer_f64_order(integer.clone(), threshold).is_some_and(pred)
}

fn int_is_positive(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value > 0,
        IntValue::I16(value) => *value > 0,
        IntValue::I32(value) => *value > 0,
        IntValue::I64(value) => *value > 0,
        IntValue::U8(value) => *value > 0,
        IntValue::U16(value) => *value > 0,
        IntValue::U32(value) => *value > 0,
        IntValue::U64(value) => *value > 0,
    }
}

fn int_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn int_is_nonnegative(value: &IntValue) -> bool {
    !int_is_negative(value)
}

fn int_is_nonpositive(value: &IntValue) -> bool {
    !int_is_positive(value)
}

fn numeric_values_all(value: &Value, pred: impl Fn(f64) -> bool) -> bool {
    match value {
        Value::Num(v) => pred(*v),
        Value::Int(v) => pred(v.to_f64()),
        Value::Bool(v) => pred(if *v { 1.0 } else { 0.0 }),
        Value::LogicalArray(a) => a.data.iter().map(|v| f64::from(*v != 0)).all(pred),
        Value::Tensor(t) => tensor::tensor_values_f64(t).into_iter().all(pred),
        Value::SparseTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            let numel = t.rows.saturating_mul(t.cols);
            (0..storage.len()).all(|index| {
                pred(
                    storage
                        .value_at(index)
                        .expect("sparse integer storage length is consistent")
                        .to_f64(),
                )
            }) && (storage.len() >= numel || pred(0.0))
        }
        Value::SparseTensor(t) => {
            let numel = t.rows.saturating_mul(t.cols);
            let values = t.materialize_f64();
            values.iter().copied().all(&pred) && (values.len() >= numel || pred(0.0))
        }
        Value::Complex(re, im) => *im == 0.0 && pred(*re),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let storage = t
                .integer_storage()
                .expect("integer storage was checked above");
            (0..storage.len()).all(|index| {
                let real = storage
                    .real
                    .value_at(index)
                    .expect("complex integer real storage length is consistent");
                let imag = storage
                    .imag
                    .value_at(index)
                    .expect("complex integer imaginary storage length is consistent");
                imag.is_zero() && pred(real.to_f64())
            })
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *im == 0.0 && pred(*re)),
        _ => false,
    }
}

fn numeric_arg(args: &[Value], index: usize) -> Result<f64, RuntimeError> {
    match args.get(index) {
        Some(Value::Num(v)) => Ok(*v),
        Some(Value::Int(v)) => Ok(v.to_f64()),
        Some(Value::Tensor(t)) if tensor::is_scalar_tensor(t) => Ok(tensor::tensor_value_f64(t, 0)),
        Some(other) => Err(invalid_argument_error(
            "argumentValidation",
            format!(
                "expected numeric scalar argument, got {}",
                class_name_for_value(other)
            ),
        )),
        None => Err(invalid_argument_error(
            "argumentValidation",
            "missing numeric scalar argument",
        )),
    }
}

fn type_names_arg(args: &[Value], index: usize) -> Result<Vec<String>, RuntimeError> {
    match args.get(index) {
        Some(value) => value_texts(value),
        None => Err(invalid_argument_error(
            "argumentValidation",
            "missing type name argument",
        )),
    }
}

fn range_inclusivity_arg(builtin: &str, args: &[Value]) -> Result<RangeInclusivity, RuntimeError> {
    match args {
        [] => Ok(RangeInclusivity::CLOSED),
        [flag] => range_inclusivity_single_flag(builtin, text_scalar_arg(builtin, flag)?.as_str()),
        [lower, upper] => {
            let lower =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, lower)?.as_str())?;
            let upper =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, upper)?.as_str())?;
            Ok(RangeInclusivity { lower, upper })
        }
        _ => Err(invalid_argument_error(
            builtin,
            "invalid range inclusivity flags",
        )),
    }
}

fn range_inclusivity_single_flag(
    builtin: &str,
    flag: &str,
) -> Result<RangeInclusivity, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(RangeInclusivity::CLOSED),
        "exclusive" => Ok(RangeInclusivity::OPEN),
        "exclude-lower" | "openleft" | "open-left" => Ok(RangeInclusivity::OPEN_LEFT),
        "exclude-upper" | "openright" | "open-right" => Ok(RangeInclusivity::OPEN_RIGHT),
        _ => Err(invalid_argument_error(
            builtin,
            "range flag must be 'inclusive', 'exclusive', 'exclude-lower', or 'exclude-upper'",
        )),
    }
}

fn range_bound_inclusive_flag(builtin: &str, flag: &str) -> Result<bool, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(true),
        "exclusive" => Ok(false),
        _ => Err(invalid_argument_error(
            builtin,
            "range bound flag must be 'inclusive' or 'exclusive'",
        )),
    }
}

fn text_scalar_arg(builtin: &str, value: &Value) -> Result<String, RuntimeError> {
    let texts = value_texts(value)?;
    match texts.as_slice() {
        [text] => Ok(text.clone()),
        _ => Err(invalid_argument_error(builtin, "expected text scalar")),
    }
}

fn value_texts(value: &Value) -> Result<Vec<String>, RuntimeError> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::StringArray(s) => Ok(s.data.clone()),
        Value::CharArray(c) if c.rows == 1 => Ok(vec![chars_to_string(c)]),
        Value::Cell(c) => {
            let mut out = Vec::with_capacity(c.data.len());
            for entry in &c.data {
                out.extend(value_texts(entry)?);
            }
            Ok(out)
        }
        other => Err(invalid_argument_error(
            "argumentValidation",
            format!("expected text, got {}", class_name_for_value(other)),
        )),
    }
}

fn chars_to_string(chars: &CharArray) -> String {
    chars.data.iter().collect()
}

pub fn isvarname_value(value: &Value) -> bool {
    value_texts(value)
        .map(|names| names.iter().all(|name| is_valid_varname(name)))
        .unwrap_or(false)
}

pub fn namedargs2cell_value(value: Value) -> BuiltinResult<Value> {
    let Value::Struct(struct_value) = value else {
        return Err(
            invalid_argument_error("namedargs2cell", "input must be a scalar struct").into(),
        );
    };
    let mut data = Vec::with_capacity(struct_value.fields.len().saturating_mul(2));
    for (field, value) in struct_value.fields {
        data.push(Value::String(field));
        data.push(value);
    }
    let cols = data.len();
    let cell = CellArray::new(data, 1, cols)
        .map_err(|err| invalid_argument_error("namedargs2cell", err))?;
    Ok(Value::Cell(cell))
}

pub fn validate_function_signatures_json(value: &Value) -> BuiltinResult<()> {
    for text in value_texts(value)? {
        serde_json::from_str::<serde_json::Value>(&text).map_err(|err| {
            invalid_argument_error(
                "validateFunctionSignaturesJSON",
                format!("invalid JSON signature payload: {err}"),
            )
        })?;
    }
    Ok(())
}

fn bool_type(
    _: &[runmat_builtins::Type],
    _: &runmat_builtins::ResolveContext,
) -> runmat_builtins::Type {
    runmat_builtins::Type::Bool
}

fn any_type(
    _: &[runmat_builtins::Type],
    _: &runmat_builtins::ResolveContext,
) -> runmat_builtins::Type {
    runmat_builtins::Type::Unknown
}

#[runtime_builtin(
    name = "isvarname",
    category = "argument-validation",
    summary = "Return true when text is a valid MATLAB variable name.",
    type_resolver(bool_type),
    descriptor(self::ISVARNAME_DESCRIPTOR),
    builtin_path = "crate::builtins::common::validation"
)]
fn isvarname_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(isvarname_value(&value)))
}

#[runtime_builtin(
    name = "namedargs2cell",
    category = "argument-validation",
    summary = "Convert a scalar name-value struct to an alternating name/value cell row.",
    type_resolver(any_type),
    descriptor(self::NAMEDARGS2CELL_DESCRIPTOR),
    builtin_path = "crate::builtins::common::validation"
)]
fn namedargs2cell_builtin(value: Value) -> BuiltinResult<Value> {
    namedargs2cell_value(value)
}

macro_rules! validator_builtin {
    ($func:ident, $name:literal) => {
        #[runtime_builtin(
            name = $name,
            category = "argument-validation",
            summary = "Validate an input argument and throw if the constraint is not satisfied.",
            sink = true,
            suppress_auto_output = true,
            descriptor(self::VALIDATOR_DESCRIPTOR),
            builtin_path = "crate::builtins::common::validation"
        )]
        fn $func(args: Vec<Value>) -> BuiltinResult<Value> {
            dispatch_validator($name, args)
        }
    };
}

validator_builtin!(must_be_a_builtin, "mustBeA");
validator_builtin!(must_be_column_builtin, "mustBeColumn");
validator_builtin!(must_be_file_builtin, "mustBeFile");
validator_builtin!(must_be_finite_builtin, "mustBeFinite");
validator_builtin!(must_be_float_builtin, "mustBeFloat");
validator_builtin!(must_be_folder_builtin, "mustBeFolder");
validator_builtin!(must_be_greater_than_builtin, "mustBeGreaterThan");
validator_builtin!(
    must_be_greater_than_or_equal_builtin,
    "mustBeGreaterThanOrEqual"
);
validator_builtin!(must_be_in_range_builtin, "mustBeInRange");
validator_builtin!(must_be_integer_builtin, "mustBeInteger");
validator_builtin!(must_be_less_than_builtin, "mustBeLessThan");
validator_builtin!(must_be_less_than_or_equal_builtin, "mustBeLessThanOrEqual");
validator_builtin!(must_be_member_builtin, "mustBeMember");
validator_builtin!(must_be_negative_builtin, "mustBeNegative");
validator_builtin!(must_be_nonempty_builtin, "mustBeNonempty");
validator_builtin!(must_be_nonmissing_builtin, "mustBeNonmissing");
validator_builtin!(must_be_non_nan_builtin, "mustBeNonNan");
validator_builtin!(must_be_nonnegative_builtin, "mustBeNonnegative");
validator_builtin!(must_be_nonpositive_builtin, "mustBeNonpositive");
validator_builtin!(must_be_nonsparse_builtin, "mustBeNonsparse");
validator_builtin!(must_be_nonzero_builtin, "mustBeNonzero");
validator_builtin!(
    must_be_nonzero_length_text_builtin,
    "mustBeNonzeroLengthText"
);
validator_builtin!(must_be_numeric_builtin, "mustBeNumeric");
validator_builtin!(must_be_numeric_or_logical_builtin, "mustBeNumericOrLogical");
validator_builtin!(must_be_positive_builtin, "mustBePositive");
validator_builtin!(must_be_real_builtin, "mustBeReal");
validator_builtin!(must_be_scalar_or_empty_builtin, "mustBeScalarOrEmpty");
validator_builtin!(must_be_sparse_builtin, "mustBeSparse");
validator_builtin!(must_be_text_builtin, "mustBeText");
validator_builtin!(must_be_text_scalar_builtin, "mustBeTextScalar");
validator_builtin!(must_be_underlying_type_builtin, "mustBeUnderlyingType");
validator_builtin!(
    must_be_valid_variable_name_builtin,
    "mustBeValidVariableName"
);
validator_builtin!(must_be_vector_builtin, "mustBeVector");
validator_builtin!(
    validate_function_signatures_json_builtin,
    "validateFunctionSignaturesJSON"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, StringArray,
        StructValue, Tensor,
    };

    fn ok(builtin: &str, args: Vec<Value>) {
        dispatch_validator(builtin, args).unwrap_or_else(|err| {
            panic!("{builtin} unexpectedly failed: {err}");
        });
    }

    fn err(builtin: &str, args: Vec<Value>) {
        assert!(
            dispatch_validator(builtin, args).is_err(),
            "{builtin} unexpectedly passed"
        );
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
    }

    fn sparse(values: Vec<f64>) -> Value {
        Value::SparseTensor(SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], values).unwrap())
    }

    #[test]
    fn resident_integer_exactness_is_class_conservative() {
        use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};

        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 1,
        };
        runmat_accelerate_api::set_handle_integer_type(&handle, IntegerElementType::U32);
        assert!(native_integer_value_is_exact_f64(&Value::GpuTensor(
            handle.clone()
        )));
        runmat_accelerate_api::set_handle_integer_type(&handle, IntegerElementType::I64);
        assert!(!native_integer_value_is_exact_f64(&Value::GpuTensor(
            handle.clone()
        )));
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }

    #[test]
    fn resident_wide_integer_exactness_is_decided_from_gathered_values() {
        use crate::builtins::common::test_support;
        use futures::executor::block_on;
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            for (value, expected) in [
                (9_007_199_254_740_992_u64, true),
                (9_007_199_254_740_993_u64, false),
            ] {
                let handle = provider
                    .upload_integer(&HostIntegerTensorView {
                        data: HostIntegerDataView::U64(std::slice::from_ref(&value)),
                        shape: &[1, 1],
                    })
                    .expect("upload resident uint64");
                assert_eq!(
                    block_on(native_integer_value_is_exact_f64_async(&Value::GpuTensor(
                        handle.clone()
                    )))
                    .expect("exactness check"),
                    expected
                );
                provider.free(&handle).expect("free resident uint64");
                runmat_accelerate_api::clear_handle_metadata(&handle);
            }
        });
    }

    #[test]
    fn resident_wide_integer_exactness_rejects_contradictory_source_metadata() {
        use crate::builtins::common::test_support;
        use futures::executor::block_on;
        use runmat_accelerate_api::{
            HostIntegerDataView, HostIntegerTensorView, ProviderPrecision,
        };

        test_support::with_test_provider(|provider| {
            let value = 9_007_199_254_740_992_u64;
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(std::slice::from_ref(&value)),
                    shape: &[1, 1],
                })
                .expect("upload resident uint64");
            runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F64);
            let error = block_on(native_integer_value_is_exact_f64_async(&Value::GpuTensor(
                handle.clone(),
            )))
            .expect_err("contradictory integer metadata must reject");
            assert!(error.message().contains("contradictory integer metadata"));
            provider.free(&handle).ok();
            runmat_accelerate_api::clear_handle_metadata(&handle);
        });
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
    }

    #[test]
    fn numeric_validators_check_all_elements() {
        let ok = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(ok)]).is_ok());

        let bad = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        assert!(dispatch_validator("mustBePositive", vec![Value::Tensor(bad)]).is_err());
    }

    #[test]
    fn numeric_validators_read_typed_integer_storage_exactly() {
        let positive =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("positive");
        ok("mustBePositive", vec![Value::Tensor(positive)]);

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1, -2]), vec![1, 2]).expect("negative");
        ok("mustBeNegative", vec![Value::Tensor(negative)]);

        let zero = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).expect("zero");
        err("mustBeNonzero", vec![Value::Tensor(zero)]);

        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        assert_eq!(wide as f64, adjacent as f64);
        let wide_nonzero =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide");
        ok("mustBeNonzero", vec![Value::Tensor(wide_nonzero)]);

        let complex_nonzero = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![0]),
                IntegerStorage::U64(vec![wide]),
            )
            .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        ok("mustBeNonzero", vec![Value::ComplexTensor(complex_nonzero)]);
    }

    #[test]
    fn value_is_empty_uses_typed_integer_storage_length() {
        let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
        assert!(!value_is_empty(&Value::Tensor(scalar)));

        let empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("empty tensor");
        assert!(value_is_empty(&Value::Tensor(empty)));

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        assert!(!value_is_empty(&Value::ComplexTensor(complex)));

        let empty_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![]), IntegerStorage::I16(vec![]))
                .expect("empty complex integer storage"),
            vec![0, 0],
        )
        .expect("empty complex integer tensor");
        assert!(value_is_empty(&Value::ComplexTensor(empty_complex)));
    }

    #[test]
    fn finite_integer_and_nan_predicates_read_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 2]), vec![1, 3]).unwrap();
        let value = Value::Tensor(tensor);

        assert!(value_is_finite(&value));
        assert!(value_is_integer(&value));
        assert!(value_is_non_nan(&value));
        ok("mustBeFinite", vec![value.clone()]);
        ok("mustBeInteger", vec![value.clone()]);
        ok("mustBeNonNan", vec![value]);

        let sparse = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 2],
                vec![0, 1],
                IntegerStorage::U8(vec![1, 2]),
            )
            .unwrap(),
        );
        assert!(value_is_finite(&sparse));
        assert!(value_is_integer(&sparse));
        assert!(value_is_non_nan(&sparse));
    }

    #[test]
    fn real_and_integer_predicates_read_authoritative_complex_integer_storage() {
        let real_storage = IntegerStorage::I16(vec![1, -2]);
        let zero_imag = IntegerStorage::I16(vec![0, 0]);
        let real_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(real_storage, zero_imag).unwrap(),
            vec![1, 2],
        )
        .unwrap();
        let value = Value::ComplexTensor(real_complex);
        assert!(value_is_finite(&value));
        assert!(value_is_real(&value));
        assert!(value_is_integer(&value));
        assert!(value_is_non_nan(&value));

        let nonreal_complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let value = Value::ComplexTensor(nonreal_complex);
        assert!(!value_is_real(&value));
        assert!(!value_is_integer(&value));
    }

    #[test]
    fn threshold_validators_read_typed_sparse_and_complex_integer_storage() {
        let sparse =
            SparseTensor::new_integer(1, 2, vec![0, 1, 1], vec![0], IntegerStorage::U8(vec![1]))
                .unwrap();
        ok("mustBeNonnegative", vec![Value::SparseTensor(sparse)]);

        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![2]), IntegerStorage::I16(vec![0]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        ok("mustBePositive", vec![Value::ComplexTensor(complex)]);
    }

    #[test]
    fn member_validator_accepts_numeric_and_text_sets() {
        let allowed = Tensor::new(vec![1.0, 3.0, 5.0], vec![1, 3]).unwrap();
        assert!(dispatch_validator(
            "mustBeMember",
            vec![Value::Num(3.0), Value::Tensor(allowed)]
        )
        .is_ok());

        let allowed = StringArray::new(vec!["on".into(), "off".into()], vec![1, 2]).unwrap();
        assert!(dispatch_validator(
            "mustBeMember",
            vec![Value::String("on".into()), Value::StringArray(allowed)]
        )
        .is_ok());
    }

    #[test]
    fn class_validators_use_native_integer_storage_metadata() {
        let dense = integer_tensor(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![1, 2],
        );
        ok(
            "mustBeA",
            vec![dense.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![dense.clone()]);
        err("mustBeA", vec![dense, Value::String("double".into())]);

        let sparse = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 1],
                vec![1],
                IntegerStorage::I64(vec![i64::MIN]),
            )
            .unwrap(),
        );
        ok(
            "mustBeA",
            vec![sparse.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![sparse.clone()]);
        err("mustBeA", vec![sparse, Value::String("double".into())]);

        let typed_complex = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![1]),
                    IntegerStorage::I16(vec![2]),
                )
                .unwrap(),
                vec![1, 1],
            )
            .unwrap(),
        );
        ok(
            "mustBeA",
            vec![typed_complex.clone(), Value::String("integer".into())],
        );
        err("mustBeFloat", vec![typed_complex]);
    }

    #[test]
    fn class_validators_use_gpu_integer_metadata_without_gather() {
        use crate::builtins::common::test_support;
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu tensor");
            let gpu = Value::GpuTensor(handle);

            ok(
                "mustBeA",
                vec![gpu.clone(), Value::String("integer".into())],
            );
            ok("mustBeInteger", vec![gpu.clone()]);
            err("mustBeFloat", vec![gpu.clone()]);
            err("mustBeA", vec![gpu, Value::String("double".into())]);
        });
    }

    #[test]
    fn member_validator_compares_native_integers_exactly() {
        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        assert_eq!(wide as f64, adjacent as f64);

        let allowed = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        ok(
            "mustBeMember",
            vec![
                Value::Int(IntValue::U64(wide)),
                Value::Tensor(allowed.clone()),
            ],
        );
        err(
            "mustBeMember",
            vec![Value::Int(IntValue::U64(adjacent)), Value::Tensor(allowed)],
        );

        let sparse_allowed = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![1],
            IntegerStorage::U64(vec![wide]),
        )
        .unwrap();
        ok(
            "mustBeMember",
            vec![
                Value::Int(IntValue::U64(0)),
                Value::SparseTensor(sparse_allowed),
            ],
        );
    }

    #[test]
    fn member_validator_does_not_equate_wide_integer_with_rounded_double() {
        let wide = 9_007_199_254_740_993_u64;
        let rounded = (wide - 1) as f64;
        assert_eq!(wide as f64, rounded);

        err(
            "mustBeMember",
            vec![Value::Int(IntValue::U64(wide)), Value::Num(rounded)],
        );
    }

    #[test]
    fn text_and_varname_validators_follow_core_shapes() {
        assert!(dispatch_validator(
            "mustBeNonzeroLengthText",
            vec![Value::CharArray(CharArray::new_row("alpha"))]
        )
        .is_ok());
        assert!(isvarname_value(&Value::String("alpha_1".into())));
        assert!(!isvarname_value(&Value::String("1alpha".into())));
    }

    #[test]
    fn namedargs2cell_preserves_field_order() {
        let mut st = StructValue::new();
        st.insert("Name", Value::String("Ada".into()));
        st.insert("Value", Value::Num(7.0));
        let out = namedargs2cell_value(Value::Struct(st)).expect("namedargs2cell");
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 4);
        assert_eq!(cell.data[0], Value::String("Name".into()));
        assert_eq!(cell.data[2], Value::String("Value".into()));
    }

    #[test]
    fn validator_surface_accepts_and_rejects_representative_values() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("data.txt");
        std::fs::write(&file_path, "ok").unwrap();
        let dir_text = Value::String(temp_dir.path().to_string_lossy().into_owned());
        let file_text = Value::String(file_path.to_string_lossy().into_owned());

        ok(
            "mustBeA",
            vec![Value::Num(1.0), Value::String("double".into())],
        );
        err(
            "mustBeA",
            vec![Value::String("x".into()), Value::String("double".into())],
        );
        ok("mustBeColumn", vec![tensor(vec![1.0, 2.0], 2, 1)]);
        err("mustBeColumn", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeFile", vec![file_text.clone()]);
        err("mustBeFile", vec![dir_text.clone()]);
        ok("mustBeFolder", vec![dir_text.clone()]);
        err("mustBeFolder", vec![file_text.clone()]);
        ok("mustBeFinite", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        err("mustBeFinite", vec![Value::Num(f64::INFINITY)]);
        ok("mustBeFloat", vec![Value::Num(1.0)]);
        err("mustBeFloat", vec![Value::Int(IntValue::I32(1))]);
        ok("mustBeInteger", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeInteger", vec![Value::Bool(true)]);
        ok(
            "mustBeInteger",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeInteger", vec![Value::Num(1.5)]);
        ok("mustBeNumeric", vec![Value::Complex(1.0, 2.0)]);
        err("mustBeNumeric", vec![Value::String("1".into())]);
        ok(
            "mustBeNumericOrLogical",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeNumericOrLogical", vec![Value::String("true".into())]);
        ok("mustBeReal", vec![Value::Complex(1.0, 0.0)]);
        err("mustBeReal", vec![Value::Complex(1.0, 1.0)]);
        ok("mustBeVector", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        err("mustBeVector", vec![tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2)]);
        ok("mustBeScalarOrEmpty", vec![Value::Num(1.0)]);
        err("mustBeScalarOrEmpty", vec![tensor(vec![1.0, 2.0], 1, 2)]);
        ok("mustBeSparse", vec![sparse(vec![1.0])]);
        err("mustBeSparse", vec![Value::Num(1.0)]);
        ok("mustBeNonsparse", vec![Value::Num(1.0)]);
        err("mustBeNonsparse", vec![sparse(vec![1.0])]);
        ok(
            "mustBeText",
            vec![Value::CharArray(CharArray::new_row("abc"))],
        );
        err("mustBeText", vec![Value::Num(1.0)]);
        ok("mustBeTextScalar", vec![Value::String("abc".into())]);
        err(
            "mustBeTextScalar",
            vec![Value::StringArray(
                StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
            )],
        );
        ok(
            "mustBeNonzeroLengthText",
            vec![Value::StringArray(
                StringArray::new_2d(vec!["a".into(), "b".into()], 1, 2).unwrap(),
            )],
        );
        err(
            "mustBeNonzeroLengthText",
            vec![Value::String(String::new())],
        );
        ok("mustBeNonempty", vec![Value::String("x".into())]);
        err(
            "mustBeNonempty",
            vec![Value::StringArray(
                StringArray::new_2d(vec![], 0, 0).unwrap(),
            )],
        );
        ok("mustBeNonmissing", vec![Value::Num(1.0)]);
        err("mustBeNonmissing", vec![Value::Num(f64::NAN)]);
        ok("mustBeNonNan", vec![Value::Complex(1.0, 0.0)]);
        err("mustBeNonNan", vec![Value::Complex(f64::NAN, 0.0)]);
        ok(
            "mustBeUnderlyingType",
            vec![Value::Int(IntValue::I16(1)), Value::String("int16".into())],
        );
        err(
            "mustBeUnderlyingType",
            vec![Value::Bool(true), Value::String("double".into())],
        );
        ok(
            "mustBeValidVariableName",
            vec![Value::String("alpha_1".into())],
        );
        err(
            "mustBeValidVariableName",
            vec![Value::String("_alpha".into())],
        );
        ok(
            "mustBeMember",
            vec![
                Value::String("on".into()),
                Value::Cell(
                    CellArray::new(
                        vec![Value::String("on".into()), Value::String("off".into())],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
            ],
        );
        err(
            "mustBeMember",
            vec![
                Value::String("bad".into()),
                Value::Cell(
                    CellArray::new(
                        vec![Value::String("on".into()), Value::String("off".into())],
                        1,
                        2,
                    )
                    .unwrap(),
                ),
            ],
        );
    }

    #[test]
    fn numeric_threshold_validators_cover_boundaries() {
        ok("mustBePositive", vec![Value::Num(1.0)]);
        ok("mustBePositive", vec![Value::Bool(true)]);
        err("mustBePositive", vec![Value::Num(0.0)]);
        ok("mustBeNegative", vec![Value::Num(-1.0)]);
        err("mustBeNegative", vec![Value::Num(0.0)]);
        ok("mustBeNonnegative", vec![Value::Num(0.0)]);
        ok(
            "mustBeNonnegative",
            vec![Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
            )],
        );
        err("mustBeNonnegative", vec![Value::Num(-1.0)]);
        ok("mustBeNonpositive", vec![Value::Num(0.0)]);
        err("mustBeNonpositive", vec![Value::Num(1.0)]);
        ok("mustBeNonzero", vec![Value::Complex(0.0, 1.0)]);
        err("mustBeNonzero", vec![Value::Num(0.0)]);
        ok("mustBeGreaterThan", vec![Value::Num(2.0), Value::Num(1.0)]);
        err("mustBeGreaterThan", vec![Value::Num(1.0), Value::Num(1.0)]);
        ok(
            "mustBeGreaterThanOrEqual",
            vec![Value::Num(1.0), Value::Num(1.0)],
        );
        err(
            "mustBeGreaterThanOrEqual",
            vec![Value::Num(0.0), Value::Num(1.0)],
        );
        ok("mustBeLessThan", vec![Value::Num(0.0), Value::Num(1.0)]);
        err("mustBeLessThan", vec![Value::Num(1.0), Value::Num(1.0)]);
        ok(
            "mustBeLessThanOrEqual",
            vec![Value::Num(1.0), Value::Num(1.0)],
        );
        err(
            "mustBeLessThanOrEqual",
            vec![Value::Num(2.0), Value::Num(1.0)],
        );
    }

    #[test]
    fn threshold_validators_read_typed_integer_storage_exactly() {
        let lower = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("lower");
        ok(
            "mustBeGreaterThan",
            vec![Value::Num(2.0), Value::Tensor(lower)],
        );

        let upper = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("upper");
        ok(
            "mustBeLessThan",
            vec![Value::Num(2.0), Value::Tensor(upper)],
        );

        let range_lower =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("range lower");
        let range_upper =
            Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("range upper");
        ok(
            "mustBeInRange",
            vec![
                Value::Num(2.0),
                Value::Tensor(range_lower),
                Value::Tensor(range_upper),
            ],
        );

        let wide = 9_007_199_254_740_993_u64;
        let adjacent = wide - 1;
        let rounded = adjacent as f64;
        assert_eq!(wide as f64, rounded);

        let wide_value =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("wide value");
        ok(
            "mustBeGreaterThan",
            vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
        );
        err(
            "mustBeLessThanOrEqual",
            vec![Value::Tensor(wide_value.clone()), Value::Num(rounded)],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Tensor(wide_value.clone()),
                Value::Num(0.0),
                Value::Num(rounded),
            ],
        );

        let complex_value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![wide]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("complex integer storage"),
            vec![1, 1],
        )
        .expect("complex integer tensor");
        ok(
            "mustBeGreaterThan",
            vec![Value::ComplexTensor(complex_value), Value::Num(rounded)],
        );

        let sparse_value = Value::SparseTensor(
            SparseTensor::new_integer(1, 1, vec![0, 1], vec![0], IntegerStorage::U64(vec![wide]))
                .expect("sparse integer value"),
        );
        ok(
            "mustBeGreaterThan",
            vec![sparse_value.clone(), Value::Num(rounded)],
        );
        err(
            "mustBeInRange",
            vec![sparse_value, Value::Num(0.0), Value::Num(rounded)],
        );
    }

    #[test]
    fn in_range_supports_interval_flags_and_rejects_extra_inputs() {
        ok(
            "mustBeInRange",
            vec![Value::Num(1.0), Value::Num(1.0), Value::Num(2.0)],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
            ],
        );
        ok(
            "mustBeInRange",
            vec![
                Value::Num(1.5),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
            ],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclusive".into()),
                Value::String("inclusive".into()),
            ],
        );
        ok(
            "mustBeInRange",
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("exclude-lower".into()),
            ],
        );
        err(
            "mustBeInRange",
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::String("inclusive".into()),
                Value::String("exclusive".into()),
                Value::String("extra".into()),
            ],
        );
    }

    #[test]
    fn varname_rules_reject_keywords_underscores_and_overlong_names() {
        assert!(isvarname_value(&Value::String("alpha_1".into())));
        assert!(!isvarname_value(&Value::String("_alpha".into())));
        assert!(!isvarname_value(&Value::String("1alpha".into())));
        assert!(!isvarname_value(&Value::String("for".into())));
        assert!(!isvarname_value(&Value::String("end".into())));
        assert!(isvarname_value(&Value::String(
            "a".repeat(MATLAB_NAME_LENGTH_MAX)
        )));
        assert!(!isvarname_value(&Value::String(
            "a".repeat(MATLAB_NAME_LENGTH_MAX + 1)
        )));
    }

    #[test]
    fn callable_validators_reject_unexpected_extra_arguments() {
        err("mustBePositive", vec![Value::Num(1.0), Value::Num(2.0)]);
        err("mustBeMember", vec![Value::String("on".into())]);
        err(
            "mustBeA",
            vec![
                Value::Num(1.0),
                Value::String("double".into()),
                Value::String("extra".into()),
            ],
        );
    }

    #[test]
    fn validate_function_signatures_json_checks_json_syntax() {
        ok(
            "validateFunctionSignaturesJSON",
            vec![Value::String(r#"{"functions":[]}"#.into())],
        );
        err(
            "validateFunctionSignaturesJSON",
            vec![Value::String("{not json}".into())],
        );
    }
}
