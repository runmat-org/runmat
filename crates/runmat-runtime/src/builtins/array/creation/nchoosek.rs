//! MATLAB-compatible `nchoosek` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, LogicalArray, NumericDType, ResolveContext, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "nchoosek";
const MAX_OUTPUT_ELEMENTS: usize = 50_000_000;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::nchoosek")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Combination materialisation runs on the host; gpuArray inputs are gathered before constructing the output.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::nchoosek")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "nchoosek materialises combinatorial-size outputs and is not eligible for elementwise fusion.",
};

fn nchoosek_type(args: &[Type], _context: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Logical { .. }) | Some(Type::Bool) => Type::logical(),
        Some(Type::String) => Type::Unknown,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Binomial coefficient or matrix whose rows contain all selected combinations.",
}];

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n_or_v",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative integer scalar n or vector v containing choices.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of choices to select.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "b = nchoosek(n, k)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = nchoosek(v, k)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.INVALID_INPUT",
    identifier: Some("RunMat:nchoosek:InvalidInput"),
    when: "Inputs are missing, malformed, unsupported, or not valid nonnegative integer choices.",
    message: "nchoosek: invalid input",
};

const ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.TOO_LARGE",
    identifier: Some("RunMat:nchoosek:TooLarge"),
    when: "The combinations matrix would exceed RunMat's supported materialisation limit.",
    message: "nchoosek: output is too large",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NCHOOSEK.INTERNAL",
    identifier: Some("RunMat:nchoosek:Internal"),
    when: "Output allocation, GPU gather, or container construction failed.",
    message: "nchoosek: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_TOO_LARGE, ERROR_INTERNAL];

pub const NCHOOSEK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "nchoosek",
    category = "array/creation",
    summary = "Return a binomial coefficient or all k-combinations of a vector.",
    keywords = "nchoosek,combinations,binomial,combinatorics,vector",
    accel = "array_construct",
    type_resolver(nchoosek_type),
    descriptor(crate::builtins::array::creation::nchoosek::NCHOOSEK_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::nchoosek"
)]
async fn nchoosek_builtin(first: Value, k: Value) -> BuiltinResult<Value> {
    evaluate(first, k).await
}

pub async fn evaluate(first: Value, k: Value) -> BuiltinResult<Value> {
    let first = gather_if_gpu(first).await?;
    let k = gather_if_gpu(k).await?;
    evaluate_host(first, k)
}

async fn gather_if_gpu(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}"))),
        other => Ok(other),
    }
}

fn evaluate_host(first: Value, k: Value) -> BuiltinResult<Value> {
    let k = parse_k(&k)?;
    if let Some(mode) = scalar_coefficient_mode(&first) {
        let class = resolve_coefficient_class(mode.class, k.class)?;
        return coefficient_value(mode.n, k.value, class);
    }
    if is_numeric_scalar(&first) {
        return Err(nchoosek_error_with(
            &ERROR_INVALID_INPUT,
            "nchoosek: scalar n must be a nonnegative integer",
        ));
    }
    combinations_value(first, k.value)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum CoefficientClass {
    Double,
    Single,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Copy, Clone)]
struct CoefficientMode {
    n: usize,
    class: CoefficientClass,
}

#[derive(Debug, Copy, Clone)]
struct ParsedK {
    value: usize,
    class: CoefficientClass,
}

fn scalar_coefficient_mode(value: &Value) -> Option<CoefficientMode> {
    match value {
        Value::Num(n) => parse_nonnegative_integer_f64(*n).map(|n| CoefficientMode {
            n,
            class: CoefficientClass::Double,
        }),
        Value::Int(int) => parse_nonnegative_integer_int(int).map(|n| CoefficientMode {
            n,
            class: int_class(int),
        }),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return parse_nonnegative_integer_int(&value).map(|n| CoefficientMode {
                    n,
                    class: int_class(&value),
                });
            }
            parse_nonnegative_integer_f64(tensor.data[0]).map(|n| CoefficientMode {
                n,
                class: tensor_class(tensor.dtype),
            })
        }
        _ => None,
    }
}

fn resolve_coefficient_class(
    n_class: CoefficientClass,
    k_class: CoefficientClass,
) -> BuiltinResult<CoefficientClass> {
    if n_class == k_class {
        return Ok(n_class);
    }
    if n_class == CoefficientClass::Double {
        return Ok(k_class);
    }
    if k_class == CoefficientClass::Double {
        return Ok(n_class);
    }
    Err(nchoosek_error_with(
        &ERROR_INVALID_INPUT,
        "nchoosek: n and k must have the same type unless one input is double",
    ))
}

fn is_numeric_scalar(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.data.len() == 1)
}

fn coefficient_value(n: usize, k: usize, class: CoefficientClass) -> BuiltinResult<Value> {
    match class {
        CoefficientClass::Double => Ok(Value::Num(binomial_f64(n, k))),
        CoefficientClass::Single => {
            let value = binomial_f64(n, k);
            Tensor::new_with_dtype(vec![value], vec![1, 1], NumericDType::F32)
                .map(Value::Tensor)
                .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
        }
        CoefficientClass::I8 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::I8(
                checked_cast(value, i8::MAX as u128)? as i8,
            )))
        }
        CoefficientClass::I16 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::I16(
                checked_cast(value, i16::MAX as u128)? as i16,
            )))
        }
        CoefficientClass::I32 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::I32(
                checked_cast(value, i32::MAX as u128)? as i32,
            )))
        }
        CoefficientClass::I64 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::I64(
                checked_cast(value, i64::MAX as u128)? as i64,
            )))
        }
        CoefficientClass::U8 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::U8(
                checked_cast(value, u8::MAX as u128)? as u8,
            )))
        }
        CoefficientClass::U16 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::U16(
                checked_cast(value, u16::MAX as u128)? as u16,
            )))
        }
        CoefficientClass::U32 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::U32(
                checked_cast(value, u32::MAX as u128)? as u32,
            )))
        }
        CoefficientClass::U64 => {
            let value = integer_coefficient(n, k)?;
            Ok(Value::Int(IntValue::U64(
                checked_cast(value, u64::MAX as u128)? as u64,
            )))
        }
    }
}

fn integer_coefficient(n: usize, k: usize) -> BuiltinResult<u128> {
    binomial_u128(n, k)
        .ok_or_else(|| nchoosek_error_with(&ERROR_TOO_LARGE, "nchoosek: coefficient overflow"))
}

fn checked_cast(value: u128, max: u128) -> BuiltinResult<u128> {
    if value > max {
        return Err(nchoosek_error_with(
            &ERROR_TOO_LARGE,
            "nchoosek: coefficient does not fit the requested integer class",
        ));
    }
    Ok(value)
}

fn combinations_value(value: Value, k: usize) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => combinations_complex(vec![(re, im)], vec![1, 1], k),
        Value::Bool(flag) => combinations_logical(vec![if flag { 1 } else { 0 }], vec![1, 1], k),
        Value::Tensor(tensor) => combinations_tensor(tensor, k),
        Value::ComplexTensor(tensor) => combinations_complex_tensor(tensor, k),
        Value::LogicalArray(array) => combinations_logical(array.data, array.shape, k),
        Value::CharArray(chars) => combinations_chars(chars, k),
        Value::Num(_) | Value::Int(_) => Err(nchoosek_error_with(
            &ERROR_INVALID_INPUT,
            "nchoosek: scalar n must be a nonnegative integer",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::Cell(_)
        | Value::SparseTensor(_)
        | Value::Symbolic(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::GpuTensor(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(nchoosek_error(&ERROR_INVALID_INPUT)),
    }
}

fn combinations_tensor(tensor: Tensor, k: usize) -> BuiltinResult<Value> {
    let n = vector_len(&tensor.shape)?;
    let rows = checked_rows(n, k)?;
    if let Some(storage) = tensor.integer_data {
        let storage = storage
            .from_exact_values_like(combination_columns(&storage.exact_values(), rows, k)?)
            .map_err(|error| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {error}")))?;
        return Tensor::new_integer(storage, vec![rows, k])
            .map(Value::Tensor)
            .map_err(|error| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {error}")));
    }
    let data = combination_columns(&tensor.data, rows, k)?;
    Tensor::new_with_dtype(data, vec![rows, k], tensor.dtype)
        .map(Value::Tensor)
        .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn combinations_complex_tensor(tensor: ComplexTensor, k: usize) -> BuiltinResult<Value> {
    combinations_complex(tensor.data, tensor.shape, k)
}

fn combinations_complex(
    values: Vec<(f64, f64)>,
    shape: Vec<usize>,
    k: usize,
) -> BuiltinResult<Value> {
    let n = vector_len(&shape)?;
    let rows = checked_rows(n, k)?;
    let data = combination_columns(&values, rows, k)?;
    ComplexTensor::new(data, vec![rows, k])
        .map(Value::ComplexTensor)
        .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn combinations_logical(data: Vec<u8>, shape: Vec<usize>, k: usize) -> BuiltinResult<Value> {
    let n = vector_len(&shape)?;
    let rows = checked_rows(n, k)?;
    let data = combination_columns(&data, rows, k)?;
    LogicalArray::new(data, vec![rows, k])
        .map(Value::LogicalArray)
        .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn combinations_chars(chars: CharArray, k: usize) -> BuiltinResult<Value> {
    let n = vector_len(&[chars.rows, chars.cols])?;
    let rows = checked_rows(n, k)?;
    let data = combination_rows(&chars.data, rows, k)?;
    CharArray::new(data, rows, k)
        .map(Value::CharArray)
        .map_err(|e| nchoosek_error_with(&ERROR_INTERNAL, format!("{NAME}: {e}")))
}

fn checked_rows(n: usize, k: usize) -> BuiltinResult<usize> {
    let rows = if k > n { 0 } else { binomial_usize(n, k)? };
    let elements = rows
        .checked_mul(k)
        .ok_or_else(|| nchoosek_error_with(&ERROR_TOO_LARGE, "nchoosek: output size overflow"))?;
    if elements > MAX_OUTPUT_ELEMENTS {
        return Err(nchoosek_error(&ERROR_TOO_LARGE));
    }
    Ok(rows)
}

fn combination_columns<T: Clone>(values: &[T], rows: usize, k: usize) -> BuiltinResult<Vec<T>> {
    if rows == 0 || k == 0 {
        return Ok(Vec::new());
    }
    let elements = rows * k;
    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|e| nchoosek_error_with(&ERROR_TOO_LARGE, format!("{NAME}: {e}")))?;
    data.resize_with(rows * k, || values[0].clone());
    let mut row = 0;
    for_each_combination(values.len(), k, |indices| {
        for (col, &idx) in indices.iter().enumerate() {
            data[col * rows + row] = values[idx].clone();
        }
        row += 1;
    });
    Ok(data)
}

fn combination_rows<T: Clone>(values: &[T], rows: usize, k: usize) -> BuiltinResult<Vec<T>> {
    if rows == 0 || k == 0 {
        return Ok(Vec::new());
    }
    let mut data = Vec::new();
    data.try_reserve_exact(rows * k)
        .map_err(|e| nchoosek_error_with(&ERROR_TOO_LARGE, format!("{NAME}: {e}")))?;
    for_each_combination(values.len(), k, |indices| {
        for &idx in indices {
            data.push(values[idx].clone());
        }
    });
    Ok(data)
}

fn for_each_combination(n: usize, k: usize, mut visit: impl FnMut(&[usize])) {
    if k == 0 {
        visit(&[]);
        return;
    }
    if k > n {
        return;
    }
    let mut indices = (0..k).collect::<Vec<_>>();
    loop {
        visit(&indices);
        let mut pos = k;
        while pos > 0 {
            pos -= 1;
            if indices[pos] != pos + n - k {
                break;
            }
        }
        if pos == 0 && indices[pos] == n - k {
            break;
        }
        indices[pos] += 1;
        for idx in pos + 1..k {
            indices[idx] = indices[idx - 1] + 1;
        }
    }
}

fn vector_len(shape: &[usize]) -> BuiltinResult<usize> {
    match shape {
        [] => Ok(1),
        [n] => Ok(*n),
        [0, _] | [_, 0] => Ok(0),
        [1, n] | [n, 1] => Ok(*n),
        _ => Err(nchoosek_error_with(
            &ERROR_INVALID_INPUT,
            "nchoosek: first input must be a scalar or vector",
        )),
    }
}

fn parse_k(value: &Value) -> BuiltinResult<ParsedK> {
    match value {
        Value::Num(n) => parse_nonnegative_integer_f64(*n)
            .map(|value| ParsedK {
                value,
                class: CoefficientClass::Double,
            })
            .ok_or_else(invalid_k),
        Value::Int(int) => parse_nonnegative_integer_int(int)
            .map(|value| ParsedK {
                value,
                class: int_class(int),
            })
            .ok_or_else(invalid_k),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                let class = int_class(&value);
                return parse_nonnegative_integer_int(&value)
                    .map(|value| ParsedK { value, class })
                    .ok_or_else(invalid_k);
            }
            parse_nonnegative_integer_f64(tensor.data[0])
                .map(|value| ParsedK {
                    value,
                    class: tensor_class(tensor.dtype),
                })
                .ok_or_else(invalid_k)
        }
        _ => Err(invalid_k()),
    }
}

fn invalid_k() -> RuntimeError {
    nchoosek_error_with(
        &ERROR_INVALID_INPUT,
        "nchoosek: k must be a nonnegative integer scalar",
    )
}

fn parse_nonnegative_integer_f64(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value.round();
    if (value - rounded).abs() > 0.0 || rounded > usize::MAX as f64 {
        return None;
    }
    Some(rounded as usize)
}

fn parse_nonnegative_integer_int(value: &IntValue) -> Option<usize> {
    match value {
        IntValue::I8(v) => (*v >= 0).then_some(*v as usize),
        IntValue::I16(v) => (*v >= 0).then_some(*v as usize),
        IntValue::I32(v) => (*v >= 0).then_some(*v as usize),
        IntValue::I64(v) => usize::try_from(*v).ok(),
        IntValue::U8(v) => Some(*v as usize),
        IntValue::U16(v) => Some(*v as usize),
        IntValue::U32(v) => usize::try_from(*v).ok(),
        IntValue::U64(v) => usize::try_from(*v).ok(),
    }
}

fn int_class(value: &IntValue) -> CoefficientClass {
    match value {
        IntValue::I8(_) => CoefficientClass::I8,
        IntValue::I16(_) => CoefficientClass::I16,
        IntValue::I32(_) => CoefficientClass::I32,
        IntValue::I64(_) => CoefficientClass::I64,
        IntValue::U8(_) => CoefficientClass::U8,
        IntValue::U16(_) => CoefficientClass::U16,
        IntValue::U32(_) => CoefficientClass::U32,
        IntValue::U64(_) => CoefficientClass::U64,
    }
}

fn tensor_class(dtype: NumericDType) -> CoefficientClass {
    match dtype {
        NumericDType::F64 => CoefficientClass::Double,
        NumericDType::F32 => CoefficientClass::Single,
        NumericDType::I8 => CoefficientClass::I8,
        NumericDType::I16 => CoefficientClass::I16,
        NumericDType::I32 => CoefficientClass::I32,
        NumericDType::I64 => CoefficientClass::I64,
        NumericDType::U8 => CoefficientClass::U8,
        NumericDType::U16 => CoefficientClass::U16,
        NumericDType::U32 => CoefficientClass::U32,
        NumericDType::U64 => CoefficientClass::U64,
    }
}

fn binomial_usize(n: usize, k: usize) -> BuiltinResult<usize> {
    let value = binomial_u128(n, k)
        .ok_or_else(|| nchoosek_error_with(&ERROR_TOO_LARGE, "nchoosek: coefficient overflow"))?;
    usize::try_from(value).map_err(|_| {
        nchoosek_error_with(
            &ERROR_TOO_LARGE,
            "nchoosek: coefficient exceeds platform limits",
        )
    })
}

fn binomial_u128(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        result = result.checked_mul((n - k + i) as u128)?;
        result /= i as u128;
    }
    Some(result)
}

fn binomial_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 1..=k {
        result *= (n - k + i) as f64;
        result /= i as f64;
    }
    result
}

fn nchoosek_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    nchoosek_error_with(error, error.message)
}

fn nchoosek_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn call(first: Value, k: Value) -> BuiltinResult<Value> {
        block_on(evaluate(first, k))
    }

    #[test]
    fn scalar_binomial_double_and_integer_outputs() {
        assert_eq!(
            call(Value::Num(5.0), Value::Num(4.0)).unwrap(),
            Value::Num(5.0)
        );
        assert_eq!(
            call(Value::Int(IntValue::U16(5)), Value::Int(IntValue::U16(2))).unwrap(),
            Value::Int(IntValue::U16(10))
        );
        assert_eq!(
            call(Value::Num(5.0), Value::Int(IntValue::U16(2))).unwrap(),
            Value::Int(IntValue::U16(10))
        );
        let single_k = Tensor::new_with_dtype(vec![2.0], vec![1, 1], NumericDType::F32).unwrap();
        let Value::Tensor(single_out) = call(Value::Num(5.0), Value::Tensor(single_k)).unwrap()
        else {
            panic!("expected single tensor coefficient");
        };
        assert_eq!(single_out.dtype, NumericDType::F32);
        assert_eq!(single_out.shape, vec![1, 1]);
        assert_eq!(single_out.data, vec![10.0]);
        let err = call(Value::Int(IntValue::U8(5)), Value::Int(IntValue::U16(2))).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:nchoosek:InvalidInput")
        );
        assert_eq!(
            call(Value::Num(3.0), Value::Num(5.0)).unwrap(),
            Value::Num(0.0)
        );
        let Value::Num(large) = call(Value::Num(200.0), Value::Num(100.0)).unwrap() else {
            panic!("expected double coefficient");
        };
        assert!(large.is_finite());
        assert!(large > 1.0e50);

        let err = call(Value::Int(IntValue::U8(20)), Value::Num(10.0)).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:nchoosek:TooLarge"));
    }

    #[test]
    fn numeric_vector_combinations_are_column_major_and_ordered() {
        let vector = Tensor::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![1, 5]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(vector), Value::Num(4.0)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![5, 4]);
        assert_eq!(
            out.data,
            vec![
                2.0, 2.0, 2.0, 2.0, 4.0, // col 1
                4.0, 4.0, 4.0, 6.0, 6.0, // col 2
                6.0, 6.0, 8.0, 8.0, 8.0, // col 3
                8.0, 10.0, 10.0, 10.0, 10.0,
            ]
        );
    }

    #[test]
    fn k_zero_and_k_larger_than_vector_return_empty_shapes() {
        let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(zero) = call(Value::Tensor(vector.clone()), Value::Num(0.0)).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(zero.shape, vec![1, 0]);
        assert!(zero.data.is_empty());

        let Value::Tensor(too_large) = call(Value::Tensor(vector), Value::Num(4.0)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(too_large.shape, vec![0, 4]);
        assert!(too_large.data.is_empty());
    }

    #[test]
    fn preserves_uint32_vector_dtype() {
        let vector =
            Tensor::new_with_dtype(vec![10.0, 20.0, 30.0], vec![1, 3], NumericDType::U32).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(vector), Value::Int(IntValue::U16(2))).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(out.dtype, NumericDType::U32);
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }

    #[test]
    fn exact_integer_scalars_and_vector_combinations_preserve_all_classes() {
        let storages = [
            IntegerStorage::I8(vec![4, 7, 9]),
            IntegerStorage::I16(vec![4, 700, 900]),
            IntegerStorage::I32(vec![4, i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![4, i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![4, 7, u8::MAX]),
            IntegerStorage::U16(vec![4, 700, u16::MAX]),
            IntegerStorage::U32(vec![4, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![4, 9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let values = storage.exact_values();
            let scalar = Tensor::new_integer(
                storage
                    .from_exact_values_like(vec![values[0].clone()])
                    .expect("scalar storage"),
                vec![1, 1],
            )
            .expect("scalar tensor");
            let k = Tensor::new_integer(
                storage
                    .from_exact_values_like(vec![one_like(&values[0])])
                    .expect("k storage"),
                vec![1, 1],
            )
            .expect("k tensor");
            assert_eq!(
                call(Value::Tensor(scalar), Value::Tensor(k)).expect("scalar coefficient"),
                Value::Int(values[0].clone())
            );

            let expected = storage
                .from_exact_values_like(vec![
                    values[0].clone(),
                    values[0].clone(),
                    values[1].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    values[2].clone(),
                ])
                .expect("expected combinations");
            let input = Tensor::new_integer(storage.clone(), vec![1, 3]).expect("integer vector");
            let Value::Tensor(output) =
                call(Value::Tensor(input), Value::Num(2.0)).expect("combinations")
            else {
                panic!("expected exact integer combinations");
            };
            assert_eq!(output.shape, vec![3, 2]);
            assert_eq!(output.integer_storage(), Some(&expected));

            let empty = storage
                .from_exact_values_like(Vec::new())
                .expect("empty expected storage");
            for (k, shape) in [(0.0, vec![1, 0]), (4.0, vec![0, 4])] {
                let input =
                    Tensor::new_integer(storage.clone(), vec![1, 3]).expect("integer vector");
                let Value::Tensor(output) =
                    call(Value::Tensor(input), Value::Num(k)).expect("empty combinations")
                else {
                    panic!("expected exact empty combinations");
                };
                assert_eq!(output.shape, shape);
                assert_eq!(output.integer_storage(), Some(&empty));
            }
        }
    }

    fn one_like(value: &IntValue) -> IntValue {
        match value {
            IntValue::I8(_) => IntValue::I8(1),
            IntValue::I16(_) => IntValue::I16(1),
            IntValue::I32(_) => IntValue::I32(1),
            IntValue::I64(_) => IntValue::I64(1),
            IntValue::U8(_) => IntValue::U8(1),
            IntValue::U16(_) => IntValue::U16(1),
            IntValue::U32(_) => IntValue::U32(1),
            IntValue::U64(_) => IntValue::U64(1),
        }
    }

    #[test]
    fn logical_char_and_complex_vectors_are_supported() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let Value::LogicalArray(logical_out) =
            call(Value::LogicalArray(logical), Value::Num(2.0)).unwrap()
        else {
            panic!("expected logical");
        };
        assert_eq!(logical_out.shape, vec![3, 2]);
        assert_eq!(logical_out.data, vec![1, 1, 0, 0, 1, 1]);

        let chars = CharArray::new_row("abcd");
        let Value::CharArray(char_out) = call(Value::CharArray(chars), Value::Num(2.0)).unwrap()
        else {
            panic!("expected char");
        };
        assert_eq!((char_out.rows, char_out.cols), (6, 2));
        assert_eq!(
            char_out.data,
            vec!['a', 'b', 'a', 'c', 'a', 'd', 'b', 'c', 'b', 'd', 'c', 'd']
        );

        let complex =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5)], vec![3, 1]).unwrap();
        let Value::ComplexTensor(complex_out) =
            call(Value::ComplexTensor(complex), Value::Num(2.0)).unwrap()
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(complex_out.shape, vec![3, 2]);
        assert_eq!(
            complex_out.data,
            vec![
                (1.0, 1.0),
                (1.0, 1.0),
                (2.0, -1.0),
                (2.0, -1.0),
                (3.0, 0.5),
                (3.0, 0.5),
            ]
        );
    }

    #[test]
    fn rejects_bad_scalar_and_non_vector_inputs() {
        let err = call(Value::Num(2.5), Value::Num(1.0)).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:nchoosek:InvalidInput")
        );

        let scalar_tensor = Tensor::new(vec![2.5], vec![1, 1]).unwrap();
        let err = call(Value::Tensor(scalar_tensor), Value::Num(1.0)).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:nchoosek:InvalidInput")
        );

        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = call(Value::Tensor(matrix), Value::Num(2.0)).unwrap_err();
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:nchoosek:InvalidInput")
        );
    }

    #[test]
    fn gpu_inputs_are_gathered_before_combining() {
        test_support::with_test_provider(|provider| {
            let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &vector.data,
                    shape: &vector.shape,
                })
                .expect("upload");
            let Value::Tensor(out) = call(Value::GpuTensor(handle), Value::Num(2.0)).unwrap()
            else {
                panic!("expected tensor");
            };
            assert_eq!(out.shape, vec![3, 2]);
            assert_eq!(out.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        });
    }
}
