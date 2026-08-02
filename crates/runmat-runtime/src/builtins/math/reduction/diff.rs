//! MATLAB-compatible `diff` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, IntegerStorage, NumericStorage, ResolveContext, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::same_class_saturating_subtract;
use crate::builtins::math::reduction::type_resolvers::diff_numeric_type;
use crate::builtins::math::symbolic::{
    symbolic_expr_to_value, symbolic_variable_name_from_value, value_to_symbolic_scalar,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "diff";

fn diff_type(args: &[Type], ctx: &ResolveContext) -> Type {
    diff_numeric_type(args, ctx)
}

const DIFF_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite differences along the selected dimension.",
}];

const DIFF_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
}];

const DIFF_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Difference order (non-negative integer scalar or empty placeholder).",
    },
];

const DIFF_INPUTS_X_N_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Difference order (non-negative integer scalar or empty placeholder).",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Reduction dimension (positive integer scalar or empty placeholder).",
    },
];

const DIFF_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = diff(X)",
        inputs: &DIFF_INPUTS_X,
        outputs: &DIFF_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = diff(X, n)",
        inputs: &DIFF_INPUTS_X_N,
        outputs: &DIFF_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = diff(X, n, dim)",
        inputs: &DIFF_INPUTS_X_N_DIM,
        outputs: &DIFF_OUTPUT_B,
    },
];

const DIFF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INVALID_ARGUMENT",
    identifier: Some("RunMat:diff:InvalidArgument"),
    when: "Argument count/order/dimension/order grammar is invalid.",
    message: "diff: invalid argument",
};

const DIFF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INVALID_INPUT",
    identifier: Some("RunMat:diff:InvalidInput"),
    when: "Input value cannot be converted to a supported diff domain.",
    message: "diff: invalid input",
};

const DIFF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INTERNAL",
    identifier: Some("RunMat:diff:Internal"),
    when: "Finite-difference execution fails due to conversion, gather, allocation, or reshape operations.",
    message: "diff: internal failure",
};

const DIFF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DIFF_ERROR_INVALID_ARGUMENT,
    DIFF_ERROR_INVALID_INPUT,
    DIFF_ERROR_INTERNAL,
];

pub const DIFF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIFF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DIFF_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diff",
    op_kind: GpuOpKind::Custom("finite-difference"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("diff_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers surface finite-difference kernels through `diff_dim`; the WGPU backend keeps tensors on the device.",
};

fn diff_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn diff_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    diff_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn diff_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INVALID_ARGUMENT, detail)
}

fn diff_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INVALID_INPUT, detail)
}

fn diff_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    diff_descriptor_error_with_detail(&DIFF_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::diff")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diff",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently delegates to the runtime implementation; providers can override with custom kernels.",
};

#[runtime_builtin(
    name = "diff",
    category = "math/reduction",
    summary = "Compute forward finite differences.",
    keywords = "diff,difference,finite difference,nth difference,gpu",
    accel = "diff",
    type_resolver(diff_type),
    descriptor(crate::builtins::math::reduction::diff::DIFF_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::diff"
)]
async fn diff_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Value::Symbolic(expr) = value {
        return diff_symbolic(expr, &rest);
    }

    let (order, dim) = parse_arguments(&rest)?;
    if order == 0 {
        return Ok(value);
    }

    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(diff_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }

    match value {
        Value::Tensor(tensor) => {
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(diff_invalid_input)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor =
                tensor::value_into_tensor_for("diff", value).map_err(diff_invalid_input)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor {
                data: vec![(re, im)],
                integer_data: None,
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            };
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::ComplexTensor(tensor) => {
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::CharArray(chars) => diff_char_array(chars, order, dim),
        Value::GpuTensor(handle) => diff_gpu(handle, order, dim).await,
        other => Err(diff_invalid_input(format!(
            "diff: unsupported input type {:?}; expected numeric, logical, or character data",
            other
        ))),
    }
}

fn diff_symbolic(expr: runmat_builtins::SymbolicExpr, args: &[Value]) -> BuiltinResult<Value> {
    let (variable, order) = parse_symbolic_diff_args(&expr, args)?;
    Ok(symbolic_expr_to_value(
        runmat_builtins::SymbolicExpr::derivative_expr(expr, variable, order),
    ))
}

fn parse_symbolic_diff_args(
    expr: &runmat_builtins::SymbolicExpr,
    args: &[Value],
) -> BuiltinResult<(String, u32)> {
    match args.len() {
        0 => Ok((infer_symbolic_diff_variable(expr)?, 1)),
        1 => {
            if let Some(variable) = symbolic_variable_name_from_value(&args[0]) {
                Ok((variable, 1))
            } else {
                Ok((
                    infer_symbolic_diff_variable(expr)?,
                    parse_symbolic_order(&args[0])?,
                ))
            }
        }
        2 => {
            if let Some(variable) = symbolic_variable_name_from_value(&args[0]) {
                Ok((variable, parse_symbolic_order(&args[1])?))
            } else if let Some(variable) = symbolic_variable_name_from_value(&args[1]) {
                Ok((variable, parse_symbolic_order(&args[0])?))
            } else {
                Err(diff_invalid_argument(
                    "diff: symbolic differentiation expects a variable and optional order",
                ))
            }
        }
        _ => Err(diff_invalid_argument(
            "diff: symbolic differentiation supports at most two trailing arguments",
        )),
    }
}

fn infer_symbolic_diff_variable(expr: &runmat_builtins::SymbolicExpr) -> BuiltinResult<String> {
    let variables = expr.variables();
    if variables.len() == 1 {
        Ok(variables.into_iter().next().unwrap_or_default())
    } else if variables.is_empty() {
        Ok(String::new())
    } else {
        Err(diff_invalid_argument(
            "diff: symbolic differentiation variable is ambiguous",
        ))
    }
}

fn parse_symbolic_order(value: &Value) -> BuiltinResult<u32> {
    let expr = value_to_symbolic_scalar(value).ok_or_else(|| {
        diff_invalid_argument("diff: symbolic differentiation order must be a scalar integer")
    })?;
    let Some(order) = expr.constant_value() else {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order must be numeric",
        ));
    };
    if !order.is_finite() || order < 0.0 || (order.round() - order).abs() > 1.0e-12 {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order must be a nonnegative integer",
        ));
    }
    if order > u32::MAX as f64 {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order is too large",
        ));
    }
    Ok(order as u32)
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<(usize, Option<usize>)> {
    match args.len() {
        0 => Ok((1, None)),
        1 => {
            let order = parse_order(&args[0])?;
            Ok((order.unwrap_or(1), None))
        }
        2 => {
            let order = parse_order(&args[0])?.unwrap_or(1);
            let dim = parse_dimension_arg(&args[1])?;
            Ok((order, dim))
        }
        _ => Err(diff_invalid_argument("diff: unsupported arguments")),
    }
}

fn parse_order(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(i) => i.try_to_usize().map(Some).ok_or_else(|| {
            diff_invalid_argument("diff: order must be a non-negative integer scalar")
        }),
        Value::Num(n) => parse_numeric_order(*n).map(Some),
        Value::Tensor(t) if tensor_element_len(t) == 1 => parse_tensor_order(t).map(Some),
        Value::Bool(b) => Ok(Some(if *b { 1 } else { 0 })),
        other => Err(diff_invalid_argument(format!(
            "diff: order must be a non-negative integer scalar, got {:?}",
            other
        ))),
    }
}

fn parse_tensor_order(tensor: &Tensor) -> BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .ok_or_else(|| diff_invalid_argument("diff: integer order storage length mismatch"))?;
        return parse_integer_order(&value);
    }
    parse_numeric_order(tensor::tensor_value_f64(tensor, 0))
}

fn parse_integer_order(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| diff_invalid_argument("diff: order must be a non-negative integer scalar"))
}

fn parse_numeric_order(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(diff_invalid_argument("diff: order must be finite"));
    }
    if value < 0.0 {
        return Err(diff_invalid_argument(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(diff_invalid_argument(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    Ok(rounded as usize)
}

fn parse_dimension_arg(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(_) | Value::Num(_) => tensor::parse_dimension(value, "diff")
            .map(Some)
            .map_err(diff_invalid_argument),
        Value::Tensor(t) if tensor_element_len(t) == 1 => tensor::parse_dimension(value, "diff")
            .map(Some)
            .map_err(diff_invalid_argument),
        other => Err(diff_invalid_argument(format!(
            "diff: dimension must be a positive integer scalar, got {:?}",
            other
        ))),
    }
}

fn is_empty_array(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if tensor_element_len(t) == 0)
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

async fn diff_gpu(
    handle: GpuTensorHandle,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<Value> {
    let working_dim = dim.unwrap_or_else(|| default_dimension(&handle.shape));
    if working_dim == 0 {
        return Err(diff_invalid_argument("diff: dimension must be >= 1"));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(device_result) = provider.diff_dim(&handle, order, working_dim.saturating_sub(1))
        {
            return Ok(Value::GpuTensor(device_result));
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|e| diff_internal_error(format!("diff: {e}")))?;
    diff_tensor_host(tensor, order, Some(working_dim)).map(tensor::tensor_into_value)
}

fn diff_char_array(chars: CharArray, order: usize, dim: Option<usize>) -> BuiltinResult<Value> {
    if order == 0 {
        return Ok(Value::CharArray(chars));
    }
    let shape = vec![chars.rows, chars.cols];
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, shape).map_err(|e| diff_internal_error(format!("diff: {e}")))?;
    diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
}

pub fn diff_tensor_host(tensor: Tensor, order: usize, dim: Option<usize>) -> BuiltinResult<Tensor> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_tensor_once(current, working_dim)?;
        if tensor::tensor_element_len(&current) == 0 {
            break;
        }
        // Preserve explicit dimension if the caller provided one; update when defaulting and shape shrinks.
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_complex_tensor(
    tensor: ComplexTensor,
    order: usize,
    dim: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_complex_tensor_once(current, working_dim)?;
        if current.data.is_empty() {
            break;
        }
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_tensor_once(tensor: Tensor, dim: usize) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| diff_internal_error(format!("diff: {error}")))?;
    match storage {
        NumericStorage::F64(values) => {
            diff_floating_tensor_once(values, shape, dim, NumericStorage::F64)
        }
        NumericStorage::F32(values) => {
            diff_floating_tensor_once(values, shape, dim, NumericStorage::F32)
        }
        storage => diff_integer_tensor_once(
            shape,
            storage
                .into_integer_storage()
                .expect("integer numeric storage"),
            dim,
        ),
    }
}

fn diff_floating_tensor_once<T, F>(
    data: Vec<T>,
    mut shape: Vec<usize>,
    dim: usize,
    wrap: F,
) -> BuiltinResult<Tensor>
where
    T: Copy + std::ops::Sub<Output = T>,
    F: FnOnce(Vec<T>) -> NumericStorage,
{
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::from_numeric_storage(wrap(Vec::new()), output_shape)
            .map_err(|e| diff_internal_error(format!("diff: {e}")));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let output_len = stride_before * (len_dim - 1) * stride_after;
    let mut out = Vec::with_capacity(output_len);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                out.push(data[idx1] - data[idx0]);
            }
        }
    }

    Tensor::from_numeric_storage(wrap(out), output_shape)
        .map_err(|e| diff_internal_error(format!("diff: {e}")))
}

fn diff_integer_tensor_once(
    mut shape: Vec<usize>,
    storage: IntegerStorage,
    dim: usize,
) -> BuiltinResult<Tensor> {
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || storage.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::new_integer(storage.zeros_like(0), output_shape)
            .map_err(|e| diff_internal_error(format!("diff: {e}")));
    }

    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let mut values = Vec::with_capacity(stride_before * (len_dim - 1) * stride_after);
    let exact = storage.exact_values();
    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                values.push(same_class_saturating_subtract(
                    exact[idx1].clone(),
                    exact[idx0].clone(),
                ));
            }
        }
    }
    Tensor::new_integer(
        storage
            .from_same_class_values(values)
            .map_err(diff_internal_error)?,
        output_shape,
    )
    .map_err(|e| diff_internal_error(format!("diff: {e}")))
}

fn diff_complex_tensor_once(tensor: ComplexTensor, dim: usize) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor {
        data, mut shape, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return ComplexTensor::new(Vec::new(), output_shape)
            .map_err(|e| diff_internal_error(format!("diff: {e}")));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let mut out = Vec::with_capacity(stride_before * (len_dim - 1) * stride_after);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                let (re0, im0) = data[idx0];
                let (re1, im1) = data[idx1];
                out.push((re1 - re0, im1 - im0));
            }
        }
    }

    ComplexTensor::new(out, output_shape).map_err(|e| diff_internal_error(format!("diff: {e}")))
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dimension_length(shape: &[usize], dim: usize) -> usize {
    let dim_index = dim.saturating_sub(1);
    if dim_index < shape.len() {
        shape[dim_index]
    } else {
        1
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, val| acc.saturating_mul(val))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, SymbolicExpr, Tensor};

    fn diff_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::diff_builtin(value, rest))
    }

    #[test]
    fn diff_typed_order_parser_preserves_platform_uint64_range() {
        assert_eq!(
            parse_order(&Value::Int(IntValue::U64(3))).expect("order"),
            Some(3)
        );
        match usize::try_from(u64::MAX) {
            Ok(expected) => assert_eq!(
                parse_order(&Value::Int(IntValue::U64(u64::MAX))).expect("uint64 order"),
                Some(expected)
            ),
            Err(_) => assert!(parse_order(&Value::Int(IntValue::U64(u64::MAX))).is_err()),
        }
        assert!(parse_order(&Value::Int(IntValue::I64(-1))).is_err());
    }

    #[test]
    fn diff_typed_integer_tensor_order_and_dimension_parse_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let mut order =
            Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).expect("typed order");
        order.data.clear();
        assert_eq!(
            parse_order(&Value::Tensor(order)).expect("typed order"),
            Some(large as usize)
        );

        let mut dim =
            Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1]).expect("typed dim");
        dim.data.clear();
        assert_eq!(
            parse_dimension_arg(&Value::Tensor(dim)).expect("typed dim"),
            Some(3)
        );
    }

    #[test]
    fn diff_typed_integer_tensor_order_and_dimension_reject_negative_values() {
        let mut order =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative order");
        order.data.clear();
        assert!(parse_order(&Value::Tensor(order)).is_err());

        let mut dim =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative dim");
        dim.data.clear();
        assert!(parse_dimension_arg(&Value::Tensor(dim)).is_err());
    }

    #[test]
    fn diff_type_defaults_tensor() {
        let out = diff_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn diff_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DIFF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"B = diff(X)"));
        assert!(labels.contains(&"B = diff(X, n)"));
        assert!(labels.contains(&"B = diff(X, n, dim)"));
    }

    #[test]
    fn diff_descriptor_errors_have_stable_codes() {
        assert!(DIFF_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == DIFF_ERROR_INVALID_ARGUMENT.code));
        assert!(DIFF_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == DIFF_ERROR_INVALID_INPUT.code));
        assert!(DIFF_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == DIFF_ERROR_INTERNAL.code));
    }

    #[test]
    fn diff_symbolic_function_with_explicit_variable() {
        let y = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

        let result = diff_builtin(
            Value::Symbolic(y),
            vec![Value::Symbolic(SymbolicExpr::variable("X"))],
        )
        .expect("diff");

        assert_eq!(result.to_string(), "diff(Y(X), X)");
    }

    #[test]
    fn diff_symbolic_function_accepts_order_before_variable() {
        let y = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

        let result = diff_builtin(
            Value::Symbolic(y),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Symbolic(SymbolicExpr::variable("X")),
            ],
        )
        .expect("diff");

        assert_eq!(result.to_string(), "diff(Y(X), X, 2)");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_row_vector_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_column_vector_second_order() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_matrix_along_columns() {
        let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_handles_empty_when_order_exceeds_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape[0], 0);
                assert!(out.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_char_array_promotes_to_double() {
        let chars = CharArray::new("ACEG".chars().collect(), 1, 4).unwrap();
        let result = diff_builtin(Value::CharArray(chars), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_complex_tensor_preserves_type() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (3.0, 2.0), (6.0, 5.0)], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![(2.0, 1.0), (3.0, 3.0)]);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_zero_order_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(0))];
        let result = diff_builtin(Value::Tensor(tensor.clone()), args).expect("diff");
        assert_eq!(result, Value::Tensor(tensor));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::Tensor(output) => assert_eq!(
                output.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![3.0, 5.0])
            ),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_preserves_all_exact_integer_classes_and_saturates() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MIN, i8::MAX, i8::MAX]),
                IntegerStorage::I8(vec![i8::MAX, 0]),
            ),
            (
                IntegerStorage::I16(vec![i16::MAX, i16::MIN, i16::MIN]),
                IntegerStorage::I16(vec![i16::MIN, 0]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, i32::MAX, i32::MAX]),
                IntegerStorage::I32(vec![i32::MAX, 0]),
            ),
            (
                IntegerStorage::I64(vec![i64::MAX, i64::MIN, i64::MIN]),
                IntegerStorage::I64(vec![i64::MIN, 0]),
            ),
            (
                IntegerStorage::U8(vec![5, 0, 0]),
                IntegerStorage::U8(vec![0, 0]),
            ),
            (
                IntegerStorage::U16(vec![0, u16::MAX, u16::MAX]),
                IntegerStorage::U16(vec![u16::MAX, 0]),
            ),
            (
                IntegerStorage::U32(vec![4, 0, 0]),
                IntegerStorage::U32(vec![0, 0]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, u64::MAX - 2]),
                IntegerStorage::U64(vec![1, 0]),
            ),
        ];

        for (storage, expected) in cases {
            let input = Tensor::new_integer(storage, vec![1, expected.len() + 1]).unwrap();
            let result = diff_builtin(Value::Tensor(input), Vec::new()).expect("diff");
            match result {
                Value::Tensor(output) => assert_eq!(output.integer_storage(), Some(&expected)),
                other => panic!("expected exact integer tensor, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_exact_integer_order_dimension_and_empty_output_preserve_storage() {
        let input = Tensor::new_integer(IntegerStorage::I64(vec![1, 4, 10]), vec![3, 1]).unwrap();
        let result = diff_builtin(Value::Tensor(input), vec![Value::Int(IntValue::I32(2))])
            .expect("second order diff");
        assert_eq!(result, Value::Int(IntValue::I64(3)));

        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]),
            vec![2, 1],
        )
        .unwrap();
        let result = diff_builtin(
            Value::Tensor(input),
            vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(5))],
        )
        .expect("trailing dimension diff");
        match result {
            Value::Tensor(output) => {
                assert_eq!(output.shape, vec![2, 1, 1, 1, 0]);
                assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::U64(Vec::new()))
                );
            }
            other => panic!("expected typed empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_accepts_empty_order_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let baseline = diff_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), vec![Value::Tensor(empty)]).expect("diff");
        assert_eq!(result, baseline);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_accepts_empty_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![1, 4]).unwrap();
        let baseline = diff_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::Tensor(empty)],
        )
        .expect("diff");
        assert_eq!(result, baseline);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_negative_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-1))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_non_integer_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Num(1.5)];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("non-negative integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_rejects_invalid_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(0))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn diff_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = diff_builtin(Value::GpuTensor(handle), Vec::new()).expect("diff");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.data, vec![3.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn diff_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];

        let cpu_result = diff_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("diff");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = diff_builtin(Value::GpuTensor(handle), args).expect("diff");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, expected.shape);
        let tol = if matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        ) {
            1e-5
        } else {
            1e-12
        };
        for (a, b) in gathered.data.iter().zip(expected.data.iter()) {
            assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
        }
    }
}
