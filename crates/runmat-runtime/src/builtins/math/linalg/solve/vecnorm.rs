//! MATLAB-compatible `vecnorm` builtin.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::solve::norm::{root_sum_of_squares, NormOrder};
use crate::builtins::math::linalg::type_resolvers::vecnorm_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "vecnorm";

const VECNORM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Vector-wise norm values.",
}];

const VECNORM_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric array.",
}];

const VECNORM_INPUTS_A_P: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input numeric array.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("2"),
        description: "Positive norm order or Inf.",
    },
];

const VECNORM_INPUTS_A_P_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input numeric array.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("2"),
        description: "Positive norm order or Inf.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dimension to operate along.",
    },
];

const VECNORM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A)",
        inputs: &VECNORM_INPUTS_A,
        outputs: &VECNORM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A, p)",
        inputs: &VECNORM_INPUTS_A_P,
        outputs: &VECNORM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "N = vecnorm(A, p, dim)",
        inputs: &VECNORM_INPUTS_A_P_DIM,
        outputs: &VECNORM_OUTPUT,
    },
];

const VECNORM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INVALID_ARGUMENT",
    identifier: Some("RunMat:vecnorm:InvalidArgument"),
    when: "The norm order or dimension argument is malformed or unsupported.",
    message: "vecnorm: invalid argument",
};

const VECNORM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INVALID_INPUT",
    identifier: Some("RunMat:vecnorm:InvalidInput"),
    when: "Input values cannot be converted to a supported numeric array domain.",
    message: "vecnorm: invalid input",
};

const VECNORM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VECNORM.INTERNAL",
    identifier: Some("RunMat:vecnorm:Internal"),
    when: "Runtime fails while reducing, allocating, gathering, or uploading values.",
    message: "vecnorm: internal runtime failure",
};

const VECNORM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    VECNORM_ERROR_INVALID_ARGUMENT,
    VECNORM_ERROR_INVALID_INPUT,
    VECNORM_ERROR_INTERNAL,
];

pub const VECNORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VECNORM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VECNORM_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::vecnorm")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(1024),
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat gathers GPU tensors, computes vector-wise norms on the host, and uploads the result when a provider is active.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::solve::vecnorm"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Vector-wise norm is a shape-changing reduction and currently executes through the runtime path.",
};

fn error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn argument_error(message: impl Into<String>) -> RuntimeError {
    error_with_message(message, &VECNORM_ERROR_INVALID_ARGUMENT)
}

fn input_error(message: impl Into<String>) -> RuntimeError {
    error_with_message(message, &VECNORM_ERROR_INVALID_INPUT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    error_with_message(message, &VECNORM_ERROR_INTERNAL)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runtime_builtin(
    name = "vecnorm",
    category = "math/linalg/solve",
    summary = "Compute vector-wise array norms.",
    keywords = "vecnorm,vector norm,array norm,euclidean,gpu",
    accel = "reduction",
    type_resolver(vecnorm_type),
    descriptor(crate::builtins::math::linalg::solve::vecnorm::VECNORM_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::solve::vecnorm"
)]
async fn vecnorm_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = VecnormArgs::parse(&rest)?;
    match value {
        Value::GpuTensor(handle) => vecnorm_gpu(handle, args).await,
        Value::ComplexTensor(tensor) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(
                &tensor, NAME,
            )?;
            let result = vecnorm_complex_tensor(&tensor, args)?;
            Ok(tensor::tensor_into_value(result))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(input_error)?;
            let result = vecnorm_complex_tensor(&tensor, args)?;
            Ok(tensor::tensor_into_value(result))
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(input_error)?;
            let result = vecnorm_real_tensor(&tensor, args)?;
            Ok(tensor::tensor_into_value(result))
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VecnormArgs {
    order: NormOrder,
    dim: Option<usize>,
}

impl VecnormArgs {
    fn parse(args: &[Value]) -> BuiltinResult<Self> {
        match args.len() {
            0 => Ok(Self {
                order: NormOrder::Two,
                dim: None,
            }),
            1 => Ok(Self {
                order: parse_order(&args[0])?,
                dim: None,
            }),
            2 => Ok(Self {
                order: parse_order(&args[0])?,
                dim: Some(parse_dim(&args[1])?),
            }),
            _ => Err(argument_error(format!(
                "{NAME}: expected A, A,p, or A,p,dim."
            ))),
        }
    }
}

async fn vecnorm_gpu(handle: GpuTensorHandle, args: VecnormArgs) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider();
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let result = vecnorm_real_tensor(&tensor, args)?;

    if let Some(provider) = provider {
        let data = tensor::tensor_values_f64_cow(&result);
        let view = HostTensorView {
            data: data.as_ref(),
            shape: &result.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => {
                runmat_accelerate_api::mark_residency(&handle);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) => {
                let message = err.to_string();
                if message == "interaction pending..." {
                    return Err(build_runtime_error("interaction pending...")
                        .with_builtin(NAME)
                        .build());
                }
            }
        }
    }

    Ok(tensor::tensor_into_value(result))
}

fn vecnorm_real_tensor(tensor: &Tensor, args: VecnormArgs) -> BuiltinResult<Tensor> {
    let dim = resolve_dim(&tensor.shape, args.dim);
    let dtype = if tensor.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let result = reduce_magnitudes(
        &tensor.shape,
        dim,
        args.order,
        // Typed integer tensors retain an f64 compatibility mirror that can
        // be stale (and cannot represent every i64/u64 exactly).  vecnorm is
        // a floating-point algorithm, so materialize from the authoritative
        // storage before taking magnitudes.
        |index| tensor::tensor_value_f64(tensor, index).abs(),
        dtype,
    )?;
    Ok(result)
}

fn vecnorm_complex_tensor(tensor: &ComplexTensor, args: VecnormArgs) -> BuiltinResult<Tensor> {
    let dim = resolve_dim(&tensor.shape, args.dim);
    reduce_magnitudes(
        &tensor.shape,
        dim,
        args.order,
        |index| {
            let (re, im) = tensor.materialize_f64()[index];
            re.hypot(im)
        },
        NumericDType::F64,
    )
}

fn reduce_magnitudes<F>(
    shape: &[usize],
    dim: usize,
    order: NormOrder,
    mut magnitude_at: F,
    dtype: NumericDType,
) -> BuiltinResult<Tensor>
where
    F: FnMut(usize) -> f64,
{
    let len: usize = shape.iter().product();
    if len == 0 {
        let out_shape = output_shape(shape, dim);
        let out_len: usize = out_shape.iter().product();
        return Tensor::new_with_dtype(vec![0.0; out_len], out_shape, dtype)
            .map_err(|err| internal_error(format!("{NAME}: {err}")));
    }

    let rank = shape.len();
    if rank == 0 || dim >= rank || shape[dim] == 1 {
        let data = (0..len)
            .map(|index| cast_output(magnitude_at(index), dtype))
            .collect();
        return Tensor::new_with_dtype(data, shape.to_vec(), dtype)
            .map_err(|err| internal_error(format!("{NAME}: {err}")));
    }

    let strides = strides_for(shape);
    let out_shape = output_shape(shape, dim);
    let out_len: usize = out_shape.iter().product();
    let dim_len = shape[dim];
    let dim_stride = strides[dim];
    let mut data = Vec::with_capacity(out_len);
    let mut coordinates = vec![0usize; rank];

    for out_linear in 0..out_len {
        unravel_index(out_linear, &out_shape, &mut coordinates);
        let base = coordinates
            .iter()
            .zip(strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum::<usize>();
        let mut magnitudes = Vec::with_capacity(dim_len);
        for offset in 0..dim_len {
            magnitudes.push(magnitude_at(base + offset * dim_stride));
        }
        let value = vector_norm(&magnitudes, order)?;
        data.push(cast_output(value, dtype));
    }

    Tensor::new_with_dtype(data, out_shape, dtype)
        .map_err(|err| internal_error(format!("{NAME}: {err}")))
}

fn vector_norm(magnitudes: &[f64], order: NormOrder) -> BuiltinResult<f64> {
    if magnitudes.iter().any(|value| value.is_nan()) {
        return Ok(f64::NAN);
    }
    match order {
        NormOrder::Default => unreachable!("vecnorm resolves default order while parsing"),
        NormOrder::Two | NormOrder::Fro => Ok(root_sum_of_squares(magnitudes)),
        NormOrder::One => Ok(magnitudes.iter().sum()),
        NormOrder::Inf => Ok(magnitudes
            .iter()
            .fold(0.0, |acc, &value| if value > acc { value } else { acc })),
        NormOrder::P(p) => Ok(scaled_p_norm(magnitudes, p)),
        NormOrder::NegInf | NormOrder::Zero | NormOrder::Nuc => Err(argument_error(format!(
            "{NAME}: p must be a positive scalar or Inf."
        ))),
    }
}

fn scaled_p_norm(magnitudes: &[f64], p: f64) -> f64 {
    let mut scale = 0.0_f64;
    for &value in magnitudes {
        if value.is_infinite() {
            return f64::INFINITY;
        }
        if value > scale {
            scale = value;
        }
    }
    if scale == 0.0 {
        return 0.0;
    }
    let sum: f64 = magnitudes
        .iter()
        .map(|&value| (value / scale).powf(p))
        .sum();
    scale * sum.powf(1.0 / p)
}

fn output_shape(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    if dim < out.len() {
        out[dim] = 1;
    }
    out
}

fn resolve_dim(shape: &[usize], explicit: Option<usize>) -> usize {
    if let Some(dim) = explicit {
        return dim - 1;
    }
    if shape.is_empty() {
        return 0;
    }
    shape.iter().position(|&size| size != 1).unwrap_or(0)
}

fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn unravel_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for (coord, &dim) in out.iter_mut().zip(shape.iter()) {
        if dim == 0 {
            *coord = 0;
        } else {
            *coord = linear % dim;
            linear /= dim;
        }
    }
}

fn parse_order(value: &Value) -> BuiltinResult<NormOrder> {
    match value {
        Value::Num(value) => parse_numeric_order(*value),
        Value::Int(value) => parse_integer_order(value),
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                if let Some(integer) = tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                {
                    parse_integer_order(&integer)
                } else {
                    parse_numeric_order(scalar_tensor_f64(tensor))
                }
            } else {
                Err(argument_error(format!(
                    "{NAME}: p must be a positive scalar or Inf."
                )))
            }
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(argument_error(format!(
            "{NAME}: p must be a positive numeric scalar or Inf."
        ))),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(argument_error(format!("{NAME}: p must be real-valued.")))
        }
        Value::GpuTensor(_) => Err(argument_error(format!(
            "{NAME}: p cannot be a GPU-resident tensor."
        ))),
        other => {
            let _ = other;
            Err(argument_error(format!(
                "{NAME}: p must be a positive numeric scalar or Inf."
            )))
        }
    }
}

fn parse_numeric_order(raw: f64) -> BuiltinResult<NormOrder> {
    if raw.is_nan() || raw <= 0.0 {
        return Err(argument_error(format!(
            "{NAME}: p must be a positive numeric scalar or Inf."
        )));
    }
    if raw.is_infinite() {
        if raw.is_sign_positive() {
            return Ok(NormOrder::Inf);
        }
        return Err(argument_error(format!(
            "{NAME}: p must be a positive numeric scalar or Inf."
        )));
    }
    if approx_eq(raw, 1.0) {
        return Ok(NormOrder::One);
    }
    if approx_eq(raw, 2.0) {
        return Ok(NormOrder::Two);
    }
    Ok(NormOrder::P(raw))
}

fn parse_integer_order(value: &IntValue) -> BuiltinResult<NormOrder> {
    let raw = exact_integer_as_f64(value).ok_or_else(|| {
        argument_error(format!(
            "{NAME}: p integer is outside the exact double range."
        ))
    })?;
    parse_numeric_order(raw)
}

fn exact_integer_as_f64(value: &IntValue) -> Option<f64> {
    const MAX_EXACT_INTEGER: u64 = 1 << 53;
    match value {
        IntValue::I8(v) => Some(*v as f64),
        IntValue::I16(v) => Some(*v as f64),
        IntValue::I32(v) => Some(*v as f64),
        IntValue::I64(v) if v.unsigned_abs() <= MAX_EXACT_INTEGER => Some(*v as f64),
        IntValue::U8(v) => Some(*v as f64),
        IntValue::U16(v) => Some(*v as f64),
        IntValue::U32(v) => Some(*v as f64),
        IntValue::U64(v) if *v <= MAX_EXACT_INTEGER => Some(*v as f64),
        _ => None,
    }
}

fn parse_dim(value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|dim| *dim >= 1)
            .ok_or_else(|| {
                argument_error(format!(
                    "{NAME}: dim must be a positive integer numeric scalar."
                ))
            });
    }
    let raw = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            return Err(argument_error(format!(
                "{NAME}: dim must be a positive integer numeric scalar."
            )));
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(argument_error(format!("{NAME}: dim must be real-valued.")));
        }
        Value::GpuTensor(_) => {
            return Err(argument_error(format!(
                "{NAME}: dim cannot be a GPU-resident tensor."
            )));
        }
        other => {
            return Err(argument_error(format!(
                "{NAME}: dim must be a positive integer numeric scalar, got {other:?}."
            )));
        }
    };

    if !raw.is_finite() || raw < 1.0 {
        return Err(argument_error(format!(
            "{NAME}: dim must be a positive integer numeric scalar."
        )));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(argument_error(format!(
            "{NAME}: dim must be a positive integer numeric scalar."
        )));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(argument_error(format!(
            "{NAME}: dim must be a positive integer numeric scalar."
        )));
    }
    Ok(rounded as usize)
}

fn scalar_tensor_f64(tensor: &Tensor) -> f64 {
    if let Some(integer) = tensor
        .integer_storage()
        .and_then(|storage| storage.value_at(0))
    {
        return integer.to_f64();
    }
    tensor::tensor_value_f64(tensor, 0)
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= f64::EPSILON * (a.abs() + b.abs() + 1.0)
}

fn cast_output(value: f64, dtype: NumericDType) -> f64 {
    if dtype == NumericDType::F32 {
        (value as f32) as f64
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage, ResolveContext, Type};

    fn assert_close(actual: f64, expected: f64) {
        if actual.is_nan() && expected.is_nan() {
            return;
        }
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-10,
            "expected {expected}, got {actual} (diff {diff})"
        );
    }

    fn call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::vecnorm_builtin(value, rest))
    }

    #[test]
    fn vecnorm_type_reduces_default_dimension() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = vecnorm_type(&[ty], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn vecnorm_type_uses_unknown_shape_for_nonliteral_explicit_dim() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        };
        let out = vecnorm_type(
            &[ty, Type::Num, Type::Num],
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
    fn vecnorm_descriptor_covers_core_forms() {
        let labels: Vec<&str> = VECNORM_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"N = vecnorm(A)"));
        assert!(labels.contains(&"N = vecnorm(A, p)"));
        assert!(labels.contains(&"N = vecnorm(A, p, dim)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_matrix_defaults_to_columns() {
        let tensor = Tensor::new(
            vec![2.0, -1.0, -3.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_close(out.materialize_f64()[0], (4.0f64 + 1.0 + 9.0).sqrt());
                assert_close(out.materialize_f64()[1], (0.0f64 + 1.0 + 9.0).sqrt());
                assert_close(out.materialize_f64()[2], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_rows_with_explicit_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = call(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_close(out.materialize_f64()[0], 4.0);
                assert_close(out.materialize_f64()[1], 6.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_accepts_fractional_positive_p() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let result = call(Value::Tensor(tensor), vec![Value::Num(0.5)]).expect("vecnorm");
        match result {
            Value::Num(value) => {
                let expected = (1.0f64.sqrt() + 4.0f64.sqrt() + 9.0f64.sqrt()).powf(2.0);
                assert_close(value, expected);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_inf_uses_max_magnitude() {
        let tensor = Tensor::new(vec![2.0, -7.0, 4.0], vec![3, 1]).unwrap();
        let result = call(Value::Tensor(tensor), vec![Value::Num(f64::INFINITY)]).expect("vecnorm");
        match result {
            Value::Num(value) => assert_close(value, 7.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_two_norm_multiple_infinities_returns_inf() {
        let tensor = Tensor::new(vec![f64::INFINITY, f64::INFINITY], vec![2, 1]).unwrap();
        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_large_p_norm_uses_scaled_accumulation() {
        let tensor = Tensor::new(vec![1.0e200, 1.0e200], vec![2, 1]).unwrap();
        let result = call(Value::Tensor(tensor), vec![Value::Num(3.0)]).expect("vecnorm");
        match result {
            Value::Num(value) => {
                let expected = 1.0e200 * 2.0f64.powf(1.0 / 3.0);
                let rel = ((value - expected) / expected).abs();
                assert!(rel < 1e-12, "expected {expected}, got {value}");
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_tiny_p_norm_uses_scaled_accumulation() {
        let tensor = Tensor::new(vec![1.0e-200, 1.0e-200], vec![2, 1]).unwrap();
        let result = call(Value::Tensor(tensor), vec![Value::Num(3.0)]).expect("vecnorm");
        match result {
            Value::Num(value) => {
                let expected = 1.0e-200 * 2.0f64.powf(1.0 / 3.0);
                let rel = ((value - expected) / expected).abs();
                assert!(rel < 1e-12, "expected {expected}, got {value}");
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_complex_uses_magnitudes() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 4.0), (5.0, 12.0), (8.0, 15.0)], vec![3, 1]).unwrap();
        let result = call(Value::ComplexTensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Num(value) => assert_close(value, (25.0f64 + 169.0 + 289.0).sqrt()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn vecnorm_rejects_typed_complex_integer_inputs() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let err = call(Value::ComplexTensor(tensor), Vec::new())
            .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    #[test]
    fn vecnorm_reads_typed_integer_storage_not_f64_mirror() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, 0]), vec![2, 1])
            .expect("typed integer tensor");

        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Num(value) => assert_eq!(value, (i64::MIN as f64).abs()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_dim_beyond_rank_returns_abs_with_original_shape() {
        let tensor = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();
        let result = call(
            Value::Tensor(tensor),
            vec![Value::Num(2.0), Value::Num(3.0)],
        )
        .expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_nan_propagates_within_each_vector() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert!(out.materialize_f64()[0].is_nan());
                assert_close(out.materialize_f64()[1], 5.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_empty_dimension_returns_zero_norms() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_rejects_nonpositive_p_and_bad_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = call(Value::Tensor(tensor.clone()), vec![Value::Num(0.0)]).unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive numeric scalar"));

        let err = call(
            Value::Tensor(tensor),
            vec![Value::Num(2.0), Value::Num(1.5)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_rejects_nonnumeric_p_and_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = call(Value::Tensor(tensor.clone()), vec![Value::from("Inf")]).unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive numeric scalar"));

        let err = call(
            Value::Tensor(tensor),
            vec![Value::Num(2.0), Value::Bool(true)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive integer numeric scalar"));
    }

    #[test]
    fn vecnorm_order_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let order =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("typed order");

        let result = call(Value::Tensor(tensor), vec![Value::Tensor(order)]).expect("vecnorm");
        match result {
            Value::Num(value) => assert_close(value, 7.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn vecnorm_order_uses_all_integer_storage_classes_without_mirror() {
        let storages = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let order = Tensor::new_integer(storage, vec![1, 1]).expect("order");
            assert!(matches!(
                parse_order(&Value::Tensor(order)),
                Ok(NormOrder::One)
            ));
        }
    }

    #[test]
    fn vecnorm_order_rejects_wide_integer_storage_instead_of_rounding() {
        let wide = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("wide order");
        assert!(parse_order(&Value::Tensor(wide)).is_err());
    }

    #[test]
    fn vecnorm_dim_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("typed dim");

        let result = call(
            Value::Tensor(tensor),
            vec![Value::Num(2.0), Value::Tensor(dim)],
        )
        .expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_close(out.materialize_f64()[0], 10.0f64.sqrt());
                assert_close(out.materialize_f64()[1], 20.0f64.sqrt());
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let wide = 9_007_199_254_740_993_u64;
        let large_dim =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("large dim");
        match usize::try_from(wide) {
            Ok(expected) => assert_eq!(parse_dim(&Value::Tensor(large_dim)).unwrap(), expected),
            Err(_) => assert!(parse_dim(&Value::Tensor(large_dim)).is_err()),
        }
    }

    #[test]
    fn vecnorm_rejects_negative_typed_integer_tensor_order_and_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let order =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("typed order");
        let err = call(Value::Tensor(tensor.clone()), vec![Value::Tensor(order)]).unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive numeric scalar"));

        let dim =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("typed dim");
        let err = call(
            Value::Tensor(tensor),
            vec![Value::Num(2.0), Value::Tensor(dim)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), VECNORM_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("positive integer numeric scalar"));
    }

    #[test]
    fn vecnorm_dim_rejects_unrepresentable_double_boundary_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert!(parse_dim(&Value::Num(boundary)).is_err());
        assert!(parse_dim(&Value::Num(1.5)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_preserves_single_for_array_outputs() {
        let tensor =
            Tensor::new_with_dtype(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2], NumericDType::F32)
                .unwrap();
        let result = call(Value::Tensor(tensor), Vec::new()).expect("vecnorm");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.numeric_dtype(), NumericDType::F32);
                assert_eq!(out.materialize_f64(), vec![5.0, 13.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vecnorm_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(Value::GpuTensor(handle), Vec::new()).expect("vecnorm");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_close(gathered.materialize_f64()[0], 5.0);
            assert_close(gathered.materialize_f64()[1], 13.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn vecnorm_wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            tracing::warn!("skipping vecnorm_wgpu_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2]).unwrap();
        let cpu = vecnorm_real_tensor(
            &tensor,
            VecnormArgs {
                order: NormOrder::Two,
                dim: None,
            },
        )
        .expect("cpu vecnorm");

        let Some(provider) = runmat_accelerate_api::provider() else {
            tracing::warn!("skipping vecnorm_wgpu_matches_cpu: provider not registered");
            return;
        };
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = call(Value::GpuTensor(handle), Vec::new()).expect("vecnorm");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert_close(*actual, *expected);
        }
    }
}
