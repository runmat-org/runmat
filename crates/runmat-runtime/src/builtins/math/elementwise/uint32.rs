//! MATLAB-compatible `uint32` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, IntValue, IntegerStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::math::elementwise::integer_cast::{
    cast_complex_value, CastError, IntegerTarget,
};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "uint32";
const UINT32_MAX_F64: f64 = u32::MAX as f64;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "uint32-converted output value.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar/array value to convert.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = uint32(X)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT32.INVALID_ARGUMENT",
    identifier: Some("RunMat:uint32:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "uint32: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT32.INVALID_INPUT",
    identifier: Some("RunMat:uint32:InvalidInput"),
    when: "Input value cannot be converted to uint32.",
    message: "uint32: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UINT32.INTERNAL",
    identifier: Some("RunMat:uint32:Internal"),
    when: "Internal conversion, gather, or provider upload failed.",
    message: "uint32: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const UINT32_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::uint32")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "uint32",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "No provider-native uint32 storage hook yet; gpuArray inputs gather to host and return a host uint32 result.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::uint32")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "uint32",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Runs outside fusion today because integer storage remains host-represented in f64 buffers.",
};

#[runtime_builtin(
    name = "uint32",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to uint32 using MATLAB saturating rounding.",
    keywords = "uint32,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::uint32::UINT32_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::uint32"
)]
async fn uint32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(error_with_detail(
            &ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    match value {
        Value::Num(n) => Ok(Value::Int(IntValue::U32(cast_scalar_to_uint32(n)))),
        Value::Int(i) => Ok(Value::Int(IntValue::U32(cast_scalar_to_uint32(
            int_to_lossless_f64(&i),
        )))),
        Value::Bool(flag) => Ok(Value::Int(IntValue::U32(if flag { 1 } else { 0 }))),
        Value::Tensor(tensor) => Ok(value_from_tensor(
            tensor_to_host(tensor).map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?,
        )),
        Value::SparseTensor(_) => Err(conversion_error("sparse")),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?;
            Ok(value_from_tensor(
                tensor_to_host(tensor).map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?,
            ))
        }
        Value::CharArray(chars) => from_char_array(chars),
        Value::GpuTensor(handle) => from_gpu(handle).await,
        value @ (Value::Complex(_, _) | Value::ComplexTensor(_)) => {
            cast_complex_value(value, IntegerTarget::U32).map_err(|cause| match cause {
                CastError::Unsupported(type_name) => conversion_error(&type_name),
                CastError::Internal(detail) => error_with_detail(&ERROR_INTERNAL, detail),
            })
        }
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Symbolic(expr) => expr
            .numeric_constant_value()
            .map(|value| Value::Int(IntValue::U32(cast_scalar_to_uint32(value))))
            .ok_or_else(|| conversion_error("sym")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }
}

fn from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<u32> = chars
        .data
        .iter()
        .map(|&ch| cast_scalar_to_uint32(ch as u32 as f64))
        .collect();
    let tensor = Tensor::new_integer(IntegerStorage::U32(data), vec![chars.rows, chars.cols])
        .map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?;
    Ok(value_from_tensor(tensor))
}

async fn from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let converted = tensor_to_host(
        gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?,
    )
    .map_err(|e| error_with_detail(&ERROR_INTERNAL, e))?;
    Ok(value_from_tensor(converted))
}

fn tensor_to_host(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.into_iter().map(cast_scalar_to_uint32).collect();
    Tensor::new_integer(IntegerStorage::U32(data), tensor.shape)
}

fn value_from_tensor(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 {
        Value::Int(IntValue::U32(cast_scalar_to_uint32(tensor.data[0])))
    } else {
        Value::Tensor(tensor)
    }
}

fn int_to_lossless_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::U64(value) if *value > UINT32_MAX_F64 as u64 => UINT32_MAX_F64,
        IntValue::U64(value) => *value as f64,
        _ => value.to_f64(),
    }
}

fn cast_scalar_to_uint32(value: f64) -> u32 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            0
        } else {
            u32::MAX
        };
    }
    value.round().clamp(0.0, UINT32_MAX_F64) as u32
}

fn conversion_error(type_name: &str) -> RuntimeError {
    error_with_detail(
        &ERROR_INVALID_INPUT,
        format!("conversion to uint32 from {type_name} is not possible"),
    )
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let message = format!("{}: {}", error.message, detail);
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerComplexStorage;

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(uint32_builtin(value, Vec::new()))
    }

    #[test]
    fn uint32_scalar_saturates_and_rounds() {
        assert_eq!(
            call(Value::Num(3.5)).expect("uint32"),
            Value::Int(IntValue::U32(4))
        );
        assert_eq!(
            call(Value::Num(-1.0)).expect("uint32"),
            Value::Int(IntValue::U32(0))
        );
        assert_eq!(
            call(Value::Num(f64::INFINITY)).expect("uint32"),
            Value::Int(IntValue::U32(u32::MAX))
        );
        assert_eq!(
            call(Value::Num(f64::NAN)).expect("uint32"),
            Value::Int(IntValue::U32(0))
        );
    }

    #[test]
    fn uint32_tensor_preserves_shape_and_class() {
        let tensor =
            Tensor::new(vec![-2.0, 2.49, 2.5, (u32::MAX as f64) + 99.0], vec![2, 2]).unwrap();
        let result = call(Value::Tensor(tensor)).expect("uint32");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 2.0, 3.0, u32::MAX as f64]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U32(vec![0, 2, 3, u32::MAX]))
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn uint32_preserves_complex_integer_input() {
        let result = call(Value::Complex(1.0, 0.0)).expect("complex integer conversion");
        let Value::ComplexTensor(tensor) = result else {
            panic!("expected typed complex integer result");
        };
        assert_eq!(
            tensor.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U32(vec![1]),
                    IntegerStorage::U32(vec![0]),
                )
                .expect("matching components")
            )
        );
    }
}
