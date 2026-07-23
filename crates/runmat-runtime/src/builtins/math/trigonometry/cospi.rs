//! MATLAB-compatible `cospi` builtin for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::pi_helpers::{cospi_complex, cospi_real};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cospi";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cospi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("trig_pi"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat gathers gpuArray inputs and evaluates cospi on the host to preserve exact integer and half-integer results.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cospi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion is disabled because lowering to cos(x*pi) would lose cospi's exactness guarantees.",
};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise cos(X*pi) result with exact integer and half-integer handling.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = cospi(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSPI.INVALID_INPUT",
    identifier: Some("RunMat:cospi:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "cospi: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSPI.INTERNAL",
    identifier: Some("RunMat:cospi:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "cospi: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const COSPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "cospi",
    category = "math/trigonometry",
    summary = "Compute cos(X*pi) accurately.",
    keywords = "cospi,cosine,pi,trigonometry,elementwise,gpu",
    sink = true,
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::cospi::COSPI_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::cospi"
)]
async fn cospi_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cospi")?;
    match value {
        Value::GpuTensor(handle) => cospi_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = cospi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => cospi_complex_tensor(tensor),
        Value::String(_) | Value::StringArray(_) => Err(cospi_error(&ERROR_INVALID_INPUT)),
        other => cospi_real_value(other),
    }
}

async fn cospi_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    match gathered {
        Value::Complex(re, im) => {
            let (out_re, out_im) = cospi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => cospi_complex_tensor(tensor),
        other => cospi_real_value(other),
    }
}

fn cospi_real_value(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| cospi_error_with_detail(&ERROR_INVALID_INPUT, err))?;
    cospi_tensor(tensor).map(tensor::tensor_into_value)
}

fn cospi_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&value| cospi_real(value)).collect();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| cospi_error_with_detail(&ERROR_INTERNAL, err))
}

fn cospi_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| cospi_complex(re, im))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| cospi_error_with_detail(&ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

fn cospi_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cospi_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    use crate::builtins::common::test_support;

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(super::cospi_builtin(value))
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(value) => value,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn descriptor_covers_core_form() {
        assert_eq!(COSPI_DESCRIPTOR.signatures[0].label, "Y = cospi(X)");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn type_resolver_preserves_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn scalar_exact_values() {
        assert_eq!(expect_num(call(Value::Num(0.0)).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Num(0.5)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(1.0)).unwrap()), -1.0);
        assert_eq!(expect_num(call(Value::Num(1.5)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(-0.5)).unwrap()), 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn tensor_preserves_shape_and_exact_values() {
        let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.data, vec![1.0, 0.0, -1.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn integer_and_logical_inputs_promote() {
        assert_eq!(expect_num(call(Value::Int(IntValue::I32(2))).unwrap()), 1.0);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.data, vec![1.0, -1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn complex_inputs_use_analytic_extension() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, 1.0)).unwrap() else {
            panic!("expected complex");
        };
        assert_eq!(re, 0.0);
        assert!((im + std::f64::consts::PI.sinh()).abs() < 1e-12);
    }

    #[test]
    fn complex_exact_zero_component_survives_overflowing_imaginary_scale() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, f64::INFINITY)).unwrap() else {
            panic!("expected complex");
        };
        assert_eq!(re, 0.0);
        assert!(im.is_infinite() && im.is_sign_negative());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gpu_input_is_gathered() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .expect("upload");
            let Value::Tensor(out) = call(Value::GpuTensor(handle)).unwrap() else {
                panic!("expected tensor");
            };
            assert_eq!(out.data, vec![1.0, 0.0, -1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn strings_are_rejected() {
        let err = call(Value::String("0.5".to_string())).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:cospi:InvalidInput"));
    }
}
