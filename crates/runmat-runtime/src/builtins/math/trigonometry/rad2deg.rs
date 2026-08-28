//! MATLAB-compatible `rad2deg` builtin for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};
use runmat_value::{NumericDType, NumericStorage};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BuiltinFusionSpec, ConstantStrategy, FusionError, FusionExprContext, FusionKernelTemplate,
    ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rad2deg";
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

const RAD2DEG_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rad2deg-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rad2deg with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Rad2degIntegerInputExtension"),
};
const RAD2DEG_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rad2deg-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rad2deg with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Rad2degLogicalInputExtension"),
};
const RAD2DEG_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RAD2DEG_INTEGER_INPUT_EXTENSION,
    RAD2DEG_LOGICAL_INPUT_EXTENSION,
];
const RAD2DEG_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "R",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents only single and double input; RunMat mode admits all eight real integer classes after an exactness check.",
    }];
pub const RAD2DEG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "D = rad2deg(integer_R)",
        inputs: &RAD2DEG_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage enters one checked binary64 radians-to-degrees boundary. Resident input gathers exactly and restores the double result to its owning provider.",
    }];

const RAD2DEG_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise radian-to-degree conversion result.",
}];

const RAD2DEG_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const RAD2DEG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = rad2deg(X)",
    inputs: &RAD2DEG_INPUTS,
    outputs: &RAD2DEG_OUTPUT,
}];

const RAD2DEG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.INVALID_INPUT",
    identifier: Some("RunMat:rad2deg:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "rad2deg: invalid input",
};

const RAD2DEG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.INTERNAL",
    identifier: Some("RunMat:rad2deg:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "rad2deg: internal error",
};
const RAD2DEG_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rad2deg:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "rad2deg: too many output arguments",
};

const RAD2DEG_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RAD2DEG_ERROR_INVALID_INPUT,
    RAD2DEG_ERROR_INTERNAL,
    RAD2DEG_ERROR_TOO_MANY_OUTPUTS,
];

pub const RAD2DEG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RAD2DEG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RAD2DEG_ERRORS,
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::rad2deg"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rad2deg",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            match ctx.scalar_ty {
                ScalarType::F64 => Ok(format!("({input} * f64({RAD_TO_DEG}))")),
                ScalarType::F32 => Ok(format!("({input} * {:.10})", 180.0 / std::f32::consts::PI)),
                other => Err(FusionError::UnsupportedPrecision(other)),
            }
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a multiplication by 180/pi for radian-to-degree conversion.",
};

fn rad2deg_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rad2deg_error_with_detail(
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

#[runtime_builtin(
    name = "rad2deg",
    category = "math/trigonometry",
    summary = "Convert angle values from radians to degrees.",
    keywords = "rad2deg,radians,degrees,angle,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::rad2deg::RAD2DEG_DESCRIPTOR),
    extensions(RAD2DEG_EXTENSIONS),
    integer_capabilities(RAD2DEG_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::rad2deg"
)]
async fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::math::trigonometry::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    ensure_extensions(&value)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &value,
        BUILTIN_NAME,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "rad2deg")?;
    match value {
        Value::GpuTensor(handle) => rad2deg_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(re * RAD_TO_DEG, im * RAD_TO_DEG)),
        Value::ComplexTensor(tensor) => rad2deg_complex_tensor(tensor),
        Value::String(_) | Value::StringArray(_) => {
            Err(rad2deg_error(&RAD2DEG_ERROR_INVALID_INPUT))
        }
        other => rad2deg_real(other),
    }
}

async fn rad2deg_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        rad2deg_error_with_detail(&RAD2DEG_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let gathered =
        crate::builtins::math::trigonometry::inverse_helpers::align_floating_value_precision(
            gathered,
            &handle,
            BUILTIN_NAME,
        )?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &gathered,
        BUILTIN_NAME,
    )?;
    let output = match gathered {
        Value::Complex(re, im) => Value::Complex(re * RAD_TO_DEG, im * RAD_TO_DEG),
        Value::ComplexTensor(tensor) => rad2deg_complex_tensor(tensor)?,
        Value::Tensor(tensor) => tensor::tensor_into_value(rad2deg_tensor(tensor)?),
        Value::Num(value) => Value::Num(value * RAD_TO_DEG),
        other => rad2deg_real(other)?,
    };
    crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
        provider,
        output,
        BUILTIN_NAME,
        &handle,
    )
}

fn ensure_extensions(value: &Value) -> BuiltinResult<()> {
    let is_integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if is_integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RAD2DEG_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RAD2DEG_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn rad2deg_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| rad2deg_error_with_detail(&RAD2DEG_ERROR_INVALID_INPUT, e))?;
    rad2deg_tensor(tensor).map(tensor::tensor_into_value)
}

fn rad2deg_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    if tensor.numeric_dtype() == NumericDType::F32 {
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| rad2deg_error_with_detail(&RAD2DEG_ERROR_INVALID_INPUT, err))?;
        let NumericStorage::F32(values) = storage else {
            unreachable!("F32 dtype must have F32 storage")
        };
        Tensor::from_numeric_storage(
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| value * (180.0 / std::f32::consts::PI))
                    .collect(),
            ),
            shape,
        )
    } else {
        Tensor::new(
            tensor::tensor_values_f64_cow(&tensor)
                .iter()
                .map(|&value| value * RAD_TO_DEG)
                .collect(),
            shape,
        )
    }
    .map_err(|err| rad2deg_error_with_detail(&RAD2DEG_ERROR_INTERNAL, err))
}

fn rad2deg_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let dtype = tensor.numeric_dtype();
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&(re, im)| (re * RAD_TO_DEG, im * RAD_TO_DEG))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::from_f64_values_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|err| rad2deg_error_with_detail(&RAD2DEG_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, LogicalArray};

    fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::rad2deg_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn rad2deg_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = RAD2DEG_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = rad2deg(X)"));
    }

    #[test]
    fn rad2deg_type_preserves_tensor_shape() {
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
    #[test]
    fn rad2deg_scalar() {
        let result = rad2deg_builtin(Value::Num(std::f64::consts::PI)).expect("rad2deg");
        match result {
            Value::Num(value) => assert!((value - 180.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_tensor_preserves_shape() {
        let tensor = Tensor::new(
            vec![
                0.0,
                std::f64::consts::PI / 6.0,
                std::f64::consts::PI / 4.0,
                std::f64::consts::PI / 3.0,
                std::f64::consts::FRAC_PI_2,
            ],
            vec![1, 5],
        )
        .unwrap();
        let result = rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 5]);
                let expected = [0.0, 30.0, 45.0, 60.0, 90.0];
                for (actual, expected) in tensor.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_reads_typed_integer_tensor_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        match rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, RAD_TO_DEG, 2.0 * RAD_TO_DEG];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_int_promotes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = rad2deg_builtin(Value::Int(IntValue::I32(1))).expect("rad2deg");
        match result {
            Value::Num(value) => assert!((value - RAD_TO_DEG).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_logical_array_promotes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = rad2deg_builtin(Value::LogicalArray(logical)).expect("rad2deg");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64()[0], 0.0);
                assert!((tensor.materialize_f64()[1] - RAD_TO_DEG).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_complex_scales_both_parts() {
        let result = rad2deg_builtin(Value::Complex(
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
        ))
        .expect("rad2deg");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 180.0).abs() < 1e-12);
                assert!((im - 90.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn rad2deg_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = rad2deg_builtin(Value::Int(IntValue::I32(1)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            RAD2DEG_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = rad2deg_builtin(Value::Int(IntValue::U64((1_u64 << 53) + 1)))
            .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(extensions);
    }

    #[test]
    fn rad2deg_preserves_single_tensor_class() {
        let tensor = Tensor::from_f32(vec![std::f32::consts::PI], vec![1, 1]).expect("single");
        match rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg") {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single tensor, got {other:?}"),
        }
    }

    #[test]
    fn rad2deg_resident_complex_preserves_owner_storage_precision_and_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(
                vec![(std::f64::consts::PI, std::f64::consts::FRAC_PI_2)],
                vec![1, 1],
            )
            .expect("complex double");
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            let output = rad2deg_builtin(Value::GpuTensor(input.clone())).expect("rad2deg");
            let Value::GpuTensor(handle) = &output else {
                panic!("expected resident result")
            };
            assert_eq!(handle.shape, vec![1, 1]);
            assert_eq!(handle.device_id, input.device_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert!(runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            match block_on(gpu_helpers::gather_value_async(&output)).expect("gather") {
                Value::ComplexTensor(result) => {
                    let values = result.materialize_f64();
                    assert!((values[0].0 - 180.0).abs() < 1.0e-12);
                    assert!((values[0].1 - 90.0).abs() < 1.0e-12);
                }
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_string_errors() {
        let err = rad2deg_builtin(Value::String("pi".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), RAD2DEG_ERROR_INVALID_INPUT.identifier);
    }
}
