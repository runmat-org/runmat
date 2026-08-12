//! MATLAB-compatible `deg2rad` builtin for RunMat.

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
use runmat_value::{ComplexTensor, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BuiltinFusionSpec, ConstantStrategy, FusionError, FusionExprContext, FusionKernelTemplate,
    ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "deg2rad";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

const DEG2RAD_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deg2rad-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deg2rad with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Deg2radIntegerInputExtension"),
};
const DEG2RAD_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deg2rad-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deg2rad with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Deg2radLogicalInputExtension"),
};
const DEG2RAD_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DEG2RAD_INTEGER_INPUT_EXTENSION,
    DEG2RAD_LOGICAL_INPUT_EXTENSION,
];
const DEG2RAD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "D",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "R2026a documents only single and double input; RunMat mode admits all eight real integer classes.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "R = deg2rad(integer_D)",
        inputs: &DEG2RAD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage enters one explicit binary64 degrees-to-radians boundary. Resident input gathers exactly and restores the double result to its owning provider.",
    }];

const DEG2RAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise degree-to-radian conversion result.",
}];

const DEG2RAD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const DEG2RAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = deg2rad(X)",
    inputs: &DEG2RAD_INPUTS,
    outputs: &DEG2RAD_OUTPUT,
}];

const DEG2RAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.INVALID_INPUT",
    identifier: Some("RunMat:deg2rad:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "deg2rad: invalid input",
};

const DEG2RAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.INTERNAL",
    identifier: Some("RunMat:deg2rad:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "deg2rad: internal error",
};
const DEG2RAD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:deg2rad:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "deg2rad: too many output arguments",
};

const DEG2RAD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DEG2RAD_ERROR_INVALID_INPUT,
    DEG2RAD_ERROR_INTERNAL,
    DEG2RAD_ERROR_TOO_MANY_OUTPUTS,
];

pub const DEG2RAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DEG2RAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DEG2RAD_ERRORS,
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::deg2rad"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "deg2rad",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            match ctx.scalar_ty {
                ScalarType::F64 => Ok(format!("({input} * f64({DEG_TO_RAD}))")),
                ScalarType::F32 => Ok(format!("({input} * {:.10})", std::f32::consts::PI / 180.0)),
                other => Err(FusionError::UnsupportedPrecision(other)),
            }
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a multiplication by pi/180 for degree-to-radian conversion.",
};

fn deg2rad_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn deg2rad_error_with_detail(
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
    name = "deg2rad",
    category = "math/trigonometry",
    summary = "Convert angles from degrees to radians.",
    keywords = "deg2rad,degrees,radians,angle,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    extensions(DEG2RAD_EXTENSIONS),
    integer_capabilities(INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::trigonometry::deg2rad::DEG2RAD_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::deg2rad"
)]
async fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::math::trigonometry::inverse_helpers::reject_excess_outputs(BUILTIN_NAME)?;
    ensure_extensions(&value)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &value,
        BUILTIN_NAME,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "deg2rad")?;
    match value {
        Value::GpuTensor(handle) => deg2rad_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(re * DEG_TO_RAD, im * DEG_TO_RAD)),
        Value::ComplexTensor(tensor) => deg2rad_complex_tensor(tensor),
        Value::String(_) | Value::StringArray(_) => {
            Err(deg2rad_error(&DEG2RAD_ERROR_INVALID_INPUT))
        }
        other => deg2rad_real(other),
    }
}

async fn deg2rad_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        deg2rad_error_with_detail(&DEG2RAD_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    let gathered =
        crate::builtins::common::gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await?;
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
        Value::Complex(re, im) => Value::Complex(re * DEG_TO_RAD, im * DEG_TO_RAD),
        Value::ComplexTensor(tensor) => deg2rad_complex_tensor(tensor)?,
        Value::Tensor(tensor) => tensor::tensor_into_value(deg2rad_tensor(tensor)?),
        Value::Num(value) => Value::Num(value * DEG_TO_RAD),
        other => deg2rad_real(other)?,
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
            &DEG2RAD_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let is_logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if is_logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DEG2RAD_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn deg2rad_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| deg2rad_error_with_detail(&DEG2RAD_ERROR_INVALID_INPUT, e))?;
    deg2rad_tensor(tensor).map(tensor::tensor_into_value)
}

fn deg2rad_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    if tensor.numeric_dtype() == NumericDType::F32 {
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| deg2rad_error_with_detail(&DEG2RAD_ERROR_INVALID_INPUT, err))?;
        let NumericStorage::F32(values) = storage else {
            unreachable!("F32 dtype must have F32 storage")
        };
        Tensor::from_numeric_storage(
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| value * (std::f32::consts::PI / 180.0))
                    .collect(),
            ),
            shape,
        )
    } else {
        Tensor::new(
            tensor::tensor_values_f64_cow(&tensor)
                .iter()
                .map(|&value| value * DEG_TO_RAD)
                .collect(),
            shape,
        )
    }
    .map_err(|err| deg2rad_error_with_detail(&DEG2RAD_ERROR_INTERNAL, err))
}

fn deg2rad_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let dtype = tensor.numeric_dtype();
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&(re, im)| (re * DEG_TO_RAD, im * DEG_TO_RAD))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::from_f64_values_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|err| deg2rad_error_with_detail(&DEG2RAD_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, LogicalArray};

    fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::deg2rad_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn deg2rad_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = DEG2RAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = deg2rad(X)"));
        assert!(DEG2RAD_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == "RM.DEG2RAD.TOO_MANY_OUTPUTS"));
    }

    #[test]
    fn deg2rad_type_preserves_tensor_shape() {
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
    fn deg2rad_scalar() {
        let result = deg2rad_builtin(Value::Num(90.0)).expect("deg2rad");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::FRAC_PI_2).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 30.0, 45.0, 60.0, 90.0], vec![1, 5]).unwrap();
        let result = deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 5]);
                let expected = [
                    0.0,
                    std::f64::consts::PI / 6.0,
                    std::f64::consts::PI / 4.0,
                    std::f64::consts::PI / 3.0,
                    std::f64::consts::FRAC_PI_2,
                ];
                for (actual, expected) in tensor.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_reads_typed_integer_tensor_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![0, 90, 180]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
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
    fn deg2rad_int_promotes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = deg2rad_builtin(Value::Int(IntValue::I32(180))).expect("deg2rad");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn deg2rad_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = deg2rad_builtin(Value::Int(IntValue::I32(1)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            DEG2RAD_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = deg2rad_builtin(Value::Int(IntValue::U64((1u64 << 53) + 1)))
            .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(compat);
    }

    #[test]
    fn deg2rad_preserves_single_tensor_class() {
        let tensor = Tensor::from_f32(vec![90.0], vec![1, 1]).expect("single");
        match deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad") {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_logical_array_promotes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = deg2rad_builtin(Value::LogicalArray(logical)).expect("deg2rad");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64()[0], 0.0);
                assert!((tensor.materialize_f64()[1] - DEG_TO_RAD).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_complex_scales_both_parts() {
        let result = deg2rad_builtin(Value::Complex(180.0, 90.0)).expect("deg2rad");
        match result {
            Value::Complex(re, im) => {
                assert!((re - std::f64::consts::PI).abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn deg2rad_resident_complex_preserves_owner_storage_precision_and_shape() {
        test_support::with_test_provider(|provider| {
            let tensor =
                ComplexTensor::new(vec![(180.00000000000003, 90.0), (0.0, -180.0)], vec![1, 2])
                    .expect("complex double");
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            let output = deg2rad_builtin(Value::GpuTensor(input.clone())).expect("deg2rad");
            let Value::GpuTensor(handle) = &output else {
                panic!("expected resident result")
            };
            assert_eq!(handle.shape, vec![1, 2]);
            assert_eq!(handle.device_id, input.device_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert!(runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            match block_on(gpu_helpers::gather_value_async(&output)).expect("gather") {
                Value::ComplexTensor(result) => {
                    let values = result.materialize_f64();
                    assert!((values[0].0 - 180.00000000000003 * DEG_TO_RAD).abs() < 1.0e-14);
                    assert!((values[0].1 - std::f64::consts::FRAC_PI_2).abs() < 1.0e-14);
                }
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[test]
    fn deg2rad_rejects_forged_single_metadata_on_double_provider() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(vec![(180.0, 90.0)], vec![1, 1]).expect("complex");
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            runmat_accelerate_api::set_handle_precision(
                &input,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            let error = deg2rad_builtin(Value::GpuTensor(input))
                .expect_err("double provider cannot restore requested single precision");
            assert!(error.message().contains("requested result precision"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_string_errors() {
        let err = deg2rad_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), DEG2RAD_ERROR_INVALID_INPUT.identifier);
    }
}
