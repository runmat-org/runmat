//! MATLAB-compatible `rectpuls` builtin for rectangular pulse samples.

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
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::type_resolvers::numeric_unary_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "rectpuls";

const RECTPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rectangular pulse samples.",
}];

const RECTPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const RECTPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times relative to the pulse center.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Pulse width.",
    },
];

const RECTPULS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T)",
        inputs: &RECTPULS_INPUTS_T,
        outputs: &RECTPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T, W)",
        inputs: &RECTPULS_INPUTS_T_W,
        outputs: &RECTPULS_OUTPUT,
    },
];

const RECTPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INVALID_INPUT",
    identifier: Some("RunMat:rectpuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "rectpuls: expected real numeric input",
};

const RECTPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.WIDTH_INVALID",
    identifier: Some("RunMat:rectpuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "rectpuls: width must be a positive finite scalar",
};

const RECTPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.ARG_COUNT",
    identifier: Some("RunMat:rectpuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "rectpuls: expected 1 or 2 arguments",
};

const RECTPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INTERNAL",
    identifier: Some("RunMat:rectpuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "rectpuls: internal error",
};

const RECTPULS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RECTPULS_ERROR_INVALID_INPUT,
    RECTPULS_ERROR_WIDTH_INVALID,
    RECTPULS_ERROR_ARG_COUNT,
    RECTPULS_ERROR_INTERNAL,
];

pub const RECTPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RECTPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RECTPULS_ERRORS,
};

const RECTPULS_INTEGER_T_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rectpuls-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rectpuls with typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RectpulsIntegerTimeExtension"),
};
const RECTPULS_INTEGER_W_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rectpuls-integer-width",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rectpuls with a typed-integer width is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RectpulsIntegerWidthExtension"),
};
const RECTPULS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rectpuls-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rectpuls with an explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RectpulsExplicitGpuInputExtension"),
};
pub const RECTPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    RECTPULS_INTEGER_T_EXTENSION,
    RECTPULS_INTEGER_W_EXTENSION,
    RECTPULS_EXPLICIT_GPU_EXTENSION,
];
const RECTPULS_INTEGER_T_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample times; typed integers are an independently gated RunMat extension.",
    }];
const RECTPULS_INTEGER_W_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "W",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public width is a positive floating scalar; typed width storage is admitted only at a checked binary64 boundary.",
    }];
pub const RECTPULS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = rectpuls(integer_T,W)",
        inputs: &RECTPULS_INTEGER_T_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Typed times reject before provider access unless RunMat extensions are enabled and exact conversion is possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = rectpuls(T,integer_W)",
        inputs: &RECTPULS_INTEGER_W_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Width is gated and checked separately from sample-time class.",
    },
];

fn rectpuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rectpuls_error_with_message(error.message, error)
}

fn rectpuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    rectpuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn rectpuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rectpuls_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(crate) fn rectpuls_scalar(t: f64, width: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let half_width = width / 2.0;
    if t >= -half_width && t < half_width {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn rectpuls_tensor(tensor: Tensor, width: f64) -> Result<Tensor, String> {
    let shape = tensor.shape.clone();
    let data = tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .map(|value| rectpuls_scalar(value, width))
        .collect::<Vec<_>>();
    Tensor::new(data, shape).map_err(|err| err.to_string())
}

pub(crate) fn validate_width(width: f64) -> Result<f64, String> {
    if !width.is_finite() || width <= 0.0 {
        Err(format!("got {width}"))
    } else {
        Ok(width)
    }
}

#[runtime_builtin(
    name = "rectpuls",
    category = "math/signal",
    summary = "Generate sampled rectangular pulses.",
    keywords = "rectpuls,rectangular pulse,pulse train,signal processing",
    type_resolver(numeric_unary_shape_type),
    descriptor(crate::builtins::math::signal::rectpuls::RECTPULS_DESCRIPTOR),
    extensions(crate::builtins::math::signal::rectpuls::RECTPULS_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::rectpuls::RECTPULS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::rectpuls"
)]
async fn rectpuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_rectpuls_extensions(&t, &rest).await?;
    let gpu_source = rectpuls_gpu_source(&t, &rest)?;
    let width = parse_width(&rest).await?;
    let tensor = rectpuls_input_tensor(t).await?;
    let output = rectpuls_tensor(tensor, width)
        .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INTERNAL, err))?;
    let Some(source) = gpu_source else {
        return Ok(tensor_into_value(output));
    };
    let restored =
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(output), BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(rectpuls_error_with_detail(
            &RECTPULS_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

async fn ensure_rectpuls_extensions(t: &Value, rest: &[Value]) -> BuiltinResult<()> {
    crate::builtins::common::validation::reject_typed_complex_integer(t, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        t,
        &RECTPULS_INTEGER_T_EXTENSION,
        BUILTIN_NAME,
        "T",
    )
    .await?;
    if let [width] = rest {
        crate::builtins::common::validation::reject_typed_complex_integer(width, BUILTIN_NAME)?;
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            width,
            &RECTPULS_INTEGER_W_EXTENSION,
            BUILTIN_NAME,
            "W",
        )
        .await?;
    }
    if std::iter::once(t)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RECTPULS_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn rectpuls_gpu_source(t: &Value, rest: &[Value]) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        std::iter::once(t)
            .chain(rest.iter())
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        BUILTIN_NAME,
    )
}

async fn parse_width(rest: &[Value]) -> BuiltinResult<f64> {
    match rest.len() {
        0 => Ok(1.0),
        1 => {
            let raw = scalar_f64_from_value_async(&rest[0])
                .await
                .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err))?
                .ok_or_else(|| rectpuls_error(&RECTPULS_ERROR_WIDTH_INVALID))?;
            validate_width(raw).map_err(|err| {
                rectpuls_error_with_detail(&RECTPULS_ERROR_WIDTH_INVALID, err.as_str())
            })
        }
        _ => Err(rectpuls_error_with_detail(
            &RECTPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        )),
    }
}

async fn rectpuls_input_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    rectpuls_error_with_source(
                        &RECTPULS_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(rectpuls_error(&RECTPULS_ERROR_INVALID_INPUT))
        }
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| rectpuls_error_with_detail(&RECTPULS_ERROR_INVALID_INPUT, err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type};
    use runmat_value::{CharArray, IntegerStorage};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(rectpuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
    }

    #[cfg(feature = "wgpu")]
    fn all_integer_storages() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![0, 1]),
            IntegerStorage::I16(vec![0, 1]),
            IntegerStorage::I32(vec![0, 1]),
            IntegerStorage::I64(vec![0, 1]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 1]),
        ]
    }

    #[test]
    fn rectpuls_type_preserves_input_shape() {
        let out = numeric_unary_shape_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn rectpuls_default_width_samples_rectangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75], vec![1, 6]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("rectpuls"));
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rectpuls_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let out =
            expect_tensor(call(Value::Tensor(input), vec![Value::Num(2.0)]).expect("rectpuls"));
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn rectpuls_custom_width_and_scalar_output() {
        let value = call(Value::Num(0.75), vec![Value::Num(2.0)]).expect("rectpuls");
        assert!(matches!(value, Value::Num(n) if n == 1.0));
    }

    #[test]
    fn rectpuls_rejects_bad_width_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(0.0)]).expect_err("width");
        assert_eq!(err.identifier(), RECTPULS_ERROR_WIDTH_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), RECTPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn rectpuls_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("input");
            let automatic = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let output = call(Value::GpuTensor(automatic), vec![Value::Num(2.0)])
                .expect("automatic resident input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected automatic resident output");
            };
            assert!(!runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![1.0, 0.0]
            );

            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit =
                explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let error = call(Value::GpuTensor(explicit), vec![Value::Num(2.0)])
                .expect_err("strict explicit input");
            assert_eq!(
                error.identifier(),
                RECTPULS_EXPLICIT_GPU_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn rectpuls_wgpu_fallback_enforces_double_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_integer_storages() {
            let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer input");
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let result = call(Value::GpuTensor(handle), vec![Value::Num(2.0)]);
            if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
                let output = result.expect("resident integer rectpuls");
                let Value::GpuTensor(output_handle) = &output else {
                    panic!("expected resident output");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
                assert_eq!(
                    test_support::gather(output)
                        .expect("gather")
                        .materialize_f64(),
                    vec![1.0, 0.0]
                );
            } else {
                let error = result.expect_err("f32 owner cannot preserve double output");
                assert!(error
                    .message()
                    .contains("cannot preserve explicit gpuArray"));
            }
        }
    }

    #[test]
    fn rectpuls_is_registered() {
        assert!(builtin_function_by_name("rectpuls").is_some());
    }
}
