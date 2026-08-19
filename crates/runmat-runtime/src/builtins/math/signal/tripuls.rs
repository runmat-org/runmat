//! MATLAB-compatible `tripuls` builtin for triangular pulse samples.

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
use runmat_value::NumericStorage;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::type_resolvers::numeric_unary_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tripuls";

const TRIPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Triangular pulse samples.",
}];

const TRIPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const TRIPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
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

const TRIPULS_INPUTS_T_W_S: [BuiltinParamDescriptor; 3] = [
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
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Skew in [-1, 1].",
    },
];

const TRIPULS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T)",
        inputs: &TRIPULS_INPUTS_T,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W)",
        inputs: &TRIPULS_INPUTS_T_W,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W, S)",
        inputs: &TRIPULS_INPUTS_T_W_S,
        outputs: &TRIPULS_OUTPUT,
    },
];

const TRIPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INVALID_INPUT",
    identifier: Some("RunMat:tripuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "tripuls: expected real numeric input",
};

const TRIPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.WIDTH_INVALID",
    identifier: Some("RunMat:tripuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "tripuls: width must be a positive finite scalar",
};

const TRIPULS_ERROR_SKEW_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.SKEW_INVALID",
    identifier: Some("RunMat:tripuls:SkewInvalid"),
    when: "Skew is not a finite scalar in [-1, 1].",
    message: "tripuls: skew must be a finite scalar in [-1, 1]",
};

const TRIPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.ARG_COUNT",
    identifier: Some("RunMat:tripuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "tripuls: expected 1, 2, or 3 arguments",
};

const TRIPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INTERNAL",
    identifier: Some("RunMat:tripuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "tripuls: internal error",
};

const TRIPULS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRIPULS_ERROR_INVALID_INPUT,
    TRIPULS_ERROR_WIDTH_INVALID,
    TRIPULS_ERROR_SKEW_INVALID,
    TRIPULS_ERROR_ARG_COUNT,
    TRIPULS_ERROR_INTERNAL,
];

pub const TRIPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRIPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRIPULS_ERRORS,
};

const TRIPULS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tripuls-integer-sample-times",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tripuls with native typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TripulsIntegerSampleTimesExtension"),
};
const TRIPULS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tripuls-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tripuls with native typed-integer width or skew is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TripulsIntegerControlsExtension"),
};
const TRIPULS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tripuls-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive tripuls gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TripulsGpuInputExtension"),
};
pub const TRIPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TRIPULS_INTEGER_DATA_EXTENSION,
    TRIPULS_INTEGER_CONTROL_EXTENSION,
    TRIPULS_GPU_INPUT_EXTENSION,
];
const TRIPULS_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sample-time domain is single or double. RunMat mode admits typed integers only when every value is exactly representable at the binary64 pulse boundary.",
    }];
const TRIPULS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "W",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed width is a mode-gated scalar and must convert exactly before positive-range validation.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed skew is a mode-gated scalar and must convert exactly before validation in the closed interval from -1 through 1.",
    },
];
pub const TRIPULS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tripuls(integer_T, ...)",
        inputs: &TRIPULS_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Compatibility and exactness checks occur before provider access; admitted integer samples produce a same-shaped double waveform.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = tripuls(T, integer_W, integer_S?)",
        inputs: &TRIPULS_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Width and skew cross checked scalar binary64 boundaries; output shape, class, and residency follow the supported sample-time path.",
    },
];

fn tripuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tripuls_error_with_message(error.message, error)
}

fn tripuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tripuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn tripuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tripuls_error_with_source(
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

pub(crate) fn tripuls_scalar(t: f64, width: f64, skew: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let left = -width / 2.0;
    let right = width / 2.0;
    if t < left || t > right {
        return 0.0;
    }
    let peak = skew * width / 2.0;
    if t == peak {
        return 1.0;
    }
    if t < peak {
        let span = peak - left;
        if span <= 0.0 {
            1.0
        } else {
            (t - left) / span
        }
    } else {
        let span = right - peak;
        if span <= 0.0 {
            1.0
        } else {
            (right - t) / span
        }
    }
}

pub(crate) fn tripuls_tensor(tensor: Tensor, width: f64, skew: f64) -> Result<Tensor, String> {
    let shape = tensor.shape.clone();
    match tensor
        .into_numeric_storage()
        .map_err(|err| err.to_string())?
    {
        NumericStorage::F32(values) => Tensor::from_f32(
            values
                .into_iter()
                .map(|value| tripuls_scalar(f64::from(value), width, skew) as f32)
                .collect(),
            shape,
        )
        .map_err(|err| err.to_string()),
        storage => Tensor::new(
            storage
                .materialize_f64()
                .into_iter()
                .map(|value| tripuls_scalar(value, width, skew))
                .collect(),
            shape,
        )
        .map_err(|err| err.to_string()),
    }
}

pub(crate) fn validate_width(width: f64) -> Result<f64, String> {
    if !width.is_finite() || width <= 0.0 {
        Err(format!("got {width}"))
    } else {
        Ok(width)
    }
}

pub(crate) fn validate_skew(skew: f64) -> Result<f64, String> {
    if !skew.is_finite() || !(-1.0..=1.0).contains(&skew) {
        Err(format!("got {skew}"))
    } else {
        Ok(skew)
    }
}

#[runtime_builtin(
    name = "tripuls",
    category = "math/signal",
    summary = "Generate sampled triangular pulses.",
    keywords = "tripuls,triangular pulse,pulse train,signal processing,skew",
    type_resolver(numeric_unary_shape_type),
    descriptor(crate::builtins::math::signal::tripuls::TRIPULS_DESCRIPTOR),
    extensions(crate::builtins::math::signal::tripuls::TRIPULS_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::tripuls::TRIPULS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::tripuls"
)]
async fn tripuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&t)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIPULS_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &t,
        &TRIPULS_INTEGER_DATA_EXTENSION,
        BUILTIN_NAME,
        "sample times",
    )
    .await?;
    for value in &rest {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value,
            &TRIPULS_INTEGER_CONTROL_EXTENSION,
            BUILTIN_NAME,
            "width or skew",
        )
        .await?;
    }
    let (width, skew) = parse_options(&rest).await?;
    match t {
        Value::GpuTensor(handle) => tripuls_gpu(handle, width, skew).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(tripuls_error(&TRIPULS_ERROR_INVALID_INPUT))
        }
        other => tripuls_real(other, width, skew),
    }
}

async fn parse_options(rest: &[Value]) -> BuiltinResult<(f64, f64)> {
    if rest.len() > 2 {
        return Err(tripuls_error_with_detail(
            &TRIPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    let width = match rest.first() {
        Some(value) => {
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err))?
                .ok_or_else(|| tripuls_error(&TRIPULS_ERROR_WIDTH_INVALID))?;
            validate_width(raw).map_err(|err| {
                tripuls_error_with_detail(&TRIPULS_ERROR_WIDTH_INVALID, err.as_str())
            })?
        }
        None => 1.0,
    };
    let skew = match rest.get(1) {
        Some(value) => {
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err))?
                .ok_or_else(|| tripuls_error(&TRIPULS_ERROR_SKEW_INVALID))?;
            validate_skew(raw).map_err(|err| {
                tripuls_error_with_detail(&TRIPULS_ERROR_SKEW_INVALID, err.as_str())
            })?
        }
        None => 0.0,
    };
    Ok((width, skew))
}

async fn tripuls_gpu(handle: GpuTensorHandle, width: f64, skew: f64) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            tripuls_error_with_source(&TRIPULS_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    let result = tripuls_tensor(tensor, width, skew)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INTERNAL, err))?;
    let restored =
        gpu_helpers::restore_class_preserving_value(&handle, Value::Tensor(result), BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&handle)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(tripuls_error_with_detail(
            &TRIPULS_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn tripuls_real(value: Value, width: f64, skew: f64) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INVALID_INPUT, err))?;
    tripuls_tensor(tensor, width, skew)
        .map(tensor_into_value)
        .map_err(|err| tripuls_error_with_detail(&TRIPULS_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{CharArray, IntegerStorage};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(tripuls_builtin(t, rest))
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

    #[test]
    fn tripuls_default_samples_triangle() {
        let input = Tensor::new(vec![-0.5, -0.25, 0.0, 0.25, 0.5], vec![1, 5]).unwrap();
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("tripuls"));
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn tripuls_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let out =
            expect_tensor(call(Value::Tensor(input), vec![Value::Num(2.0)]).expect("tripuls"));
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn tripuls_preserves_documented_single_output_class() {
        let input = Tensor::from_f32(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("single input");
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("tripuls"));
        assert_eq!(
            out.into_numeric_storage().expect("single output"),
            NumericStorage::F32(vec![0.0, 1.0, 0.0])
        );
    }

    #[test]
    fn tripuls_integer_roles_are_independently_compatibility_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let input = integer_tensor(vec![-1, 0, 1], vec![1, 3]);
        let data_error = call(Value::Tensor(input), vec![Value::Num(2.0)])
            .expect_err("typed sample times are an extension");
        assert_eq!(
            data_error.identifier(),
            TRIPULS_INTEGER_DATA_EXTENSION.error_identifier
        );

        let control =
            Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1]).expect("typed width");
        let control_error = call(Value::Num(0.0), vec![Value::Tensor(control)])
            .expect_err("typed width is an extension");
        assert_eq!(
            control_error.identifier(),
            TRIPULS_INTEGER_CONTROL_EXTENSION.error_identifier
        );
    }

    #[test]
    fn tripuls_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("input");
            let automatic = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let _matlab_mode = crate::compatibility::push_runmat_extensions_enabled(false);
            let output =
                call(Value::GpuTensor(automatic), Vec::new()).expect("automatic resident input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected automatic resident output");
            };
            assert!(!runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );

            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit =
                explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let error = call(Value::GpuTensor(explicit), Vec::new())
                .expect_err("MATLAB mode rejects explicit interactive gpuArray input");
            assert_eq!(
                error.identifier(),
                TRIPULS_GPU_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn tripuls_runmat_gpu_extension_preserves_explicit_provenance() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![-0.5, 0.0, 0.5], vec![1, 3]).expect("input");
            let explicit = gpu_helpers::upload_tensor(provider, &input).expect("upload");
            let explicit =
                explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let _runmat_mode = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = call(Value::GpuTensor(explicit), Vec::new())
                .expect("RunMat mode accepts explicit gpuArray input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected explicit resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );
        });
    }

    #[test]
    fn tripuls_skew_moves_peak() {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(Value::Tensor(input), vec![Value::Num(2.0), Value::Num(1.0)]).expect("tripuls"),
        );
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn tripuls_rejects_bad_options_and_text_input() {
        let err = call(Value::Num(0.0), vec![Value::Num(-1.0)]).expect_err("width");
        assert_eq!(err.identifier(), TRIPULS_ERROR_WIDTH_INVALID.identifier);

        let err = call(Value::Num(0.0), vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("skew");
        assert_eq!(err.identifier(), TRIPULS_ERROR_SKEW_INVALID.identifier);

        let err =
            call(Value::CharArray(CharArray::new_row("abc")), Vec::new()).expect_err("text input");
        assert_eq!(err.identifier(), TRIPULS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn tripuls_is_registered() {
        assert!(builtin_function_by_name("tripuls").is_some());
    }
}
