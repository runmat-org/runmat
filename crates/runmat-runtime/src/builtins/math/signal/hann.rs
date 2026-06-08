use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::signal::common::{
    parse_window_options, provider_precision_matches, window_tensor, WindowArgError,
    WindowOutputType, WindowSampling,
};
use crate::builtins::math::signal::type_resolvers::window_vector_type;

const BUILTIN_NAME: &str = "hann";

const HANN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hann window column vector.",
}];

const HANN_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Window length.",
}];

const HANN_SIG_SAMPLING_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Window length.",
    },
    BuiltinParamDescriptor {
        name: "sampling",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"symmetric\""),
        description: "Sampling mode: \"symmetric\" or \"periodic\".",
    },
];

const HANN_SIG_TYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Window length.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Output precision: \"double\" or \"single\".",
    },
];

const HANN_SIG_FULL_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Window length.",
    },
    BuiltinParamDescriptor {
        name: "sampling",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"symmetric\""),
        description: "Sampling mode: \"symmetric\" or \"periodic\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Output precision: \"double\" or \"single\".",
    },
];

const HANN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "w = hann(n)",
        inputs: &HANN_SIG_N_INPUTS,
        outputs: &HANN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, sampling)",
        inputs: &HANN_SIG_SAMPLING_INPUTS,
        outputs: &HANN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, precision)",
        inputs: &HANN_SIG_TYPE_INPUTS,
        outputs: &HANN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hann(n, sampling, precision)",
        inputs: &HANN_SIG_FULL_INPUTS,
        outputs: &HANN_OUTPUT,
    },
];

const HANN_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INVALID_LENGTH",
    identifier: Some("RunMat:hann:InvalidLength"),
    when: "Length input is not a finite nonnegative scalar value.",
    message: "hann: expected a nonnegative scalar integer length",
};

const HANN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INVALID_OPTION",
    identifier: Some("RunMat:hann:InvalidOption"),
    when: "An option argument is not a string-like sampling/precision token.",
    message: "hann: unrecognized option",
};

const HANN_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.UNKNOWN_OPTION",
    identifier: Some("RunMat:hann:UnknownOption"),
    when: "An option string is not recognized by hann.",
    message: "hann: unrecognized option",
};

const HANN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.INTERNAL",
    identifier: Some("RunMat:hann:InternalError"),
    when: "Window materialization fails internally.",
    message: "hann: internal error",
};

const HANN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    HANN_ERROR_INVALID_LENGTH,
    HANN_ERROR_INVALID_OPTION,
    HANN_ERROR_UNKNOWN_OPTION,
    HANN_ERROR_INTERNAL,
];

pub const HANN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HANN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HANN_ERRORS,
};

fn hann_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hann_error_with_message(error.message, error)
}

fn hann_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    hann_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn hann_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hann_map_window_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::InvalidLength => hann_error(&HANN_ERROR_INVALID_LENGTH),
        WindowArgError::InvalidOptionType => hann_error(&HANN_ERROR_INVALID_OPTION),
        WindowArgError::UnknownOption(option) => {
            hann_error_with_detail(&HANN_ERROR_UNKNOWN_OPTION, format!("'{option}'"))
        }
        WindowArgError::TensorBuild(detail) => hann_error_with_detail(&HANN_ERROR_INTERNAL, detail),
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::hann")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hann",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("hann_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Hann window directly on the active provider when the custom hook is available; otherwise falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::hann")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hann",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "hann materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "hann",
    category = "math/signal",
    summary = "Generate Hann windows.",
    keywords = "hann,window,signal processing,dsp,fft",
    type_resolver(window_vector_type),
    descriptor(crate::builtins::math::signal::hann::HANN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::hann"
)]
async fn hann_builtin(
    n: runmat_builtins::Value,
    varargin: Vec<runmat_builtins::Value>,
) -> crate::BuiltinResult<runmat_builtins::Value> {
    let options = parse_window_options(n, &varargin, true).map_err(hann_map_window_error)?;
    if options.len > 1 && provider_precision_matches(options.output_type) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.hann_window(
                options.len,
                matches!(options.sampling, WindowSampling::Periodic),
            ) {
                let precision = match options.output_type {
                    WindowOutputType::Double => runmat_accelerate_api::ProviderPrecision::F64,
                    WindowOutputType::Single => runmat_accelerate_api::ProviderPrecision::F32,
                };
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(runmat_builtins::Value::GpuTensor(handle));
            }
        }
    }
    window_tensor(options, |idx, total| {
        let denom = (total - 1) as f64;
        let phase = 2.0 * std::f64::consts::PI * idx as f64 / denom;
        0.5 - 0.5 * phase.cos()
    })
    .map_err(hann_map_window_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type, Value};

    #[test]
    fn hann_type_uses_literal_length() {
        let out = window_vector_type(
            &[Type::Num],
            &ResolveContext::new(vec![runmat_builtins::LiteralValue::Number(8.0)]),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(8), Some(1)])
            }
        );
    }

    #[test]
    fn hann_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("hann builtin");
        let descriptor = builtin.descriptor.expect("hann descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"w = hann(n)"));
        assert!(labels.contains(&"w = hann(n, sampling)"));
        assert!(labels.contains(&"w = hann(n, precision)"));
        assert!(labels.contains(&"w = hann(n, sampling, precision)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.HANN.INVALID_LENGTH"));
    }

    #[test]
    fn hann_returns_expected_values() {
        let _guard = test_support::accel_test_lock();
        let t = test_support::gather(
            block_on(hann_builtin(Value::Num(8.0), Vec::new())).expect("hann"),
        )
        .expect("gather hann");
        let expected = [
            0.0,
            0.1882550990706332,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781573,
            0.1882550990706333,
            0.0,
        ];
        assert_eq!(t.shape, vec![8, 1]);
        for (got, want) in t.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn hann_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(hann_builtin(Value::Num(0.0), Vec::new())).expect("hann(0)"),
        )
        .expect("gather hann(0)");
        assert_eq!(zero.shape, vec![0, 1]);
        assert!(zero.data.is_empty());

        let one = test_support::gather(
            block_on(hann_builtin(Value::Num(1.0), Vec::new())).expect("hann(1)"),
        )
        .expect("gather hann(1)");
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.data, vec![1.0]);
    }

    #[test]
    fn hann_rejects_invalid_lengths() {
        let _guard = test_support::accel_test_lock();
        assert!(block_on(hann_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let rounded = test_support::gather(
            block_on(hann_builtin(Value::Num(2.5), Vec::new())).expect("hann rounded"),
        )
        .expect("gather hann rounded");
        assert_eq!(rounded.shape, vec![3, 1]);
        assert!(block_on(hann_builtin(
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn hann_supports_periodic_and_single_overloads() {
        let _guard = test_support::accel_test_lock();
        let periodic = test_support::gather(
            block_on(hann_builtin(Value::Num(4.0), vec![Value::from("periodic")]))
                .expect("hann periodic"),
        )
        .expect("gather hann periodic");
        assert_eq!(periodic.shape, vec![4, 1]);
        assert!((periodic.data[1] - 0.5).abs() < 1e-12);

        let single = test_support::gather(
            block_on(hann_builtin(Value::Num(4.0), vec![Value::from("single")]))
                .expect("hann single"),
        )
        .expect("gather hann single");
        assert_eq!(single.dtype, runmat_builtins::NumericDType::F32);
    }

    #[test]
    fn hann_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value = block_on(hann_builtin(Value::Num(8.0), Vec::new())).expect("hann gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.data[3] - 0.9504844339512095).abs() < 1e-12);

            let periodic = block_on(hann_builtin(Value::Num(4.0), vec![Value::from("periodic")]))
                .expect("hann periodic gpu");
            let periodic = test_support::gather(periodic).expect("gather periodic");
            assert_eq!(periodic.shape, vec![4, 1]);
            assert!((periodic.data[1] - 0.5).abs() < 1e-12);

            let periodic_one =
                block_on(hann_builtin(Value::Num(1.0), vec![Value::from("periodic")]))
                    .expect("hann periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.data, vec![1.0]);
        });
    }
}
