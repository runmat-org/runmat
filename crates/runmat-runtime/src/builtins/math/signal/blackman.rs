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

const BUILTIN_NAME: &str = "blackman";

const BLACKMAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Blackman window column vector.",
}];

const BLACKMAN_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Window length.",
}];

const BLACKMAN_SIG_SAMPLING_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const BLACKMAN_SIG_TYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const BLACKMAN_SIG_FULL_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const BLACKMAN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "w = blackman(n)",
        inputs: &BLACKMAN_SIG_N_INPUTS,
        outputs: &BLACKMAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, sampling)",
        inputs: &BLACKMAN_SIG_SAMPLING_INPUTS,
        outputs: &BLACKMAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, precision)",
        inputs: &BLACKMAN_SIG_TYPE_INPUTS,
        outputs: &BLACKMAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = blackman(n, sampling, precision)",
        inputs: &BLACKMAN_SIG_FULL_INPUTS,
        outputs: &BLACKMAN_OUTPUT,
    },
];

const BLACKMAN_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INVALID_LENGTH",
    identifier: Some("RunMat:blackman:InvalidLength"),
    when: "Length input is not a finite nonnegative scalar value.",
    message: "blackman: expected a nonnegative scalar integer length",
};

const BLACKMAN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INVALID_OPTION",
    identifier: Some("RunMat:blackman:InvalidOption"),
    when: "An option argument is not a string-like sampling/precision token.",
    message: "blackman: unrecognized option",
};

const BLACKMAN_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.UNKNOWN_OPTION",
    identifier: Some("RunMat:blackman:UnknownOption"),
    when: "An option string is not recognized by blackman.",
    message: "blackman: unrecognized option",
};

const BLACKMAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLACKMAN.INTERNAL",
    identifier: Some("RunMat:blackman:InternalError"),
    when: "Window materialization fails internally.",
    message: "blackman: internal error",
};

const BLACKMAN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BLACKMAN_ERROR_INVALID_LENGTH,
    BLACKMAN_ERROR_INVALID_OPTION,
    BLACKMAN_ERROR_UNKNOWN_OPTION,
    BLACKMAN_ERROR_INTERNAL,
];

pub const BLACKMAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLACKMAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BLACKMAN_ERRORS,
};

fn blackman_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    blackman_error_with_message(error.message, error)
}

fn blackman_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    blackman_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn blackman_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn blackman_map_window_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::InvalidLength => blackman_error(&BLACKMAN_ERROR_INVALID_LENGTH),
        WindowArgError::InvalidOptionType => blackman_error(&BLACKMAN_ERROR_INVALID_OPTION),
        WindowArgError::UnknownOption(option) => {
            blackman_error_with_detail(&BLACKMAN_ERROR_UNKNOWN_OPTION, format!("'{option}'"))
        }
        WindowArgError::TensorBuild(detail) => {
            blackman_error_with_detail(&BLACKMAN_ERROR_INTERNAL, detail)
        }
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "blackman",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("blackman_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Blackman window directly on the active provider when the custom hook is available; otherwise falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "blackman",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "blackman materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "blackman",
    category = "math/signal",
    summary = "Generate a Blackman window vector.",
    keywords = "blackman,window,signal processing,dsp,fft",
    type_resolver(window_vector_type),
    descriptor(crate::builtins::math::signal::blackman::BLACKMAN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::blackman"
)]
async fn blackman_builtin(
    n: runmat_builtins::Value,
    varargin: Vec<runmat_builtins::Value>,
) -> crate::BuiltinResult<runmat_builtins::Value> {
    let options = parse_window_options(n, &varargin, true).map_err(blackman_map_window_error)?;
    if options.len > 1 && provider_precision_matches(options.output_type) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.blackman_window(
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
        0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos()
    })
    .map_err(blackman_map_window_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, Value};

    #[test]
    fn blackman_returns_expected_values() {
        let _guard = test_support::accel_test_lock();
        let t = test_support::gather(
            block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman"),
        )
        .expect("gather blackman");
        let expected = [
            -1.3877787807814457e-17,
            0.09045342435412812,
            0.45918295754596355,
            0.9203636180999081,
            0.9203636180999083,
            0.45918295754596383,
            0.09045342435412818,
            -1.3877787807814457e-17,
        ];
        assert_eq!(t.shape, vec![8, 1]);
        for (got, want) in t.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn blackman_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("blackman builtin");
        let descriptor = builtin.descriptor.expect("blackman descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"w = blackman(n)"));
        assert!(labels.contains(&"w = blackman(n, sampling)"));
        assert!(labels.contains(&"w = blackman(n, precision)"));
        assert!(labels.contains(&"w = blackman(n, sampling, precision)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.BLACKMAN.INVALID_LENGTH"));
    }

    #[test]
    fn blackman_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(blackman_builtin(Value::Num(0.0), Vec::new())).expect("blackman(0)"),
        )
        .expect("gather blackman(0)");
        assert_eq!(zero.shape, vec![0, 1]);
        assert!(zero.data.is_empty());

        let one = test_support::gather(
            block_on(blackman_builtin(Value::Num(1.0), Vec::new())).expect("blackman(1)"),
        )
        .expect("gather blackman(1)");
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.data, vec![1.0]);
    }

    #[test]
    fn blackman_rejects_invalid_lengths() {
        let _guard = test_support::accel_test_lock();
        assert!(block_on(blackman_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let rounded = test_support::gather(
            block_on(blackman_builtin(Value::Num(2.5), Vec::new())).expect("blackman rounded"),
        )
        .expect("gather blackman rounded");
        assert_eq!(rounded.shape, vec![3, 1]);
        assert!(block_on(blackman_builtin(
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn blackman_supports_periodic_and_single_overloads() {
        let _guard = test_support::accel_test_lock();
        let periodic = test_support::gather(
            block_on(blackman_builtin(
                Value::Num(4.0),
                vec![Value::from("periodic")],
            ))
            .expect("blackman periodic"),
        )
        .expect("gather blackman periodic");
        assert_eq!(periodic.shape, vec![4, 1]);
        assert!((periodic.data[1] - 0.34).abs() < 1e-12);

        let single = test_support::gather(
            block_on(blackman_builtin(
                Value::Num(4.0),
                vec![Value::from("single")],
            ))
            .expect("blackman single"),
        )
        .expect("gather blackman single");
        assert_eq!(single.dtype, runmat_builtins::NumericDType::F32);
    }

    #[test]
    fn blackman_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value =
                block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.data[3] - 0.9203636180999081).abs() < 1e-12);

            let periodic_one = block_on(blackman_builtin(
                Value::Num(1.0),
                vec![Value::from("periodic")],
            ))
            .expect("blackman periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.data, vec![1.0]);
        });
    }
}
