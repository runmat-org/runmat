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
    parse_window_options, provider_precision_matches, window_tensor, WindowArgError, WindowSampling,
};
use crate::builtins::math::signal::type_resolvers::window_vector_type;

const BUILTIN_NAME: &str = "hamming";

const HAMMING_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hamming window column vector.",
}];

const HAMMING_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Window length.",
}];

const HAMMING_SIG_SAMPLING_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const HAMMING_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "w = hamming(n)",
        inputs: &HAMMING_SIG_N_INPUTS,
        outputs: &HAMMING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, sampling)",
        inputs: &HAMMING_SIG_SAMPLING_INPUTS,
        outputs: &HAMMING_OUTPUT,
    },
];

const HAMMING_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INVALID_LENGTH",
    identifier: Some("RunMat:hamming:InvalidLength"),
    when: "Length input is not a finite nonnegative scalar value.",
    message: "hamming: expected a nonnegative scalar integer length",
};

const HAMMING_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INVALID_OPTION",
    identifier: Some("RunMat:hamming:InvalidOption"),
    when: "An option argument is not a string-like sampling token.",
    message: "hamming: unrecognized option",
};

const HAMMING_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.UNKNOWN_OPTION",
    identifier: Some("RunMat:hamming:UnknownOption"),
    when: "An option string is not recognized by hamming.",
    message: "hamming: unrecognized option",
};

const HAMMING_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.INTERNAL",
    identifier: Some("RunMat:hamming:InternalError"),
    when: "Window materialization fails internally.",
    message: "hamming: internal error",
};

const HAMMING_ERRORS: [BuiltinErrorDescriptor; 4] = [
    HAMMING_ERROR_INVALID_LENGTH,
    HAMMING_ERROR_INVALID_OPTION,
    HAMMING_ERROR_UNKNOWN_OPTION,
    HAMMING_ERROR_INTERNAL,
];

pub const HAMMING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HAMMING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HAMMING_ERRORS,
};

fn hamming_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hamming_error_with_message(error.message, error)
}

fn hamming_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    hamming_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn hamming_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hamming_map_window_error(error: WindowArgError) -> RuntimeError {
    match error {
        WindowArgError::InvalidLength => hamming_error(&HAMMING_ERROR_INVALID_LENGTH),
        WindowArgError::InvalidOptionType => hamming_error(&HAMMING_ERROR_INVALID_OPTION),
        WindowArgError::UnknownOption(option) => {
            hamming_error_with_detail(&HAMMING_ERROR_UNKNOWN_OPTION, format!("'{option}'"))
        }
        WindowArgError::TensorBuild(detail) => {
            hamming_error_with_detail(&HAMMING_ERROR_INTERNAL, detail)
        }
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::hamming")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hamming",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("hamming_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Hamming window directly on the active provider when the custom hook is available; otherwise falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::hamming")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hamming",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "hamming materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "hamming",
    category = "math/signal",
    summary = "Generate Hamming windows.",
    keywords = "hamming,window,signal processing,dsp,fft",
    type_resolver(window_vector_type),
    descriptor(crate::builtins::math::signal::hamming::HAMMING_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::hamming"
)]
async fn hamming_builtin(
    n: runmat_builtins::Value,
    varargin: Vec<runmat_builtins::Value>,
) -> crate::BuiltinResult<runmat_builtins::Value> {
    let options = parse_window_options(n, &varargin, false).map_err(hamming_map_window_error)?;
    if options.len > 1 && provider_precision_matches(options.output_type) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.hamming_window(
                options.len,
                matches!(options.sampling, WindowSampling::Periodic),
            ) {
                return Ok(runmat_builtins::Value::GpuTensor(handle));
            }
        }
    }
    window_tensor(options, |idx, total| {
        let denom = (total - 1) as f64;
        let phase = 2.0 * std::f64::consts::PI * idx as f64 / denom;
        0.54 - 0.46 * phase.cos()
    })
    .map_err(hamming_map_window_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, Value};

    #[test]
    fn hamming_returns_expected_values() {
        let _guard = test_support::accel_test_lock();
        let t = test_support::gather(
            block_on(hamming_builtin(Value::Num(8.0), Vec::new())).expect("hamming"),
        )
        .expect("gather hamming");
        let expected = [
            0.08,
            0.25319469114498255,
            0.6423596296199047,
            0.9544456792351128,
            0.9544456792351128,
            0.6423596296199048,
            0.25319469114498266,
            0.08,
        ];
        assert_eq!(t.shape, vec![8, 1]);
        for (got, want) in t.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn hamming_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("hamming builtin");
        let descriptor = builtin.descriptor.expect("hamming descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"w = hamming(n)"));
        assert!(labels.contains(&"w = hamming(n, sampling)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.HAMMING.INVALID_LENGTH"));
    }

    #[test]
    fn hamming_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(hamming_builtin(Value::Num(0.0), Vec::new())).expect("hamming(0)"),
        )
        .expect("gather hamming(0)");
        assert_eq!(zero.shape, vec![0, 1]);
        assert!(zero.data.is_empty());

        let one = test_support::gather(
            block_on(hamming_builtin(Value::Num(1.0), Vec::new())).expect("hamming(1)"),
        )
        .expect("gather hamming(1)");
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.data, vec![1.0]);
    }

    #[test]
    fn hamming_rejects_invalid_lengths() {
        let _guard = test_support::accel_test_lock();
        assert!(block_on(hamming_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let rounded = test_support::gather(
            block_on(hamming_builtin(Value::Num(2.5), Vec::new())).expect("hamming rounded"),
        )
        .expect("gather hamming rounded");
        assert_eq!(rounded.shape, vec![3, 1]);
        assert!(block_on(hamming_builtin(
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn hamming_supports_periodic_overload() {
        let _guard = test_support::accel_test_lock();
        let periodic = test_support::gather(
            block_on(hamming_builtin(
                Value::Num(4.0),
                vec![Value::from("periodic")],
            ))
            .expect("hamming periodic"),
        )
        .expect("gather hamming periodic");
        assert_eq!(periodic.shape, vec![4, 1]);
        assert!((periodic.data[1] - 0.54).abs() < 1e-12);
    }

    #[test]
    fn hamming_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value =
                block_on(hamming_builtin(Value::Num(8.0), Vec::new())).expect("hamming gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.data[0] - 0.08).abs() < 1e-12);

            let periodic_one = block_on(hamming_builtin(
                Value::Num(1.0),
                vec![Value::from("periodic")],
            ))
            .expect("hamming periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.data, vec![1.0]);
        });
    }
}
