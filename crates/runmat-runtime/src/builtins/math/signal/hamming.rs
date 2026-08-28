use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::signal::common::{
    keyword, parse_window_options, provider_precision_matches, window_tensor, WindowArgError,
    WindowOutputType, WindowSampling,
};
use crate::builtins::math::signal::type_resolvers::window_vector_type;
use crate::builtins::math::trigonometry::pi_helpers::cospi_real;

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

const HAMMING_SIG_TYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const HAMMING_SIG_FULL_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const HAMMING_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
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
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, precision)",
        inputs: &HAMMING_SIG_TYPE_INPUTS,
        outputs: &HAMMING_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "w = hamming(n, sampling, precision)",
        inputs: &HAMMING_SIG_FULL_INPUTS,
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

const HAMMING_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HAMMING.ARG_COUNT",
    identifier: Some("RunMat:hamming:ArgumentCount"),
    when: "More than a sampling option and an output-type option are supplied.",
    message: "hamming: too many input arguments",
};

const HAMMING_ERRORS: [BuiltinErrorDescriptor; 5] = [
    HAMMING_ERROR_INVALID_LENGTH,
    HAMMING_ERROR_INVALID_OPTION,
    HAMMING_ERROR_UNKNOWN_OPTION,
    HAMMING_ERROR_INTERNAL,
    HAMMING_ERROR_ARG_COUNT,
];

const HAMMING_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hamming-logical-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hamming with a logical scalar length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HammingLogicalLengthExtension"),
};

const HAMMING_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HAMMING_LOGICAL_LENGTH_EXTENSION];

const HAMMING_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "L",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "L accepts every built-in integer class as a nonnegative scalar length; floating L is rounded to the nearest integer.",
    }];

pub const HAMMING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "w = hamming(L[, sflag][, typeName])",
        inputs: &HAMMING_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default and sampling-only forms return a double column vector; typeName selects double or single. L is a host structural scalar with no interactive gpuArray overload; RunMat may materialize the new floating output on its active provider as an internal acceleration choice.",
    }];

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
    extensions(HAMMING_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::hamming::HAMMING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::hamming"
)]
async fn hamming_builtin(
    n: runmat_value::Value,
    varargin: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    validate_hamming_options(&varargin)?;
    if matches!(n, runmat_value::Value::Bool(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HAMMING_LOGICAL_LENGTH_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let options = parse_window_options(n, &varargin, true).map_err(hamming_map_window_error)?;
    if options.len > 1 && provider_precision_matches(options.output_type) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.hamming_window(
                options.len,
                matches!(options.sampling, WindowSampling::Periodic),
            ) {
                let precision = match options.output_type {
                    WindowOutputType::Double => runmat_accelerate_api::ProviderPrecision::F64,
                    WindowOutputType::Single => runmat_accelerate_api::ProviderPrecision::F32,
                };
                if valid_provider_window(&handle, provider, options.len, precision) {
                    return Ok(runmat_value::Value::GpuTensor(handle));
                }
                free_rejected_provider_window(&handle, provider);
            }
        }
    }
    window_tensor(options, |idx, total| {
        let denom = (total - 1) as f64;
        0.54 - 0.46 * cospi_real(2.0 * idx as f64 / denom)
    })
    .map_err(hamming_map_window_error)
}

fn validate_hamming_options(args: &[runmat_value::Value]) -> crate::BuiltinResult<()> {
    if args.len() > 2 {
        return Err(hamming_error(&HAMMING_ERROR_ARG_COUNT));
    }
    let keywords = args
        .iter()
        .map(|arg| keyword(arg).ok_or_else(|| hamming_error(&HAMMING_ERROR_INVALID_OPTION)))
        .collect::<Result<Vec<_>, _>>()?;
    let valid = match keywords.as_slice() {
        [] => true,
        [option] => matches!(
            option.as_str(),
            "symmetric" | "periodic" | "double" | "single"
        ),
        [sampling, output_type] => {
            matches!(sampling.as_str(), "symmetric" | "periodic")
                && matches!(output_type.as_str(), "double" | "single")
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        let option = keywords
            .iter()
            .find(|option| {
                !matches!(
                    option.as_str(),
                    "symmetric" | "periodic" | "double" | "single"
                )
            })
            .or_else(|| keywords.first())
            .map(String::as_str)
            .unwrap_or_default();
        Err(hamming_error_with_detail(
            &HAMMING_ERROR_UNKNOWN_OPTION,
            format!("'{option}'"),
        ))
    }
}

fn valid_provider_window(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    len: usize,
    precision: runmat_accelerate_api::ProviderPrecision,
) -> bool {
    handle.shape == [len, 1]
        && handle.device_id == provider.device_id()
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(handle) == Some(precision)
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_provider_window(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
) {
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(invoked_provider);
    let _ = owner.free(handle);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider as _;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{IntValue, NumericDType, Value};

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
        for (got, want) in t.materialize_f64().iter().zip(expected.iter()) {
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
        assert!(labels.contains(&"w = hamming(n, precision)"));
        assert!(labels.contains(&"w = hamming(n, sampling, precision)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.HAMMING.INVALID_LENGTH"));
        assert_eq!(HAMMING_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            HAMMING_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
        assert_eq!(HAMMING_EXTENSIONS, [HAMMING_LOGICAL_LENGTH_EXTENSION]);
    }

    #[test]
    fn hamming_handles_zero_and_one_lengths() {
        let _guard = test_support::accel_test_lock();
        let zero = test_support::gather(
            block_on(hamming_builtin(Value::Num(0.0), Vec::new())).expect("hamming(0)"),
        )
        .expect("gather hamming(0)");
        assert_eq!(zero.shape, vec![0, 1]);
        assert!(zero.materialize_f64().is_empty());

        let one = test_support::gather(
            block_on(hamming_builtin(Value::Num(1.0), Vec::new())).expect("hamming(1)"),
        )
        .expect("gather hamming(1)");
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.materialize_f64(), vec![1.0]);
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
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
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
        assert!((periodic.materialize_f64()[1] - 0.54).abs() < 1e-12);
    }

    #[test]
    fn hamming_accepts_every_integer_length_class_and_selects_output_precision() {
        let _guard = test_support::accel_test_lock();
        for length in [
            IntValue::I8(5),
            IntValue::I16(5),
            IntValue::I32(5),
            IntValue::I64(5),
            IntValue::U8(5),
            IntValue::U16(5),
            IntValue::U32(5),
            IntValue::U64(5),
        ] {
            let output = test_support::gather(
                block_on(hamming_builtin(Value::Int(length.clone()), Vec::new()))
                    .expect("integer length"),
            )
            .expect("gather default output");
            assert_eq!(output.shape, vec![5, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F64);

            let single = test_support::gather(
                block_on(hamming_builtin(
                    Value::Int(length),
                    vec![Value::from("single")],
                ))
                .expect("integer length with single output"),
            )
            .expect("gather single output");
            assert_eq!(single.shape, vec![5, 1]);
            assert_eq!(single.numeric_dtype(), NumericDType::F32);
        }
    }

    #[test]
    fn hamming_logical_length_is_mode_gated_and_argument_count_precedes_the_gate() {
        let _guard = test_support::accel_test_lock();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(hamming_builtin(Value::Bool(true), Vec::new()))
                .expect_err("logical length must be gated");
            assert_eq!(
                error.identifier(),
                HAMMING_LOGICAL_LENGTH_EXTENSION.error_identifier
            );
            let error = block_on(hamming_builtin(
                Value::Bool(true),
                vec![
                    Value::from("symmetric"),
                    Value::from("double"),
                    Value::from("single"),
                ],
            ))
            .expect_err("argument count must reject first");
            assert_eq!(error.identifier(), HAMMING_ERROR_ARG_COUNT.identifier);
        }

        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = test_support::gather(
                block_on(hamming_builtin(Value::Bool(true), Vec::new()))
                    .expect("RunMat mode accepts logical length"),
            )
            .expect("gather logical-length output");
            assert_eq!(output.shape, vec![1, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
        }
    }

    #[test]
    fn hamming_rejects_reversed_duplicate_and_excess_options() {
        let _guard = test_support::accel_test_lock();
        for options in [
            vec![Value::from("single"), Value::from("periodic")],
            vec![Value::from("periodic"), Value::from("periodic")],
        ] {
            let error = block_on(hamming_builtin(Value::Num(5.0), options))
                .expect_err("invalid option order");
            assert_eq!(error.identifier(), HAMMING_ERROR_UNKNOWN_OPTION.identifier);
        }
        let error = block_on(hamming_builtin(
            Value::Num(5.0),
            vec![
                Value::from("symmetric"),
                Value::from("double"),
                Value::from("single"),
            ],
        ))
        .expect_err("too many options");
        assert_eq!(error.identifier(), HAMMING_ERROR_ARG_COUNT.identifier);
    }

    #[test]
    fn hamming_uses_cospi_coefficient_accuracy() {
        let _guard = test_support::accel_test_lock();
        let output = test_support::gather(
            block_on(hamming_builtin(Value::Int(IntValue::U8(4)), Vec::new()))
                .expect("four-point window"),
        )
        .expect("gather four-point window");
        let improved = 0.54 - 0.46 * cospi_real(4.0 / 3.0);
        let legacy = 0.54 - 0.46 * (4.0 * std::f64::consts::PI / 3.0).cos();
        assert_ne!(improved.to_bits(), legacy.to_bits());
        assert_eq!(output.materialize_f64()[2].to_bits(), improved.to_bits());
    }

    #[test]
    fn hamming_rejects_negative_huge_complex_and_resident_lengths() {
        {
            let _guard = test_support::accel_test_lock();
            assert!(block_on(hamming_builtin(Value::Int(IntValue::I64(-1)), Vec::new())).is_err());
            assert!(block_on(hamming_builtin(
                Value::Int(IntValue::U64(u64::MAX)),
                vec![Value::from("periodic")],
            ))
            .is_err());
            assert!(block_on(hamming_builtin(Value::Complex(4.0, 0.0), Vec::new())).is_err());
        }
        test_support::with_test_provider(|provider| {
            let length = runmat_value::Tensor::new_integer(
                runmat_value::IntegerStorage::U16(vec![4]),
                vec![1, 1],
            )
            .expect("length");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &length)
                .expect("upload resident length");
            assert!(block_on(hamming_builtin(Value::GpuTensor(handle), Vec::new())).is_err());
        });
    }

    #[test]
    fn hamming_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value =
                block_on(hamming_builtin(Value::Num(8.0), Vec::new())).expect("hamming gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.materialize_f64()[0] - 0.08).abs() < 1e-12);

            let periodic_one = block_on(hamming_builtin(
                Value::Num(1.0),
                vec![Value::from("periodic")],
            ))
            .expect("hamming periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.materialize_f64(), vec![1.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn hamming_integer_length_generates_current_window_on_wgpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let output_type = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => "double",
            runmat_accelerate_api::ProviderPrecision::F32 => "single",
        };
        let result = block_on(hamming_builtin(
            Value::Int(IntValue::U16(4)),
            vec![Value::from("periodic"), Value::from(output_type)],
        ))
        .expect("WGPU Hamming window");
        assert!(matches!(result, Value::GpuTensor(_)));
        let output = test_support::gather(result).expect("gather WGPU window");
        assert_eq!(output.shape, vec![4, 1]);
        let expected = 0.54 - 0.46 * cospi_real(0.5);
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-14,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-6,
        };
        assert!((output.materialize_f64()[1] - expected).abs() < tolerance);
    }
}
