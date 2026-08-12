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

const HANN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HANN.ARG_COUNT",
    identifier: Some("RunMat:hann:ArgumentCount"),
    when: "More than a sampling option and an output-type option are supplied.",
    message: "hann: too many input arguments",
};

const HANN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    HANN_ERROR_INVALID_LENGTH,
    HANN_ERROR_INVALID_OPTION,
    HANN_ERROR_UNKNOWN_OPTION,
    HANN_ERROR_INTERNAL,
    HANN_ERROR_ARG_COUNT,
];

const HANN_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hann-logical-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hann with a logical scalar length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HannLogicalLengthExtension"),
};

const HANN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HANN_LOGICAL_LENGTH_EXTENSION];

const HANN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "L",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "L accepts every built-in integer class as a nonnegative scalar length; floating L is rounded to the nearest integer.",
    }];

pub const HANN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "w = hann(L[, sflag][, typeName])",
        inputs: &HANN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default and sampling-only forms return a double column vector; typeName selects double or single. L is a host structural scalar with no interactive gpuArray overload; RunMat may materialize the new floating output on its active provider as an internal acceleration choice.",
    }];

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
    extensions(HANN_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::hann::HANN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::hann"
)]
async fn hann_builtin(
    n: runmat_value::Value,
    varargin: Vec<runmat_value::Value>,
) -> crate::BuiltinResult<runmat_value::Value> {
    validate_hann_options(&varargin)?;
    if matches!(n, runmat_value::Value::Bool(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HANN_LOGICAL_LENGTH_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
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
                if valid_provider_window(&handle, provider, options.len, precision) {
                    return Ok(runmat_value::Value::GpuTensor(handle));
                }
                free_rejected_provider_window(&handle);
            }
        }
    }
    window_tensor(options, |idx, total| {
        let denom = (total - 1) as f64;
        0.5 - 0.5 * cospi_real(2.0 * idx as f64 / denom)
    })
    .map_err(hann_map_window_error)
}

fn validate_hann_options(args: &[runmat_value::Value]) -> crate::BuiltinResult<()> {
    if args.len() > 2 {
        return Err(hann_error(&HANN_ERROR_ARG_COUNT));
    }
    let keywords = args
        .iter()
        .map(|arg| keyword(arg).ok_or_else(|| hann_error(&HANN_ERROR_INVALID_OPTION)))
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
        Err(hann_error_with_detail(
            &HANN_ERROR_UNKNOWN_OPTION,
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

fn free_rejected_provider_window(handle: &runmat_accelerate_api::GpuTensorHandle) {
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
        if owner.free(handle).is_ok() {
            runmat_accelerate_api::clear_residency(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type};
    use runmat_value::{IntValue, Value};

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
    fn hann_integer_metadata_and_logical_extension_are_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("hann builtin");
        assert_eq!(builtin.integer_capabilities.len(), 1);
        assert_eq!(builtin.extensions.len(), 1);

        let integer = block_on(super::hann_builtin(
            Value::Int(IntValue::U64(4)),
            Vec::new(),
        ))
        .expect("documented integer length");
        assert_eq!(test_support::gather(integer).unwrap().shape, vec![4, 1]);

        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::hann_builtin(Value::Bool(true), Vec::new()))
            .expect_err("strict mode rejects logical length");
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:HannLogicalLengthExtension")
        );
    }

    #[test]
    fn hann_rejects_extra_or_misordered_options() {
        let extra = block_on(super::hann_builtin(
            Value::Num(4.0),
            vec![
                Value::from("periodic"),
                Value::from("single"),
                Value::from("double"),
            ],
        ))
        .expect_err("too many options");
        assert_eq!(extra.identifier(), Some("RunMat:hann:ArgumentCount"));

        let misordered = block_on(super::hann_builtin(
            Value::Num(4.0),
            vec![Value::from("single"), Value::from("periodic")],
        ))
        .expect_err("sampling must precede output type");
        assert_eq!(misordered.identifier(), Some("RunMat:hann:UnknownOption"));
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
        for (got, want) in t.materialize_f64().iter().zip(expected.iter()) {
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
        assert!(zero.materialize_f64().is_empty());

        let one = test_support::gather(
            block_on(hann_builtin(Value::Num(1.0), Vec::new())).expect("hann(1)"),
        )
        .expect("gather hann(1)");
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.materialize_f64(), vec![1.0]);
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
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
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
        assert!((periodic.materialize_f64()[1] - 0.5).abs() < 1e-12);

        let single = test_support::gather(
            block_on(hann_builtin(Value::Num(4.0), vec![Value::from("single")]))
                .expect("hann single"),
        )
        .expect("gather hann single");
        assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[test]
    fn hann_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value = block_on(hann_builtin(Value::Num(8.0), Vec::new())).expect("hann gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.materialize_f64()[3] - 0.9504844339512095).abs() < 1e-12);

            let periodic = block_on(hann_builtin(Value::Num(4.0), vec![Value::from("periodic")]))
                .expect("hann periodic gpu");
            let periodic = test_support::gather(periodic).expect("gather periodic");
            assert_eq!(periodic.shape, vec![4, 1]);
            assert!((periodic.materialize_f64()[1] - 0.5).abs() < 1e-12);

            let periodic_one =
                block_on(hann_builtin(Value::Num(1.0), vec![Value::from("periodic")]))
                    .expect("hann periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.materialize_f64(), vec![1.0]);
        });
    }
}
