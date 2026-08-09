//! Canonical `close` builtin dispatcher.
//!
//! This module owns the single runtime registration for `close` and routes
//! requests to plotting or networking close handlers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const CLOSE_OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar double status: 1 when the requested close operation completes, 0 when it is refused.",
}];

const CLOSE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CLOSE_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Figure target, tcp resource handle, option token, or target container.",
}];
const CLOSE_INPUTS_TARGETS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more close targets.",
}];

const CLOSE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "result = close()",
        inputs: &CLOSE_INPUTS_NONE,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = close(target)",
        inputs: &CLOSE_INPUTS_TARGET,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = close(targets...)",
        inputs: &CLOSE_INPUTS_TARGETS,
        outputs: &CLOSE_OUTPUT_RESULT,
    },
];

const CLOSE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:close:InvalidArgument"),
    when: "Close target values are invalid or unsupported.",
    message: "close: invalid argument",
};
const CLOSE_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INVALID_HANDLE",
    identifier: Some("RunMat:close:InvalidHandle"),
    when: "A structure target is not a valid networking resource handle.",
    message: "close: invalid handle",
};
const CLOSE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLOSE.INTERNAL",
    identifier: None,
    when: "Internal networking, gather, or plotting close processing fails.",
    message: "close: internal error",
};
const CLOSE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CLOSE_ERROR_INVALID_ARGUMENT,
    CLOSE_ERROR_INVALID_HANDLE,
    CLOSE_ERROR_INTERNAL,
];

pub const CLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLOSE_ERRORS,
};

pub(crate) const CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-integer-figure-number",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "close with a typed integer figure number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CloseIntegerFigureNumberExtension"),
    };

pub(crate) const CLOSE_VARIADIC_TARGETS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "close-variadic-targets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "close with separate variadic targets is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CloseVariadicTargetsExtension"),
    };

pub const CLOSE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLOSE_VARIADIC_TARGETS_EXTENSION,
];

const CLOSE_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode interprets a scalar or array of any built-in integer class as figure numbers. The public compatibility contract documents figure numbers but does not advertise typed integer classes.",
    }];

pub const CLOSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "status = close(integer_fig)",
        inputs: &CLOSE_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Figure numbers are read from authoritative host integer storage and must be positive and representable as RunMat's u32 figure identifier. Resident numeric targets are rejected before networking/provider gather. Successful and no-op plotting closures return scalar double 1; callback-driven refusal remains unavailable until CloseRequestFcn is implemented.",
    }];

#[runtime_builtin(
    name = "close",
    category = "general",
    summary = "Close figures or networking resources.",
    keywords = "close,figure,tcpclient,tcpserver,networking",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::close_type),
    descriptor(crate::builtins::close::CLOSE_DESCRIPTOR),
    extensions(crate::builtins::close::CLOSE_EXTENSIONS),
    integer_capabilities(crate::builtins::close::CLOSE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::close"
)]
pub async fn close_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if args.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLOSE_VARIADIC_TARGETS_EXTENSION,
            "close",
        )?;
    }
    if args.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
            "close",
        )?;
    }
    if let Some(status) = crate::builtins::io::net::close::close_if_network_targets(&args).await? {
        return Ok(status);
    }

    close_plotting_targets(&args)
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[cfg(feature = "plot-core")]
fn close_plotting_targets(args: &[Value]) -> crate::BuiltinResult<f64> {
    crate::builtins::plotting::close::close_plot_targets(args)
}

#[cfg(not(feature = "plot-core"))]
fn close_plotting_targets(_args: &[Value]) -> crate::BuiltinResult<f64> {
    let mut builder =
        crate::build_runtime_error(CLOSE_ERROR_INVALID_ARGUMENT.message).with_builtin("close");
    if let Some(identifier) = CLOSE_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn compatibility_mode_rejects_all_integer_figure_number_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let storages = [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];

        for storage in storages {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("figure"));
            let err = futures::executor::block_on(close_builtin(vec![value]))
                .expect_err("typed integer extension must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:CloseIntegerFigureNumberExtension")
            );
        }
    }

    #[test]
    fn close_integer_capability_is_structural_and_host_only() {
        let capability = &CLOSE_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
    }

    #[test]
    fn compatibility_mode_rejects_separate_variadic_targets_before_dispatch() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = futures::executor::block_on(close_builtin(vec![
            Value::String("clients".into()),
            Value::String("servers".into()),
        ]))
        .expect_err("RunMat-only variadic close targets");
        assert_eq!(
            err.identifier(),
            CLOSE_VARIADIC_TARGETS_EXTENSION.error_identifier
        );
    }

    #[test]
    fn resident_numeric_figure_target_rejects_without_provider_dispatch() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = futures::executor::block_on(close_builtin(vec![resident]))
            .expect_err("resident figure target");
        assert!(!err.message().to_ascii_lowercase().contains("provider"));
    }

    #[test]
    fn invalid_network_structure_uses_canonical_invalid_handle_error() {
        let invalid = Value::Struct(runmat_builtins::StructValue::new());
        let err = futures::executor::block_on(close_builtin(vec![invalid]))
            .expect_err("invalid networking handle");
        assert_eq!(err.identifier(), Some("RunMat:close:InvalidHandle"));
        assert!(CLOSE_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.identifier == Some("RunMat:close:InvalidHandle")));
    }
}
