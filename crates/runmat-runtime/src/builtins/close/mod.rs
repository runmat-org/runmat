//! Canonical `close` builtin dispatcher.
//!
//! This module owns the single runtime registration for `close` and routes
//! requests to plotting or networking close handlers.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const CLOSE_OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closed handle for single-target calls or count/status for multi/all closures.",
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
const CLOSE_ERRORS: [BuiltinErrorDescriptor; 1] = [CLOSE_ERROR_INVALID_ARGUMENT];

pub const CLOSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLOSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLOSE_ERRORS,
};

#[runtime_builtin(
    name = "close",
    category = "general",
    summary = "Close plotting figures or networking resources.",
    keywords = "close,figure,tcpclient,tcpserver,networking",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::close_type),
    descriptor(crate::builtins::close::CLOSE_DESCRIPTOR),
    builtin_path = "crate::builtins::close"
)]
pub async fn close_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    if let Some(status) = crate::builtins::io::net::close::close_if_network_targets(&args).await? {
        return Ok(status);
    }

    close_plotting_targets(&args)
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
