use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

const POOL_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pool",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Execution pool owned by the current RunMat session.",
}];
const PARPOOL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional pool kind and worker count.",
}];
const GCP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Use 'nocreate' to inspect the current pool without creating one.",
}];
const PARPOOL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = parpool(___)",
    inputs: &PARPOOL_INPUTS,
    outputs: &POOL_OUTPUTS,
}];
const GCP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = gcp(option)",
    inputs: &GCP_INPUTS,
    outputs: &POOL_OUTPUTS,
}];
const POOL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.POOL_INVALID_INPUT",
    identifier: Some("RunMat:parpool:InvalidInput"),
    when: "The requested pool kind, worker count, or option is invalid for the active host.",
    message: "parpool: invalid pool request",
}];
pub const PARPOOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARPOOL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};
pub const GCP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};

fn active_context(operation: &str) -> crate::BuiltinResult<crate::context::RuntimeContext> {
    crate::context::legacy::active().ok_or_else(|| {
        crate::build_runtime_error(format!("{operation}: no active runtime context"))
            .with_builtin(operation)
            .with_identifier("RunMat:parallel:RuntimeContextUnavailable")
            .build()
    })
}

#[runtime_builtin(
    name = "parpool",
    category = "parallel",
    summary = "Create or return the execution pool for the current session.",
    keywords = "parallel,pool,workers,parpool",
    descriptor(crate::builtins::parallel::pool::PARPOOL_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::pool"
)]
async fn parpool_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::pool::ensure(&active_context("parpool")?, &arguments)
}

#[runtime_builtin(
    name = "gcp",
    category = "parallel",
    summary = "Return the current execution pool.",
    keywords = "parallel,pool,current,gcp,nocreate",
    descriptor(crate::builtins::parallel::pool::GCP_DESCRIPTOR),
    builtin_path = "crate::builtins::parallel::pool"
)]
async fn gcp_builtin(arguments: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::parallel::pool::current(&active_context("gcp")?, &arguments)
}
