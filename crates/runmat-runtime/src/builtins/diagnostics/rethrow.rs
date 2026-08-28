use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_value::Value;

const RETHROW_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const RETHROW_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error value to rethrow.",
}];

const RETHROW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "rethrow(err)",
    inputs: &RETHROW_INPUTS,
    outputs: &RETHROW_OUTPUT,
}];

pub const RETHROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RETHROW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

pub const RETHROW_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "rethrow consumes a caught exception object and has no numeric data, control, class-preserving output, or provider role. Numeric, logical, symbolic, and resident values fail directly without gather or provider access.",
};

#[runmat_macros::runtime_builtin(
    name = "rethrow",
    descriptor(self::RETHROW_DESCRIPTOR),
    integer_audit(self::RETHROW_INTEGER_AUDIT),
    builtin_path = "crate::builtins::diagnostics::rethrow"
)]
pub async fn rethrow_builtin_registered(e: Value) -> crate::BuiltinResult<Value> {
    match e {
        Value::MException(_) | Value::String(_) => crate::rethrow_builtin(e).await,
        other => Err(
            crate::build_runtime_error(format!("RunMat:error: {other:?}"))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build(),
        ),
    }
}
