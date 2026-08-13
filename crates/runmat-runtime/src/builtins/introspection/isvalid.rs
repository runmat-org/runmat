use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const ISVALID_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when handle/listener is valid.",
}];

const ISVALID_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to inspect.",
}];

const ISVALID_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvalid(value)",
    inputs: &ISVALID_INPUTS,
    outputs: &ISVALID_OUTPUT,
}];

const ISVALID_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVALID.INVALID_INPUT",
    identifier: Some("RunMat:isvalid:InvalidInput"),
    when: "The input is not a handle object or listener.",
    message: "isvalid: input must be a handle object or listener",
};

pub const ISVALID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISVALID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISVALID_ERROR_INVALID_INPUT],
};
pub const ISVALID_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "isvalid is a handle-object method; fundamental integer host or resident values are invalid-domain inputs and reject before payload or provider access.",
};

#[runtime_builtin(
    name = "isvalid",
    category = "introspection",
    summary = "Return true for valid handles and listeners.",
    keywords = "handle,listener,validity,classdef",
    descriptor(crate::builtins::introspection::isvalid::ISVALID_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::isvalid::ISVALID_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::isvalid"
)]
pub async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    if !matches!(v, Value::HandleObject(_) | Value::Listener(_)) {
        return Err(
            crate::build_runtime_error(ISVALID_ERROR_INVALID_INPUT.message)
                .with_builtin("isvalid")
                .with_identifier(
                    ISVALID_ERROR_INVALID_INPUT
                        .identifier
                        .expect("isvalid error identifier"),
                )
                .build()
                .into(),
        );
    }
    crate::isvalid_builtin(v).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn all_integer_classes_reject_as_invalid_handle_inputs() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            let error = block_on(isvalid_builtin(Value::Int(value)))
                .expect_err("integer is not a handle input");
            assert_eq!(error.identifier(), ISVALID_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn resident_integer_rejects_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let error = block_on(isvalid_builtin(Value::GpuTensor(handle)))
                .expect_err("resident integer is not a handle input");
            assert_eq!(error.identifier(), ISVALID_ERROR_INVALID_INPUT.identifier);
        });
    }
}
