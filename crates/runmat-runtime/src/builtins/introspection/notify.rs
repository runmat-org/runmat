use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

pub const NOTIFY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The documented handle.notify surface accepts a handle event source, a textual event name, and optionally an event.EventData object. Integer values have no direct data, control, output-class, or backend role; application fields nested inside an event-data object remain ordinary object payloads.",
};

const NOTIFY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Always zero on success.",
}];

const NOTIFY_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target object or handle.",
    },
    BuiltinParamDescriptor {
        name: "event_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Event name.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Event callback arguments.",
    },
];

const NOTIFY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = notify(target, event_name, varargin)",
    inputs: &NOTIFY_INPUTS,
    outputs: &NOTIFY_OUTPUT,
}];

const NOTIFY_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.NOTIFY.TARGET_INVALID",
    identifier: Some("RunMat:NotifyTargetInvalid"),
    when: "Target is not an object or handle object.",
    message: "notify: target must be handle or object",
}];

pub const NOTIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NOTIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NOTIFY_ERRORS,
};

#[runtime_builtin(
    name = "notify",
    category = "introspection",
    summary = "Dispatch event notifications to registered listeners.",
    keywords = "events,listener,notify,callback,classdef",
    descriptor(crate::builtins::introspection::notify::NOTIFY_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::notify::NOTIFY_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::notify"
)]
pub async fn notify_builtin(
    target: Value,
    event_name: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    crate::notify_builtin(target, event_name, rest).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_is_integer_inapplicable() {
        assert_eq!(
            NOTIFY_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(NOTIFY_INTEGER_AUDIT.canonical_builtin.is_none());
    }
}
