use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "listener",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Registered listener value.",
}];
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Live handle-object event source.",
    },
    BuiltinParamDescriptor {
        name: "event_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Case-sensitive event name.",
    },
    BuiltinParamDescriptor {
        name: "callback",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Function handle or textual callback reference.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "listener = addlistener(target, event_name, callback)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "addlistener(target, event_name, callback)",
        inputs: &INPUTS,
        outputs: &[],
    },
];
pub const ADDLISTENER_ERROR_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.TARGET_INVALID",
    identifier: Some("RunMat:AddListenerTargetInvalid"),
    when: "Target is not a live handle object.",
    message: "addlistener: target must be a valid handle object",
};
pub const ADDLISTENER_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:addlistener:TooManyOutputs"),
    when: "More than one output is explicitly requested.",
    message: "addlistener: expected at most one output",
};
pub const ADDLISTENER_ERROR_ALLOCATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.ALLOCATION_FAILED",
    identifier: Some("RunMat:AddListenerAllocationFailed"),
    when: "GC rooting or listener allocation fails before registration.",
    message: "addlistener: failed to retain listener target or callback",
};
pub const ADDLISTENER_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.CALLBACK_INVALID",
    identifier: Some("RunMat:AddListenerCallbackInvalid"),
    when: "Callback is not a supported nonempty callable or scalar text handle.",
    message: "addlistener: callback must be callable or scalar text",
};
pub const ADDLISTENER_ERROR_EVENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.EVENT_UNKNOWN",
    identifier: Some("RunMat:AddListenerEventUnknown"),
    when: "The target class or its superclasses do not declare the named event.",
    message: "addlistener: unknown event",
};
pub const ADDLISTENER_ERROR_EVENT_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.EVENT_ACCESS",
    identifier: Some("RunMat:AddListenerEventAccess"),
    when: "The active class context lacks declared ListenAccess.",
    message: "addlistener: event is not accessible",
};
pub const ADDLISTENER_ERROR_EVENT_CHANGED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.EVENT_CHANGED",
    identifier: Some("RunMat:AddListenerEventChanged"),
    when: "The event declaration changes before listener publication.",
    message: "addlistener: event changed during registration",
};
pub const ADDLISTENER_ERRORS: [BuiltinErrorDescriptor; 7] = [
    ADDLISTENER_ERROR_TARGET,
    ADDLISTENER_ERROR_OUTPUTS,
    ADDLISTENER_ERROR_ALLOCATION,
    ADDLISTENER_ERROR_CALLBACK,
    ADDLISTENER_ERROR_EVENT_UNKNOWN,
    ADDLISTENER_ERROR_EVENT_ACCESS,
    ADDLISTENER_ERROR_EVENT_CHANGED,
];
pub const ADDLISTENER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDLISTENER_ERRORS,
};
pub const ADDLISTENER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Target is a live handle object, event name is text, callback is callable/text; numeric values have no direct role. Captured callback values may include exact integer or resident payloads without addlistener inspecting them.",
};
