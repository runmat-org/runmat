use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat zero status on successful dispatch.",
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
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Event-data object or RunMat callback arguments forwarded unchanged.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "status = notify(target, event_name, varargin)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "notify(target, event_name, varargin)",
        inputs: &INPUTS,
        outputs: &[],
    },
];
pub const NOTIFY_ERROR_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.TARGET_INVALID",
    identifier: Some("RunMat:NotifyTargetInvalid"),
    when: "Target is not a live handle object.",
    message: "notify: target must be a valid handle object",
};
pub const NOTIFY_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.CALLBACK_INVALID",
    identifier: Some("RunMat:NotifyInvalidCallback"),
    when: "A registered callback handle cannot be retrieved.",
    message: "notify: invalid listener callback handle",
};
pub const NOTIFY_ERROR_ARGUMENT_LIFETIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.ARGUMENT_LIFETIME",
    identifier: Some("RunMat:NotifyArgumentLifetime"),
    when: "Event data or an extra callback argument cannot be retained through dispatch.",
    message: "notify: failed to retain callback arguments",
};
pub const NOTIFY_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:notify:TooManyOutputs"),
    when: "More than one output is explicitly requested.",
    message: "notify: expected at most one output",
};
pub const NOTIFY_ERROR_EVENT_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.EVENT_UNKNOWN",
    identifier: Some("RunMat:NotifyEventUnknown"),
    when: "The target class or its superclasses do not declare the named event.",
    message: "notify: unknown event",
};
pub const NOTIFY_ERROR_EVENT_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.EVENT_ACCESS",
    identifier: Some("RunMat:NotifyEventAccess"),
    when: "The active class context lacks declared NotifyAccess.",
    message: "notify: event is not accessible",
};
pub const NOTIFY_ERROR_EVENT_CHANGED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.EVENT_CHANGED",
    identifier: Some("RunMat:NotifyEventChanged"),
    when: "The event declaration changes during callback dispatch.",
    message: "notify: event changed during dispatch",
};
pub const NOTIFY_ERROR_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NOTIFY.DATA_INVALID",
    identifier: Some("RunMat:NotifyDataInvalid"),
    when: "A supplied standard event payload is not a valid event.EventData handle object.",
    message: "notify: data must be a valid event.EventData object",
};
pub const NOTIFY_ERRORS: [BuiltinErrorDescriptor; 8] = [
    NOTIFY_ERROR_TARGET,
    NOTIFY_ERROR_CALLBACK,
    NOTIFY_ERROR_ARGUMENT_LIFETIME,
    NOTIFY_ERROR_OUTPUTS,
    NOTIFY_ERROR_EVENT_UNKNOWN,
    NOTIFY_ERROR_EVENT_ACCESS,
    NOTIFY_ERROR_EVENT_CHANGED,
    NOTIFY_ERROR_DATA,
];
pub const NOTIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NOTIFY_ERRORS,
};
pub const NOTIFY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Target and event name are handle/text, not numeric controls. Nested integer or resident values in event data and RunMat extra callback arguments are forwarded unchanged to callbacks.",
};
pub const NOTIFY_STATUS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "notify-status-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "notify's returned zero status is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NotifyStatusExtension"),
};
pub const NOTIFY_EXTRA_ARGS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "notify-extra-callback-arguments",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "notify forwarding multiple callback data arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NotifyExtraArgumentsExtension"),
};
pub const NOTIFY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [NOTIFY_STATUS_EXTENSION, NOTIFY_EXTRA_ARGS_EXTENSION];
