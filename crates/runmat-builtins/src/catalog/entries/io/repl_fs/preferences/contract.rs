use crate::*;

const GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Preference group name as one character row or string scalar.",
};
const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pref",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One name or a shape-preserving string/cell array of names.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A default or stored value; multiple names require a matching cell array.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stored value or a cell array matching the preference-name shape.",
};
const EXISTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar logical or logical array matching the name shape.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat extension: zero after a successful update.",
};
const GET_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "groups = getpref()",
        inputs: &[],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "value = getpref(group, pref)",
        inputs: &[GROUP, NAME],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "value = getpref(group, pref, default)",
        inputs: &[GROUP, NAME, VALUE],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "group_values = getpref(group)",
        inputs: &[GROUP],
        outputs: &[RESULT],
    },
];
const IS_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "tf = ispref(group)",
        inputs: &[GROUP],
        outputs: &[EXISTS],
    },
    BuiltinSignatureDescriptor {
        label: "tf = ispref(group, pref)",
        inputs: &[GROUP, NAME],
        outputs: &[EXISTS],
    },
];
const SET_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "setpref(group, pref, value)",
        inputs: &[GROUP, NAME, VALUE],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = setpref(group, pref, value)",
        inputs: &[GROUP, NAME, VALUE],
        outputs: &[STATUS],
    },
];

pub const GETPREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: GET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        GETPREF_ERROR_TOO_MANY_OUTPUTS,
        GETPREF_ERROR_MISSING,
        GETPREF_ERROR_SERVICE_UNAVAILABLE,
        GETPREF_ERROR_STORAGE,
    ],
};
pub const ISPREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: IS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ISPREF_ERROR_TOO_MANY_OUTPUTS,
        ISPREF_ERROR_SERVICE_UNAVAILABLE,
        ISPREF_ERROR_STORAGE,
    ],
};
pub const SETPREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        SETPREF_ERROR_TOO_MANY_OUTPUTS,
        SETPREF_ERROR_SERVICE_UNAVAILABLE,
        SETPREF_ERROR_STORAGE,
    ],
};
pub const GETPREF_ERROR_SERVICE_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETPREF.SERVICE_UNAVAILABLE",
    identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
    when: "The host has not installed an application preference service.",
    message: "getpref: application preference service is unavailable",
};
pub const GETPREF_ERROR_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETPREF.MISSING",
    identifier: None,
    when: "A named preference does not exist and no default was supplied.",
    message: "getpref: preference '<group>/<name>' does not exist",
};
pub const ISPREF_ERROR_SERVICE_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISPREF.SERVICE_UNAVAILABLE",
    identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
    when: "The host has not installed an application preference service.",
    message: "ispref: application preference service is unavailable",
};
pub const SETPREF_ERROR_SERVICE_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETPREF.SERVICE_UNAVAILABLE",
    identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
    when: "The host has not installed an application preference service.",
    message: "setpref: application preference service is unavailable",
};
pub const GETPREF_ERROR_STORAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETPREF.STORAGE",
    identifier: Some("RunMat:getpref:Storage"),
    when: "The application setting cannot be read, created, decoded, or committed.",
    message: "getpref: application preferences failed",
};
pub const ISPREF_ERROR_STORAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISPREF.STORAGE",
    identifier: Some("RunMat:ispref:Storage"),
    when: "The application setting cannot be read or decoded.",
    message: "ispref: application preferences failed",
};
pub const SETPREF_ERROR_STORAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETPREF.STORAGE",
    identifier: Some("RunMat:setpref:Storage"),
    when: "The application setting cannot be encoded or committed.",
    message: "setpref: application preferences failed",
};
pub const GETPREF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETPREF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:getpref:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "getpref: too many output arguments",
};
pub const ISPREF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISPREF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ispref:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "ispref: too many output arguments",
};
pub const SETPREF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SETPREF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:setpref:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "setpref: too many output arguments",
};
pub const GETPREF_GROUP_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getpref-group-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getpref(group) group-structure query is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetprefGroupQueryExtension"),
};
pub(super) const GET_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[GETPREF_GROUP_QUERY_EXTENSION];
pub const SETPREF_STATUS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "setpref-status-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A numeric status output from setpref is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SetprefStatusOutputExtension"),
};
pub(super) const SET_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[SETPREF_STATUS_EXTENSION];

const INTEGER_VALUE: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "value",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "All eight integer classes preserve exact class, shape, and payload as preference values.",
}];
pub const GETPREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "value = getpref(group, pref, integer_default)",
        inputs: INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A missing preference retains its exact integer default; an existing value wins unchanged.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "integer_value = getpref(group, pref)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A stored integer retains its exact class; multiple names produce a cell array.",
    },
];
pub const SETPREF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "setpref(group, pref, integer_value)",
        inputs: INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Resident integer values are gathered after control admission without float conversion.",
    },
];
pub const ISPREF_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "Group and preference-name controls are text; integers are invalid before provider access.",
};
