use crate::*;

const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The restored ordered search path as a character row.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "restoredefaultpath()",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "path = restoredefaultpath()",
        inputs: &[],
        outputs: &[PATH],
    },
];
const ERRORS: &[BuiltinErrorDescriptor] = &[
    BuiltinErrorDescriptor {
        code: "RM.RESTOREDEFAULTPATH.ARITY",
        identifier: Some("RunMat:restoredefaultpath:InvalidCall"),
        when: "An input is supplied.",
        message: "restoredefaultpath: too many input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.RESTOREDEFAULTPATH.OUTPUTS",
        identifier: Some("RunMat:restoredefaultpath:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "restoredefaultpath: too many output arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.RESTOREDEFAULTPATH.STALE_USER",
        identifier: Some("RunMat:restoredefaultpath:StaleUserPath"),
        when: "The application user-path setting must be rehydrated before restoration.",
        message: "restoredefaultpath: user path is stale",
    },
    BuiltinErrorDescriptor {
        code: "RM.RESTOREDEFAULTPATH.UNAVAILABLE",
        identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"),
        when: "The host has no session search-path service.",
        message: "restoredefaultpath: ordered search-path service is unavailable",
    },
];
pub const RESTOREDEFAULTPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const RESTOREDEFAULTPATH_RETURN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "restoredefaultpath-return",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "restoredefaultpath returning a path is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RestoredefaultpathReturnExtension"),
    };
pub const RESTOREDEFAULTPATH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "restoredefaultpath accepts no inputs and performs no integer computation.",
    };
