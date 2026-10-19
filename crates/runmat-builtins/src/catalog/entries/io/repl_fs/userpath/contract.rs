use crate::*;

const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "path_or_command",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One character row or string scalar: a path, reset, or clear.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The exact stored spelling, or the resolved default path, as a character row.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "path = userpath()",
        inputs: &[],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "path = userpath(path_or_command)",
        inputs: &[PATH],
        outputs: &[RESULT],
    },
];

const ERRORS: &[BuiltinErrorDescriptor] = &[
    BuiltinErrorDescriptor { code: "RM.USERPATH.ARITY", identifier: Some("RunMat:userpath:InvalidCall"), when: "More than one input is passed.", message: "userpath: too many input arguments" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.OUTPUTS", identifier: Some("RunMat:userpath:TooManyOutputs"), when: "More than one output is requested.", message: "userpath: too many output arguments" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.PATH", identifier: Some("RunMat:userpath:InvalidPath"), when: "The argument is not host scalar text or is not one path entry.", message: "userpath: invalid path or command" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.STORAGE", identifier: Some("RunMat:userpath:Storage"), when: "The application-owned setting cannot be read or committed.", message: "userpath: application setting failed" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.AUTHORITY", identifier: Some("RunMat:userpath:FilesystemAuthority"), when: "The captured path namespace is unavailable or revoked.", message: "userpath: filesystem authority failed" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.COMMITTED_LOCAL", identifier: Some("RunMat:userpath:CommittedButLocalPathUnavailable"), when: "The application setting was committed, but the captured local path authority was lost before publication.", message: "userpath: committed application setting cannot be published locally" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.STALE_REVISION", identifier: Some("RunMat:userpath:StaleRevision"), when: "A storage read returns an older revision than the session has observed.", message: "userpath: stale application setting revision" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.UNAVAILABLE", identifier: Some("RunMat:RuntimeContext:CapabilityUnavailable"), when: "The host has not installed application settings or session search-path services.", message: "userpath: application settings service is unavailable" },
    BuiltinErrorDescriptor { code: "RM.USERPATH.CANCELLED", identifier: Some("RunMat:Cancelled"), when: "The application settings operation is cancelled before publication.", message: "userpath: operation cancelled" },
];

pub const USERPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const USERPATH_PARTIAL_PATH_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "userpath-partial-path",
        mode: BuiltinExtensionMode::AvailableSince(MatlabRelease {
            year: 2026,
            half: MatlabReleaseHalf::B,
        }),
        description: "userpath partial paths require R2026b or RunMat extension mode",
        error_identifier: Some("RunMat:compatibility:UserpathPartialPathExtension"),
    };
pub const USERPATH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "A user path is host text. Integer and provider-resident numeric values are rejected before application storage or filesystem effects.",
};
