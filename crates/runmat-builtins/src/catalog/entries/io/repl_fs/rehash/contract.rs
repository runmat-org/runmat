use crate::*;

/// Closed public rehash modes. The runtime and literal inference share this
/// boundary; cache ownership and refresh behavior remain with the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RehashMode {
    Path,
    Toolbox,
    ToolboxCache,
}

impl RehashMode {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("path") {
            Some(Self::Path)
        } else if text.eq_ignore_ascii_case("toolbox") {
            Some(Self::Toolbox)
        } else if text.eq_ignore_ascii_case("toolboxcache") {
            Some(Self::ToolboxCache)
        } else {
            None
        }
    }
}

const MODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "path, toolbox, or toolboxcache.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat status extension: zero when the source cache refresh completes.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "rehash",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "rehash(mode)",
        inputs: &[MODE],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = rehash(mode)",
        inputs: &[MODE],
        outputs: &[STATUS],
    },
    BuiltinSignatureDescriptor {
        label: "status = rehash()",
        inputs: &[],
        outputs: &[STATUS],
    },
];
pub const REHASH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rehash:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "rehash: too many output arguments",
};
pub const REHASH_ERROR_INVALID_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.INVALID_MODE",
    identifier: Some("RunMat:rehash:InvalidMode"),
    when: "A mode is not scalar host text, or is not path, toolbox, or toolboxcache.",
    message: "rehash: expected path, toolbox, or toolboxcache",
};
pub const REHASH_ERROR_INVALID_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.INVALID_ARITY",
    identifier: Some("RunMat:rehash:InvalidCall"),
    when: "More than one mode argument is supplied.",
    message: "rehash: too many input arguments",
};
pub const REHASH_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.UNAVAILABLE",
    identifier: Some("RunMat:rehash:Unavailable"),
    when: "The runtime has no source-cache refresh service.",
    message: "rehash: no source-cache refresh service is installed",
};
const CANCELLED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REHASH.CANCELLED",
    identifier: Some("RunMat:Cancelled"),
    when: "The invocation is cancelled before or during the source-cache refresh.",
    message: "The source-cache refresh was cancelled.",
};
pub const REHASH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        REHASH_ERROR_TOO_MANY_OUTPUTS,
        REHASH_ERROR_INVALID_MODE,
        REHASH_ERROR_INVALID_ARITY,
        REHASH_ERROR_UNAVAILABLE,
        CANCELLED,
    ],
};
pub const REHASH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "rehash accepts only a host text mode; it does not compute on integer values.",
};
