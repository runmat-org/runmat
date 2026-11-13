use crate::*;

pub const STRIP_DIRECTION_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "strip-direction-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "The leading and trailing direction aliases are RunMat extensions to left and right",
        error_identifier: Some("RunMat:compatibility:StripDirectionAliasExtension"),
    };

pub const STRIP_CHARACTER_SET_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "strip-character-set",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "Multi-character sets and per-element strip-character controls are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:StripCharacterSetExtension"),
};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stripped text preserving input container kind and shape.",
}];
const fn base() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String, character, or cell text input.",
    }
}
const fn direction() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"both\""),
        description:
            "Side to strip: left, right, or both; RunMat also accepts leading and trailing.",
    }
}
const fn characters() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "stripCharacters",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar or shape-matched characters to remove.",
    }
}
const BASE: [BuiltinParamDescriptor; 1] = [base()];
const WITH_DIRECTION: [BuiltinParamDescriptor; 2] = [base(), direction()];
const WITH_CHARACTERS: [BuiltinParamDescriptor; 2] = [base(), characters()];
const WITH_BOTH: [BuiltinParamDescriptor; 3] = [base(), direction(), characters()];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "out = strip(str)",
        inputs: &BASE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "out = strip(str, direction)",
        inputs: &WITH_DIRECTION,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "out = strip(str, stripCharacters)",
        inputs: &WITH_CHARACTERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "out = strip(str, direction, stripCharacters)",
        inputs: &WITH_BOTH,
        outputs: &OUTPUTS,
    },
];

pub const STRIP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.INVALID_INPUT", identifier: Some("RunMat:strip:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: "strip: first argument must be a string array, character array, or cell array of character vectors",
};
pub const STRIP_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.CELL_ELEMENT",
    identifier: Some("RunMat:strip:CellElement"),
    when: "Cell array contains a non-text or non-row character element.",
    message: "strip: cell array elements must be string scalars or character vectors",
};
pub const STRIP_ERROR_DIRECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.DIRECTION",
    identifier: Some("RunMat:strip:InvalidDirection"),
    when: "Direction is not left/right/both or an accepted synonym.",
    message: "strip: direction must be 'left', 'right', or 'both'",
};
pub const STRIP_ERROR_CHARACTERS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.CHARACTERS", identifier: Some("RunMat:strip:InvalidCharacters"),
    when: "stripCharacters is not a valid text container.",
    message: "strip: characters to remove must be a string array, character vector, or cell array of character vectors",
};
pub const STRIP_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.SIZE_MISMATCH",
    identifier: Some("RunMat:strip:SizeMismatch"),
    when: "Per-element stripCharacters does not match input shape and size.",
    message:
        "strip: stripCharacters must be the same size as the input when supplying multiple values",
};
pub const STRIP_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.ARG_COUNT",
    identifier: Some("RunMat:strip:ArgCount"),
    when: "More than three inputs were supplied.",
    message: "strip: too many input arguments",
};
pub const STRIP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRIP.INTERNAL",
    identifier: Some("RunMat:strip:InternalError"),
    when: "Output container construction failed.",
    message: "strip: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    STRIP_ERROR_INVALID_INPUT,
    STRIP_ERROR_CELL_ELEMENT,
    STRIP_ERROR_DIRECTION,
    STRIP_ERROR_CHARACTERS,
    STRIP_ERROR_SIZE_MISMATCH,
    STRIP_ERROR_ARG_COUNT,
    STRIP_ERROR_INTERNAL,
];
pub const STRIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const STRIP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "strip accepts host string, character and cell text, plus textual direction and character controls. Numeric, logical, symbolic and provider-resident values reject before provider access.",
};
