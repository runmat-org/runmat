use crate::*;

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const IN_TEXT_BOUNDARY_REPLACEMENT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input string array, character vector, or cell array of character vectors.",
    },
    BuiltinParamDescriptor {
        name: "patOrPos",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or pattern marker, or positive one-based numeric position.",
    },
    BuiltinParamDescriptor {
        name: "newText",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text to insert; scalar or the same size as str.",
    },
];

pub const INSERT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.INSERT.INVALID_INPUT",
        identifier: Some("RunMat:insert:InvalidInput"),
        when: "The call does not contain exactly one text input, one text or numeric boundary, and one replacement-text input.",
        message: "text insertion: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.INSERT.INVALID_BOUNDARY",
        identifier: Some("RunMat:insert:InvalidBoundary"),
        when: "A numeric position is not a positive integer or a textual boundary is invalid or absent.",
        message: "text insertion: invalid boundary",
    },
    BuiltinErrorDescriptor {
        code: "RM.INSERT.SIZE_MISMATCH",
        identifier: Some("RunMat:insert:SizeMismatch"),
        when: "A nonscalar boundary or replacement-text array does not have the same shape as the input text array.",
        message: "text insertion: array size must match input text",
    },
];

pub const INSERT_BEFORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newStr = insertBefore(str, patOrPos, newText)",
        inputs: &IN_TEXT_BOUNDARY_REPLACEMENT,
        outputs: &OUT_ANY,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INSERT_ERRORS,
};

const INSERT_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "pos",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The numeric position is a one-based structural control and accepts scalar or same-size arrays from every built-in integer class, single, or double; logical is not numeric.",
    }];
const INSERT_TEXT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "str",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The source is text; integer data is rejected before provider access.",
    },
    BuiltinIntegerInputCapability {
        name: "newText",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The inserted value is text; integer data is rejected before provider access.",
    },
];
const fn insert_position_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &INSERT_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions are read exactly and output preserves the source text container class and shape.",
    }
}
const fn insert_text_capability(form: &'static str) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &INSERT_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer source and replacement inputs are outside the public text domain and reject without numeric-to-text conversion.",
    }
}
pub const INSERT_BEFORE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    insert_position_capability("newStr = insertBefore(str, integer_pos, newText)"),
    insert_text_capability("newStr = insertBefore(integer_str, boundary, integer_newText)"),
];

pub const INSERT_BEFORE_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor = insert_extension(
    "insertbefore-resident-position",
    "insertBefore with an interactive resident numeric position is a RunMat extension",
    "RunMat:compatibility:InsertBeforeResidentPositionExtension",
);
pub const INSERT_BEFORE_BROAD_TEXT_EXTENSION: BuiltinExtensionDescriptor = insert_extension(
    "insertbefore-broad-text-containers",
    "insertBefore row-wise character matrices and non-cellstr text cells are RunMat extensions",
    "RunMat:compatibility:InsertBeforeBroadTextExtension",
);
const fn insert_extension(
    id: &'static str,
    description: &'static str,
    error_identifier: &'static str,
) -> BuiltinExtensionDescriptor {
    BuiltinExtensionDescriptor {
        id,
        mode: BuiltinExtensionMode::RunMatOnly,
        description,
        error_identifier: Some(error_identifier),
    }
}
pub const INSERT_BEFORE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    INSERT_BEFORE_RESIDENT_POSITION_EXTENSION,
    INSERT_BEFORE_BROAD_TEXT_EXTENSION,
];
