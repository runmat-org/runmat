use crate::*;

const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];

const IN_TEXT_BOUNDARY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "text",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input string array, character vector, or cell array of character vectors.",
    },
    BuiltinParamDescriptor {
        name: "boundary",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Text/pattern boundary or positive numeric position, scalar or the same size as text.",
    },
];

pub const EXTRACT_AFTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.INVALID_INPUT",
        identifier: Some("RunMat:extractAfter:InvalidInput"),
        when: "The call does not contain exactly one documented text input and one boundary.",
        message: "extractAfter: invalid input",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.INVALID_BOUNDARY",
        identifier: Some("RunMat:extractAfter:InvalidBoundary"),
        when: "A numeric boundary is not a positive integer or a text/pattern boundary is invalid.",
        message: "extractAfter: invalid boundary",
    },
    BuiltinErrorDescriptor {
        code: "RM.EXTRACT_AFTER.SIZE_MISMATCH",
        identifier: Some("RunMat:extractAfter:SizeMismatch"),
        when: "A nonscalar boundary does not have the same shape as the text input.",
        message: "extractAfter: boundary size must match text",
    },
];
pub const EXTRACT_AFTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "s = extractAfter(text, boundary)",
        inputs: &IN_TEXT_BOUNDARY,
        outputs: &OUT_ANY,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXTRACT_AFTER_ERRORS,
};

const EXTRACT_BOUNDARY_POSITION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "pos",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The numeric position is a one-based structural control and accepts scalar or same-size arrays from every built-in integer class.",
    }];
const EXTRACT_BOUNDARY_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "str",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The first argument is text; integer data is rejected before provider access.",
    }];
pub const EXTRACT_AFTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    extract_boundary_position_capability("newStr = extractAfter(str, integer_pos)"),
    extract_boundary_text_capability("newStr = extractAfter(integer_str, boundary)"),
];

const fn extract_boundary_position_capability(
    form: &'static str,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &EXTRACT_BOUNDARY_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions are read exactly with one-based indexing; the output preserves the input text container class and shape.",
    }
}

const fn extract_boundary_text_capability(
    form: &'static str,
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs: &EXTRACT_BOUNDARY_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer text input is outside the public text domain and rejects without numeric-to-text conversion.",
    }
}

pub const EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    extract_boundary_extension(
        "extractafter-resident-position",
        "extractAfter with a resident numeric position is a RunMat extension",
        "RunMat:compatibility:ExtractAfterResidentPositionExtension",
    );
pub const EXTRACT_AFTER_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    extract_boundary_extension(
        "extractafter-char-matrix",
        "extractAfter row-wise character-matrix input is a RunMat extension",
        "RunMat:compatibility:ExtractAfterCharMatrixExtension",
    );
pub const EXTRACT_AFTER_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor =
    extract_boundary_extension(
        "extractafter-string-cell",
        "extractAfter cells containing string scalars or nested cells are a RunMat extension",
        "RunMat:compatibility:ExtractAfterStringCellExtension",
    );
pub const EXTRACT_AFTER_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EXTRACT_AFTER_RESIDENT_POSITION_EXTENSION,
    EXTRACT_AFTER_CHAR_MATRIX_EXTENSION,
    EXTRACT_AFTER_STRING_CELL_EXTENSION,
];

const fn extract_boundary_extension(
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
