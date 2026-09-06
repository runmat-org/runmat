use crate::*;

const FIRST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filepart1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First character, string, or cell path component.",
};
const REST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filepartN",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Additional path components with compatible container shapes.",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Joined path preserving the selected text-container representation and shape.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "file = fullfile(filepart1, ..., filepartN)",
    inputs: &[FIRST, REST],
    outputs: &[OUTPUT],
}];

macro_rules! error {
    ($name:ident, $code:literal, $id:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($id),
            when: $when,
            message: $message,
        };
    };
}
error!(
    FULLFILE_ERROR_NOT_ENOUGH_INPUTS,
    "RM.FULLFILE.NOT_ENOUGH_INPUTS",
    "RunMat:fullfile:NotEnoughInputs",
    "No path component is supplied.",
    "fullfile: not enough input arguments"
);
error!(FULLFILE_ERROR_ARGUMENT_TYPE, "RM.FULLFILE.ARGUMENT_TYPE", "RunMat:fullfile:InvalidArgument", "A component is not an admitted text container or RunMat numeric character-code row.", "fullfile: arguments must be character vectors, string arrays, or cell arrays of character vectors");
error!(
    FULLFILE_ERROR_SHAPE,
    "RM.FULLFILE.SHAPE",
    "RunMat:fullfile:IncompatibleSizes",
    "Nonscalar string or cell inputs have different shapes.",
    "fullfile: nonscalar string and cell inputs must have the same size"
);
error!(
    FULLFILE_ERROR_PROVIDER,
    "RM.FULLFILE.PROVIDER",
    "RunMat:fullfile:ProviderFailed",
    "An admitted resident numeric character-code row cannot be gathered.",
    "fullfile: unable to read resident character codes"
);

pub const FULLFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        FULLFILE_ERROR_NOT_ENOUGH_INPUTS,
        FULLFILE_ERROR_ARGUMENT_TYPE,
        FULLFILE_ERROR_SHAPE,
        FULLFILE_ERROR_PROVIDER,
    ],
};

pub const FULLFILE_NUMERIC_CHARACTER_CODES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fullfile-numeric-character-codes",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fullfile with numeric character-code rows is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FullfileNumericCharacterCodesExtension"),
    };

const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "filepart",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A dense real integer row may encode one path component.",
}];
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "file = fullfile(integer_character_codes, ...)",
        inputs: INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Code points decode exactly before lexical path assembly.",
    }];
