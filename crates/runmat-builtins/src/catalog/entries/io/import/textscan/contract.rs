use crate::*;

const TEXTSCAN_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array containing one output per conversion, or collected groups.",
}];
const TEXTSCAN_INPUTS_TEXT_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "textOrFileID",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or file identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format specification such as '%f %s'.",
    },
];
const TEXTSCAN_INPUTS_TEXT_FORMAT_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "textOrFileID",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or file identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format specification such as '%f %s'.",
    },
    BuiltinParamDescriptor {
        name: "args...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional repeat count followed by name-value pairs.",
    },
];
const TEXTSCAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = textscan(textOrFileID, formatSpec)",
        inputs: &TEXTSCAN_INPUTS_TEXT_FORMAT,
        outputs: &TEXTSCAN_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = textscan(textOrFileID, formatSpec, args...)",
        inputs: &TEXTSCAN_INPUTS_TEXT_FORMAT_OPTIONS,
        outputs: &TEXTSCAN_OUTPUTS,
    },
];

pub const TEXTSCAN_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.ARGUMENT",
    identifier: Some("RunMat:textscan:InvalidArgument"),
    when: "Input, format specification, repeat count, or name-value options are malformed.",
    message: "textscan: invalid argument",
};
pub const TEXTSCAN_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.FORMAT",
    identifier: Some("RunMat:textscan:InvalidFormat"),
    when: "Format specification cannot be parsed or contains unsupported conversions.",
    message: "textscan: invalid format specification",
};
pub const TEXTSCAN_ERROR_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.FILE",
    identifier: Some("RunMat:textscan:File"),
    when: "A file identifier is invalid or cannot be read.",
    message: "textscan: file read failed",
};
pub const TEXTSCAN_ERROR_PARSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCAN.PARSE",
    identifier: Some("RunMat:textscan:Parse"),
    when: "Input text cannot be parsed according to the format specification.",
    message: "textscan: parse failed",
};
const TEXTSCAN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    TEXTSCAN_ERROR_ARGUMENT,
    TEXTSCAN_ERROR_FORMAT,
    TEXTSCAN_ERROR_FILE,
    TEXTSCAN_ERROR_PARSE,
];

pub const TEXTSCAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEXTSCAN_ERRORS,
};

pub const TEXTSCAN_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "textscan-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "textscan accepts typed-integer file identifiers, repeat counts, and numeric controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextscanTypedIntegerControlExtension"),
    };
pub const TEXTSCAN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [TEXTSCAN_TYPED_INTEGER_CONTROL_EXTENSION];
const TEXTSCAN_INTEGER_FORMAT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "%d/%i/%u/%x/%b integer conversion specifier",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The format text selects int8/int16/int32/int64 or uint8/uint16/uint32/uint64 through documented decimal, hexadecimal, and binary suffixes; unsuffixed decimal signed/unsigned conversions return int32/uint32, while unsuffixed hexadecimal and binary conversions return uint64.",
    }];
const TEXTSCAN_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "file identifier, repeat count, HeaderLines, or boolean numeric control",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public forms expose double file identifiers and numeric controls. RunMat mode admits all eight integer classes, validates authoritative scalars structurally, and rejects negative or oversized values without floating conversion.",
    }];
pub const TEXTSCAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = textscan(textOrFileID, integer_conversion_format, ___)",
        inputs: &TEXTSCAN_INTEGER_FORMAT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer fields retain their documented classes and exact values. CollectOutput combines only adjacent numeric columns of the same class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = textscan(typed_integer_file_id, formatSpec, typed_integer_controls...)",
        inputs: &TEXTSCAN_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Compatibility mode rejects the typed-control extension before gather or file access; RunMat mode preserves the pre-existing exact structural behavior. Automatic residency remains a transparent host fallback.",
    },
];
