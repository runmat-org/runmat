use crate::*;

const IMPORTDATA_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "headerlinesIn",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonnegative scalar header count accepts single, double, and all eight integer classes; logical is excluded and typed values are parsed exactly before file access.",
    }];
pub const IMPORTDATA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[A,delimiterOut,headerlinesOut] = importdata(filename,delimiterIn,integer_headerlinesIn)",
        inputs: &IMPORTDATA_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Header count is structural and does not determine imported payload class. Text numeric data is double; helper-format payloads retain their own documented classes. This implementation remains a host text-import boundary.",
    }];

const IMPORTDATA_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Imported numeric matrix or import structure.",
}];
const IMPORTDATA_ALL_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Imported numeric matrix or import structure.",
    },
    BuiltinParamDescriptor {
        name: "delimiterOut",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Detected or requested text delimiter.",
    },
    BuiltinParamDescriptor {
        name: "headerlinesOut",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of imported header lines.",
    },
];
const IMPORTDATA_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File to import.",
}];
const IMPORTDATA_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File to import.",
    },
    BuiltinParamDescriptor {
        name: "delimiterIn",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Delimiter to use for text files.",
    },
];
const IMPORTDATA_INPUTS_DELIMITER_HEADER: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File to import.",
    },
    BuiltinParamDescriptor {
        name: "delimiterIn",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Delimiter to use for text files.",
    },
    BuiltinParamDescriptor {
        name: "headerlinesIn",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of header lines to skip.",
    },
];
const IMPORTDATA_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename)",
        inputs: &IMPORTDATA_INPUTS_FILENAME,
        outputs: &IMPORTDATA_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename, delimiterIn)",
        inputs: &IMPORTDATA_INPUTS_DELIMITER,
        outputs: &IMPORTDATA_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "A = importdata(filename, delimiterIn, headerlinesIn)",
        inputs: &IMPORTDATA_INPUTS_DELIMITER_HEADER,
        outputs: &IMPORTDATA_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label:
            "[A, delimiterOut, headerlinesOut] = importdata(filename, delimiterIn, headerlinesIn)",
        inputs: &IMPORTDATA_INPUTS_DELIMITER_HEADER,
        outputs: &IMPORTDATA_ALL_OUTPUTS,
    },
];

pub const IMPORTDATA_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.ARGUMENT",
    identifier: Some("RunMat:importdata:InvalidArgument"),
    when: "Filename, delimiter, or header line arguments are malformed.",
    message: "importdata: invalid argument",
};
pub const IMPORTDATA_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.IO",
    identifier: Some("RunMat:importdata:Io"),
    when: "The input file cannot be read.",
    message: "importdata: unable to read file",
};
pub const IMPORTDATA_ERROR_PARSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMPORTDATA.PARSE",
    identifier: Some("RunMat:importdata:Parse"),
    when: "Text content cannot be imported as supported numeric/header data.",
    message: "importdata: unable to parse text data",
};
const IMPORTDATA_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMPORTDATA_ERROR_ARGUMENT,
    IMPORTDATA_ERROR_IO,
    IMPORTDATA_ERROR_PARSE,
];

pub const IMPORTDATA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMPORTDATA_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMPORTDATA_ERRORS,
};
