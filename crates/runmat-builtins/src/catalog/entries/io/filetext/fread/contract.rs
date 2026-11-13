use crate::*;
pub const FREAD_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the fread \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadLikeExtension"),
};
pub const FREAD_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fread file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerIdExtension"),
};
pub const FREAD_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer fread size controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerSizeExtension"),
};
pub const FREAD_INTEGER_SKIP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-integer-skip",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer fread skip controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadIntegerSkipExtension"),
};
pub const FREAD_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fread-logical-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "logical fread file identifiers, sizes, and skips are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FreadLogicalControlExtension"),
    };
pub const FREAD_SINGLE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fread-single-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fread file identifiers, sizes, and skips are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FreadSingleControlExtension"),
};
pub const FREAD_RESIDENT_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fread-resident-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "provider-resident fread control arguments are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FreadResidentControlExtension"),
    };

pub const FREAD_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    FREAD_LIKE_EXTENSION,
    FREAD_INTEGER_ID_EXTENSION,
    FREAD_INTEGER_SIZE_EXTENSION,
    FREAD_INTEGER_SKIP_EXTENSION,
    FREAD_LOGICAL_CONTROL_EXTENSION,
    FREAD_SINGLE_CONTROL_EXTENSION,
    FREAD_RESIDENT_CONTROL_EXTENSION,
];

mod integer;
pub use integer::*;

const FREAD_OUTPUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Read data as numeric tensor or character array depending on precision.",
}];
const FREAD_OUTPUT_DATA_COUNT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Read data as numeric tensor or character array depending on precision.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Number of source elements actually read, excluding padding.",
    },
];
const FREAD_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FREAD_INPUTS_FID_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8=>double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "size",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"inf\""),
        description: "Element count or size vector ([m n]); supports \"inf\".",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label (for example \"double\", \"uint8\", \"*char\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
];
const FREAD_INPUTS_FID_PRECISION_MACHINEFMT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Read precision label when size is omitted.",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element read.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FREAD_INPUTS_WITH_LIKE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "arg",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Positional fread arguments before the like clause.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: Some("\"like\""),
        description: "Prototype keyword; currently only 'like'.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Prototype value controlling output class/residency.",
    },
];
const FREAD_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "data = fread(fid)",
        inputs: &FREAD_INPUTS_FID,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size)",
        inputs: &FREAD_INPUTS_FID_SIZE,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, size, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_SIZE_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision)",
        inputs: &FREAD_INPUTS_FID_PRECISION,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, precision, skip, machinefmt)",
        inputs: &FREAD_INPUTS_FID_PRECISION_SKIP_MACHINEFMT,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = fread(fid, ..., \"like\", prototype)",
        inputs: &FREAD_INPUTS_WITH_LIKE,
        outputs: &FREAD_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "[data, count] = fread(...)",
        inputs: &FREAD_INPUTS_FID,
        outputs: &FREAD_OUTPUT_DATA_COUNT,
    },
];

pub const FREAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_INPUT",
    identifier: Some("RunMat:fread:InvalidInput"),
    when: "Identifier/argument cardinality/type constraints are violated.",
    message: "fread: invalid input arguments",
};
pub const FREAD_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fread:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fread: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FREAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INVALID_OPTION",
    identifier: Some("RunMat:fread:InvalidOption"),
    when: "Precision, skip, machine format, or like option values are invalid.",
    message: "fread: invalid option configuration",
};
pub const FREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.IO",
    identifier: Some("RunMat:fread:IoFailure"),
    when: "Read/seek or data-shape materialization fails.",
    message: "fread: file read failed",
};
pub const FREAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREAD.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow conversion failed.",
    message: "fread: internal error",
};
const FREAD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FREAD_ERROR_INVALID_INPUT,
    FREAD_ERROR_INVALID_IDENTIFIER,
    FREAD_ERROR_INVALID_OPTION,
    FREAD_ERROR_IO,
    FREAD_ERROR_INTERNAL,
];
pub const FREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREAD_ERRORS,
};
