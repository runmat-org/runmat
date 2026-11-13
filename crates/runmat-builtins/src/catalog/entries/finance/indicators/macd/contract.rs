use crate::*;

pub const MACD_NONDOUBLE_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "macd-nondouble-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "raw non-double macd matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:MacdNondoubleMatrixExtension"),
    };

pub const MACD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MACD_NONDOUBLE_MATRIX_EXTENSION];

const MACD_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Data matrix",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents raw matrix Data as double; RunMat mode admits native integer M-by-4 matrices only when every price is exactly representable at the binary64 EMA boundary.",
    }];

const MACD_INTEGER_TABLE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "High/Low/Open/Close table variables",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented table and timetable containers impose names and vector shapes without restricting the numeric storage class of their price variables.",
    }];

pub const MACD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[MACDLine, SignalLine] = macd(integer_Data_matrix)",
        inputs: &MACD_INTEGER_MATRIX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The raw-matrix extension is gated before provider access, reads authoritative integer storage, rejects lossy binary64 conversion, and computes host-double column outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[MACDLine, SignalLine] = macd(table_with_integer_prices)",
        inputs: &MACD_INTEGER_TABLE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer price variables remain exact until a checked binary64 EMA boundary; outputs preserve the table or timetable container and use double Close data.",
    },
];

const PARAM_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Price data as an M-by-4 matrix, table, or timetable.",
};

const OUTPUT_MACD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "MACDLine",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Moving Average Convergence/Divergence series.",
};

const OUTPUT_SIGNAL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SignalLine",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Nine-period exponential moving average of the MACD line.",
};

const INPUTS: [BuiltinParamDescriptor; 1] = [PARAM_DATA];
const OUTPUTS_MACD: [BuiltinParamDescriptor; 1] = [OUTPUT_MACD];
const OUTPUTS_BOTH: [BuiltinParamDescriptor; 2] = [OUTPUT_MACD, OUTPUT_SIGNAL];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "MACDLine = macd(Data)",
        inputs: &INPUTS,
        outputs: &OUTPUTS_MACD,
    },
    BuiltinSignatureDescriptor {
        label: "[MACDLine, SignalLine] = macd(Data)",
        inputs: &INPUTS,
        outputs: &OUTPUTS_BOTH,
    },
];

pub const MACD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
        code: "RM.macd.INVALID_INPUT",
        identifier: Some("RunMat:macd:InvalidInput"),
        when: "Input is not a numeric M-by-4 matrix or table/timetable with High, Low, Open, and Close variables.",
        message: "macd: invalid input",
    };
pub const MACD_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.macd.OUTPUT_COUNT",
    identifier: Some("RunMat:macd:OutputCount"),
    when: "More than two output arguments are requested.",
    message: "macd: too many output arguments",
};
pub const MACD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.macd.INTERNAL",
    identifier: Some("RunMat:macd:Internal"),
    when: "Output construction fails.",
    message: "macd: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    MACD_ERROR_INVALID_INPUT,
    MACD_ERROR_OUTPUT_COUNT,
    MACD_ERROR_INTERNAL,
];

pub const MACD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
