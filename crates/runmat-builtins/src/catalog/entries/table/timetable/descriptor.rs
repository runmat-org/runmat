use super::super::shared::TABLE_ERRORS;
use crate::*;
const TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output timetable.",
}];
const TIMETABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "rowTimes or timing options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Datetime/duration row times, SampleRate, TimeStep, or preallocation options.",
    },
    BuiltinParamDescriptor {
        name: "variables and Name,Value options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Same-height data variables plus VariableNames, Size, VariableTypes, and timing options.",
    },
];
const TIMETABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "TT = timetable()",
        inputs: &[],
        outputs: &TIMETABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "TT = timetable(rowTimes, variables..., Name=Value...)",
        inputs: &TIMETABLE_INPUTS,
        outputs: &TIMETABLE_OUTPUT,
    },
];
pub const TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const TIMETABLE2TABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output table.",
}];
const TIMETABLE2TABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "TT",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input timetable.",
    },
    BuiltinParamDescriptor {
        name: "ConvertRowTimes",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: Some("true"),
        description:
            "Logical flag that controls whether row times become the first table variable.",
    },
];
const TIMETABLE2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = timetable2table(TT, ConvertRowTimes=value)",
    inputs: &TIMETABLE2TABLE_INPUTS,
    outputs: &TIMETABLE2TABLE_OUTPUT,
}];
pub const TIMETABLE2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMETABLE2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const TABLE2TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output timetable.",
}];
const TABLE2TIMETABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "RowTimes, SampleRate, TimeStep, and StartTime options.",
    },
];
const TABLE2TIMETABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TT = table2timetable(T, Name=Value...)",
    inputs: &TABLE2TIMETABLE_INPUTS,
    outputs: &TABLE2TIMETABLE_OUTPUT,
}];
pub const TABLE2TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE2TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const ARRAY2TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timetable whose variables are the columns of X.",
}];
const ARRAY2TIMETABLE_INPUTS_ROW_TIMES: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "RowTimes",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the row-time option.",
    },
    BuiltinParamDescriptor {
        name: "rowTimes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime or duration column vector labeling the rows.",
    },
];
const ARRAY2TIMETABLE_INPUTS_SAMPLE_RATE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "SampleRate",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the sample-rate option.",
    },
    BuiltinParamDescriptor {
        name: "Fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive sample rate in samples per second.",
    },
];
const ARRAY2TIMETABLE_INPUTS_TIME_STEP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "TimeStep",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the time-step option.",
    },
    BuiltinParamDescriptor {
        name: "dt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive duration or calendarDuration scalar time step.",
    },
];
const ARRAY2TIMETABLE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Timing, StartTime, VariableNames, or DimensionNames options.",
    },
];
const ARRAY2TIMETABLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "TT = array2timetable(X, \"RowTimes\", rowTimes)",
        inputs: &ARRAY2TIMETABLE_INPUTS_ROW_TIMES,
        outputs: &ARRAY2TIMETABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "TT = array2timetable(X, \"SampleRate\", Fs)",
        inputs: &ARRAY2TIMETABLE_INPUTS_SAMPLE_RATE,
        outputs: &ARRAY2TIMETABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "TT = array2timetable(X, \"TimeStep\", dt)",
        inputs: &ARRAY2TIMETABLE_INPUTS_TIME_STEP,
        outputs: &ARRAY2TIMETABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "TT = array2timetable(X, nameValuePairs...)",
        inputs: &ARRAY2TIMETABLE_INPUTS_NAME_VALUE,
        outputs: &ARRAY2TIMETABLE_OUTPUT,
    },
];
pub const ARRAY2TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY2TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};

pub const READTIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "TT = readtimetable(filename, Name=Value...)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Text or spreadsheet data source.",
            },
            BuiltinParamDescriptor {
                name: "Name,Value",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "Import and row-time options.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "TT",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Imported timetable.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
