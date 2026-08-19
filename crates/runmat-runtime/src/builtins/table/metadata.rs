use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};

use super::TABLE_CLASS;

const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const NUM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Count.",
}];
const TABLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table input.",
}];
const HEIGHT_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table, timetable, or array input.",
}];
const HEAD_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table, timetable, or array input.",
}];
const HEAD_INPUTS_COUNT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table, timetable, or array input.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive number of leading rows to select.",
    },
];
const READTABLE_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text or spreadsheet file path.",
}];
const READTABLE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value import options.",
    },
];
const SPREADSHEET_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Spreadsheet import options struct.",
}];
const SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    }];
const DETECT_IMPORT_OPTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "opts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Detected import options struct accepted by readtable/readmatrix.",
}];
const DETECT_IMPORT_OPTIONS_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] =
    [BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    }];
const DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or spreadsheet file path to inspect.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Detection overrides such as Delimiter, Range, Sheet, Encoding, or TextType.",
    },
];
const PARQUETREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path.",
}];
const PARQUETREAD_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parquet file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Options such as SelectedVariableNames, RowGroups, or OutputType.",
    },
];
const PARQUETINFO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parquet file path to inspect.",
}];
const TABLE_INPUTS_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Variables to assemble as table columns.",
}];
const TABLE_INPUTS_PREALLOCATE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Size",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element table size vector [rows variables].",
    },
    BuiltinParamDescriptor {
        name: "VariableTypes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String array or cellstr naming each preallocated variable type.",
    },
    BuiltinParamDescriptor {
        name: "VariableNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional names for the preallocated variables.",
    },
    BuiltinParamDescriptor {
        name: "RowNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional names for the preallocated rows.",
    },
];
const GROUPSUMMARY_TABLE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variable name or names.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "datavars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Data variable name or names.",
    },
];
const GROUPSUMMARY_TABLE_BIN_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variable name or names.",
    },
    BuiltinParamDescriptor {
        name: "groupbins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping bin specification.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "datavars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Data variable name or names, optionally followed by name-value pairs.",
    },
];
const GROUPSUMMARY_ARRAY_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric array or cell array of numeric arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping array or cell array of grouping arrays.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
];
const GROUPSUMMARY_ARRAY_BIN_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric array or cell array of numeric arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping array or cell array of grouping arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupbins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping bin specification.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Grouping options such as IncludedEdge and IncludeEmptyGroups.",
    },
];
const GROUPSUMMARY_ARRAY_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary values.",
    },
    BuiltinParamDescriptor {
        name: "BG",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grouping values.",
    },
    BuiltinParamDescriptor {
        name: "BC",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Group counts.",
    },
];
const GRPSTATS_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix or table.",
    },
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variables, variable selectors, or empty grouping.",
    },
    BuiltinParamDescriptor {
        name: "whichstats",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"mean\""),
        description: "Summary statistic name or names.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Alpha, DataVars, and VarNames options.",
    },
];
const OBJECT_INDEX_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index payload.",
    },
];
const OBJECT_ASSIGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table object receiver.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];
const VALUE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];
const VALUE_AND_ARGS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options or conversion arguments.",
    },
];
const ARRAY2TABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table whose variables are the columns of A.",
}];
const ARRAY2TABLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Homogeneous array to split into table variables by column.",
}];
const ARRAY2TABLE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into table variables by column.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "VariableNames, RowNames, or DimensionNames options.",
    },
];
const ARRAY2TIMETABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TT",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Timetable whose variables are the columns of X.",
}];
const ARRAY2TIMETABLE_INPUTS_ROW_TIMES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "rowTimes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime or duration column vector labeling the rows.",
    },
];
const ARRAY2TIMETABLE_INPUTS_SAMPLE_RATE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
    },
    BuiltinParamDescriptor {
        name: "Fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive sample rate in samples per second.",
    },
];
const ARRAY2TIMETABLE_INPUTS_TIME_STEP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Homogeneous array to split into timetable variables by column.",
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
const ARRAY_DATASTORE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arrds",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "ArrayDatastore object retaining the in-memory data and read properties.",
}];
const ARRAY_DATASTORE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "In-memory array managed by the datastore.",
}];
const ARRAY_DATASTORE_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "In-memory array managed by the datastore.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "ReadSize, IterationDimension, or OutputType options.",
    },
];
const VARIADIC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Input values and name-value options.",
}];
const PREDICATE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TF",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicate result.",
}];
const WRITE_NO_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const READTABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename)",
        inputs: &READTABLE_INPUTS_FILENAME,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = readtable(filename, nameValuePairs...)",
        inputs: &READTABLE_INPUTS_NAME_VALUE,
        outputs: &ANY_OUTPUT,
    },
];
const SPREADSHEET_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions()",
        inputs: &[],
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = spreadsheetImportOptions(nameValuePairs...)",
        inputs: &SPREADSHEET_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &SPREADSHEET_IMPORT_OPTIONS_OUTPUT,
    },
];
const DETECT_IMPORT_OPTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_FILENAME,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "opts = detectImportOptions(filename, nameValuePairs...)",
        inputs: &DETECT_IMPORT_OPTIONS_INPUTS_NAME_VALUE,
        outputs: &DETECT_IMPORT_OPTIONS_OUTPUT,
    },
];
const DICTIONARY_KEYS_VALUES_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Uniform or cell-wrapped dictionary keys.",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Uniform, scalar-expanded, or cell-wrapped dictionary values.",
    },
];
const DICTIONARY_PAIR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keyValuePair1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First key/value argument pair.",
    },
    BuiltinParamDescriptor {
        name: "keyValuePairN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional key/value argument pairs.",
    },
];
const DICTIONARY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "d = dictionary()",
        inputs: &[],
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(keys, values)",
        inputs: &DICTIONARY_KEYS_VALUES_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(k1, v1, ..., kN, vN)",
        inputs: &DICTIONARY_PAIR_INPUTS,
        outputs: &ANY_OUTPUT,
    },
];
const PARQUETREAD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename)",
        inputs: &PARQUETREAD_INPUTS_FILENAME,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = parquetread(filename, nameValuePairs...)",
        inputs: &PARQUETREAD_INPUTS_NAME_VALUE,
        outputs: &ANY_OUTPUT,
    },
];
const PARQUETINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = parquetinfo(filename)",
    inputs: &PARQUETINFO_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = table(variables...)",
        inputs: &TABLE_INPUTS_VALUES,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = table(Size=sz, VariableTypes=varTypes, Name=Value...)",
        inputs: &TABLE_INPUTS_PREALLOCATE,
        outputs: &ANY_OUTPUT,
    },
];
const ARRAY2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = array2table(A)",
        inputs: &ARRAY2TABLE_INPUT,
        outputs: &ARRAY2TABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = array2table(A, nameValuePairs...)",
        inputs: &ARRAY2TABLE_INPUTS_NAME_VALUE,
        outputs: &ARRAY2TABLE_OUTPUT,
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
const ARRAY_DATASTORE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A)",
        inputs: &ARRAY_DATASTORE_INPUT,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "arrds = arrayDatastore(A, nameValuePairs...)",
        inputs: &ARRAY_DATASTORE_INPUTS_NAME_VALUE,
        outputs: &ARRAY_DATASTORE_OUTPUT,
    },
];
const GROUPSUMMARY_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "G = groupsummary(T, groupvars, method, datavars)",
        inputs: &GROUPSUMMARY_TABLE_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = groupsummary(T, groupvars, groupbins, method, datavars)",
        inputs: &GROUPSUMMARY_TABLE_BIN_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = groupsummary(A, groupvars, method)",
        inputs: &GROUPSUMMARY_ARRAY_INPUTS,
        outputs: &GROUPSUMMARY_ARRAY_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[B, BG, BC] = groupsummary(A, groupvars, groupbins, method)",
        inputs: &GROUPSUMMARY_ARRAY_BIN_INPUTS,
        outputs: &GROUPSUMMARY_ARRAY_OUTPUTS,
    },
];
const GRPSTATS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tblstats = grpstats(tbl, groupvars)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "stats = grpstats(X, group)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[stats1, ..., statsN] = grpstats(X, group, whichstats)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &ANY_OUTPUT,
    },
];
const HEIGHT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = height(A)",
    inputs: &HEIGHT_INPUT,
    outputs: &NUM_OUTPUT,
}];
const HEAD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = head(A)",
        inputs: &HEAD_INPUT,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = head(A, n)",
        inputs: &HEAD_INPUTS_COUNT,
        outputs: &ANY_OUTPUT,
    },
];
const WIDTH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "n = width(T)",
    inputs: &TABLE_INPUT,
    outputs: &NUM_OUTPUT,
}];
const COMPAT_VALUE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(A, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const COMPAT_VARIADIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = tabularBuiltin(args...)",
    inputs: &VARIADIC_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const PREDICATE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "TF = tabularPredicate(A)",
    inputs: &VALUE_INPUT,
    outputs: &PREDICATE_OUTPUT,
}];
const WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "writeTabular(T, filename, ...)",
    inputs: &VALUE_AND_ARGS_INPUTS,
    outputs: &WRITE_NO_OUTPUT,
}];
const OBJECT_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = table.subsref(obj, kind, payload)",
    inputs: &OBJECT_INDEX_INPUTS,
    outputs: &ANY_OUTPUT,
}];
const OBJECT_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = table.subsasgn(obj, kind, payload, rhs)",
    inputs: &OBJECT_ASSIGN_INPUTS,
    outputs: &ANY_OUTPUT,
}];

pub(super) const TABLE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_ARGUMENT",
    identifier: Some("RunMat:table:InvalidArgument"),
    when: "Arguments or table metadata are invalid.",
    message: "table: invalid argument",
};
pub(super) const TABLE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_INDEX",
    identifier: Some("RunMat:table:InvalidIndex"),
    when: "Table indexing is invalid.",
    message: "table: invalid index",
};
pub(super) const TABLE_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TABLE.INVALID_VARIABLE",
    identifier: Some("RunMat:table:InvalidVariable"),
    when: "A table variable name or value is invalid.",
    message: "table: invalid variable",
};
pub(super) const TABLE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.IO",
    identifier: Some("RunMat:readtable:IOError"),
    when: "readtable cannot open or read the requested file.",
    message: "readtable: file read failed",
};
pub(super) const TABLE_ERROR_UNSUPPORTED_FILE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READTABLE.UNSUPPORTED_FILE",
    identifier: Some("RunMat:readtable:UnsupportedFileType"),
    when: "readtable receives a file type outside the text or spreadsheet import backends.",
    message: "readtable: unsupported file type",
};
const TABLE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_IO,
    TABLE_ERROR_UNSUPPORTED_FILE,
];

pub const READTABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READTABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPREADSHEET_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const DETECT_IMPORT_OPTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DETECT_IMPORT_OPTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const DICTIONARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DICTIONARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const PARQUETREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const PARQUETINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARQUETINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const ARRAY2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const ARRAY2TIMETABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY2TIMETABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const ARRAY_DATASTORE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY_DATASTORE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const GROUPSUMMARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GROUPSUMMARY_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const GRPSTATS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRPSTATS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const HEIGHT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEIGHT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const HEAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const WIDTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WIDTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_COMPAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VALUE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_VARIADIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COMPAT_VARIADIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_PREDICATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PREDICATE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_WRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TABLE_ERRORS,
};
pub const TABLE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OBJECT_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &TABLE_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::table")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "table",
    op_kind: GpuOpKind::Custom("table"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Tables are host containers. GPU variables are gathered when tabular algorithms need row-wise access.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::table")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "table",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Tables are structured host containers and are not fusion operands.",
};

pub(super) fn table_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(TABLE_CLASS);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn table_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(TABLE_CLASS)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_ARGUMENT, message)
}

pub(super) fn invalid_index(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_INDEX, message)
}

pub(super) fn invalid_variable(message: impl Into<String>) -> RuntimeError {
    table_error(&TABLE_ERROR_INVALID_VARIABLE, message)
}

pub(super) fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(ToString::to_string);
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(TABLE_CLASS)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
