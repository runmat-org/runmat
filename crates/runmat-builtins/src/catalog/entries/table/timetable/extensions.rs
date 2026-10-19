use crate::*;
pub const TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable2table-typed-integer-convert-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "timetable2table with a typed-integer ConvertRowTimes value is a RunMat extension",
        error_identifier: Some(
            "RunMat:compatibility:Timetable2TableIntegerConvertRowTimesExtension",
        ),
    };
pub const TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable2table-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable2table with explicitly GPU-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Timetable2TableExplicitGpuInputExtension"),
    };
pub const TIMETABLE2TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TIMETABLE2TABLE_INTEGER_OPTION_EXTENSION,
    TIMETABLE2TABLE_EXPLICIT_GPU_EXTENSION,
];
pub const TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable-numeric-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable with numeric row times is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimetableNumericRowTimesExtension"),
    };
pub const TIMETABLE_GENERATED_ROW_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable-generated-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable with data variables but no timing source uses generated row times as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimetableGeneratedRowTimesExtension"),
    };
pub const TIMETABLE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timetable-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timetable with explicitly GPU-resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimetableExplicitGpuInputExtension"),
    };
pub const TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
    TIMETABLE_GENERATED_ROW_TIMES_EXTENSION,
    TIMETABLE_EXPLICIT_GPU_EXTENSION,
];
pub const TABLE2TIMETABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable with an explicit resident GPU payload is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableGpuInputExtension"),
    };
pub const TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-generated-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable without a time variable or timing option uses generated row times as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableGeneratedTimesExtension"),
    };
pub const TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "table2timetable-numeric-row-times",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "table2timetable with a numeric row-time vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Table2TimetableNumericRowTimesExtension"),
    };
pub const TABLE2TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TABLE2TIMETABLE_GPU_INPUT_EXTENSION,
    TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION,
    TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
];
pub const ARRAY2TIMETABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "array2timetable-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "array2timetable with an interactive resident GPU argument is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Array2TimetableGpuInputExtension"),
    };

pub const ARRAY2TIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY2TIMETABLE_GPU_INPUT_EXTENSION];

pub const READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readtimetable-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readtimetable accepts typed-integer controls whose public datatype tables are floating-only as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadtimetableTypedIntegerControlExtension"),
    };
pub const READTIMETABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    READTIMETABLE_TYPED_INTEGER_CONTROL_EXTENSION,
    TABLE2TIMETABLE_NUMERIC_ROW_TIMES_EXTENSION,
    TABLE2TIMETABLE_GENERATED_TIMES_EXTENSION,
];
