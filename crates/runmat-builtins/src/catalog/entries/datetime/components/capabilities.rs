use crate::*;

pub const HOUR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hour-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hour with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:HourTypedLegacySerialExtension"),
    };
pub const YEAR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "year-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "year with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:YearTypedLegacySerialExtension"),
    };
pub const MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "minute-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "minute with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MinuteTypedLegacySerialExtension"),
    };
pub const MONTH_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "month-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "month with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MonthTypedLegacySerialExtension"),
    };
pub const DAY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const HOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const YEAR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MINUTE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MONTH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
