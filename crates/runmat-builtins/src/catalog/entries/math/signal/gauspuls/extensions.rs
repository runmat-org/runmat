use crate::*;

pub const GAUSPULS_INTEGER_TIME_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-integer-time",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with typed-integer sample times is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsIntegerTimeExtension"),
    };

pub const GAUSPULS_LOGICAL_TIME_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-logical-time",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with logical sample times is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsLogicalTimeExtension"),
    };

pub const GAUSPULS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with a typed-integer scalar control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsIntegerControlExtension"),
    };

pub const GAUSPULS_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-logical-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with a logical scalar control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsLogicalControlExtension"),
    };

pub const GAUSPULS_SINGLE_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-single-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with a single-precision scalar control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsSingleControlExtension"),
    };

pub const GAUSPULS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gauspuls-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gauspuls with an interactive resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GauspulsResidentInputExtension"),
    };

pub const GAUSPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    GAUSPULS_INTEGER_TIME_EXTENSION,
    GAUSPULS_LOGICAL_TIME_EXTENSION,
    GAUSPULS_INTEGER_CONTROL_EXTENSION,
    GAUSPULS_LOGICAL_CONTROL_EXTENSION,
    GAUSPULS_SINGLE_CONTROL_EXTENSION,
    GAUSPULS_RESIDENT_INPUT_EXTENSION,
];
