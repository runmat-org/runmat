use crate::*;

pub const GRPSTATS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "grpstats-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "grpstats on interactive resident GPU data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GrpstatsResidentInputExtension"),
    };
pub const GRPSTATS_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "grpstats-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "grpstats with a native-class integer matrix X is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GrpstatsIntegerDataExtension"),
    };
pub const GRPSTATS_INTEGER_ALPHA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "grpstats-integer-alpha",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "grpstats with a native-class integer Alpha value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GrpstatsIntegerAlphaExtension"),
    };
pub const GRPSTATS_INTEGER_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "grpstats-integer-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "grpstats with a native-class integer table selector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GrpstatsIntegerSelectorExtension"),
    };
pub const GRPSTATS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    GRPSTATS_RESIDENT_INPUT_EXTENSION,
    GRPSTATS_INTEGER_DATA_EXTENSION,
    GRPSTATS_INTEGER_ALPHA_EXTENSION,
    GRPSTATS_INTEGER_SELECTOR_EXTENSION,
];
