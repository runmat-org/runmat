use crate::*;

pub const GROUPSUMMARY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupsummary-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "groupsummary on interactive resident GPU data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupsummaryResidentInputExtension"),
    };
pub const GROUPSUMMARY_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "groupsummary-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "groupsummary with a native-class integer name-value control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GroupsummaryIntegerControlExtension"),
    };
pub const GROUPSUMMARY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    GROUPSUMMARY_RESIDENT_INPUT_EXTENSION,
    GROUPSUMMARY_INTEGER_CONTROL_EXTENSION,
];
