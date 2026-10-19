use crate::*;
pub const CONFUSIONMAT_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer group labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerGroupExtension"),
    };
pub const CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-grouphat",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer grouphat labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerGrouphatExtension"),
    };
pub const CONFUSIONMAT_INTEGER_ORDER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-integer-order",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with typed-integer Order labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatIntegerOrderExtension"),
    };
pub const CONFUSIONMAT_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "confusionmat-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "confusionmat with resident inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ConfusionmatResidentInputExtension"),
    };
pub const CONFUSIONMAT_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    CONFUSIONMAT_INTEGER_GROUP_EXTENSION,
    CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION,
    CONFUSIONMAT_INTEGER_ORDER_EXTENSION,
    CONFUSIONMAT_RESIDENT_INPUT_EXTENSION,
];
