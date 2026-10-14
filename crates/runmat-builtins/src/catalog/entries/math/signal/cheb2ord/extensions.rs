use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const CHEB2ORD_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cheb2ord-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "cheb2ord with typed-integer passband or stopband frequencies is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cheb2ordIntegerFrequencyExtension"),
    };
pub const CHEB2ORD_INTEGER_ATTENUATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cheb2ord-integer-attenuation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "cheb2ord with typed-integer ripple or attenuation values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cheb2ordIntegerAttenuationExtension"),
    };
pub const CHEB2ORD_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cheb2ord-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cheb2ord with logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cheb2ordLogicalInputExtension"),
    };
pub const CHEB2ORD_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cheb2ord-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cheb2ord with interactive resident input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Cheb2ordResidentInputExtension"),
    };

pub const CHEB2ORD_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    CHEB2ORD_INTEGER_FREQUENCY_EXTENSION,
    CHEB2ORD_INTEGER_ATTENUATION_EXTENSION,
    CHEB2ORD_LOGICAL_INPUT_EXTENSION,
    CHEB2ORD_RESIDENT_INPUT_EXTENSION,
];
