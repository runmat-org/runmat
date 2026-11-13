use crate::*;

pub const DECONV_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deconv-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deconv with typed-integer coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DeconvIntegerInputExtension"),
};
pub const DECONV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deconv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deconv with logical coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DeconvLogicalInputExtension"),
};
pub const DECONV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DECONV_INTEGER_INPUT_EXTENSION,
    DECONV_LOGICAL_INPUT_EXTENSION,
];
