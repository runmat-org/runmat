use crate::*;

pub const PIE_INTEGER_VALUES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pie-integer-values",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pie accepts typed-integer slice values as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PieIntegerValuesExtension"),
};

pub const PIE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PIE_INTEGER_VALUES_EXTENSION];
