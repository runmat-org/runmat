use crate::*;

pub const FILL_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fill-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fill with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FillIntegerAxesHandleExtension"),
};

pub const FILL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FILL_INTEGER_AXES_EXTENSION];
