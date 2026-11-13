use crate::*;

pub const FILL3_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fill3-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fill3 with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fill3IntegerAxesHandleExtension"),
};

pub const FILL3_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FILL3_INTEGER_AXES_EXTENSION];
