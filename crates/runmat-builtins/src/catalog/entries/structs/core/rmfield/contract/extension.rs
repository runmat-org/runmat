use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const RMFIELD_VARIADIC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rmfield-variadic-field-arguments",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "separate variadic field-name arguments are a RunMat extension; pass one character array, string array, or cell array in MATLAB compatibility mode",
    error_identifier: Some("RunMat:compatibility:RmfieldVariadicExtension"),
};

pub const RMFIELD_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[RMFIELD_VARIADIC_EXTENSION];
