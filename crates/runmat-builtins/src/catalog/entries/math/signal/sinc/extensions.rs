use crate::*;
pub const SINC_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "sinc-nonfloating-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "sinc with integer or logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SincNonfloatingInputExtension"),
    };

pub const SINC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SINC_NONFLOATING_INPUT_EXTENSION];
