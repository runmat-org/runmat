use crate::*;

pub const POLARPLOT_COMPLEX_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "polarplot-complex-integer-z",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "polarplot accepts paired complex-integer Z data as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PolarplotComplexIntegerZExtension"),
    };

pub const POLARPLOT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [POLARPLOT_COMPLEX_INTEGER_EXTENSION];
