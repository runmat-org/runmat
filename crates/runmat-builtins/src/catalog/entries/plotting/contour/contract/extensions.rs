use crate::*;

pub const CONTOUR_INTEGER_LINE_COLOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contour-integer-line-color",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contour with a typed-integer RGB line color is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContourIntegerLineColorExtension"),
    };

pub const CONTOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [CONTOUR_INTEGER_LINE_COLOR_EXTENSION];
