use crate::*;

pub const SPLINE_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spline-integer-sample-locations",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spline with typed-integer sample locations is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SplineIntegerSampleLocationsExtension"),
};
pub const SPLINE_INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spline-integer-sample-values",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spline with typed-integer sample values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SplineIntegerSampleValuesExtension"),
};
pub const SPLINE_INTEGER_XQ_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "spline-integer-query-points",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "spline with typed-integer query points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SplineIntegerQueryPointsExtension"),
};
pub const SPLINE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SPLINE_INTEGER_X_EXTENSION,
    SPLINE_INTEGER_Y_EXTENSION,
    SPLINE_INTEGER_XQ_EXTENSION,
];
