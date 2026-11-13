use crate::*;

pub const INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ppval-integer-query-points",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ppval accepts typed-integer query points as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PpvalIntegerQueryPointsExtension"),
};

pub const PPVAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [INTEGER_QUERY_EXTENSION];
