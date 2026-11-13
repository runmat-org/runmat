use crate::*;

pub const PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pchip-integer-sample-locations",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pchip accepts typed-integer sample locations as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PchipIntegerSampleLocationsExtension"),
    };
pub const PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pchip-integer-sample-values",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pchip accepts typed-integer sample values as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PchipIntegerSampleValuesExtension"),
    };
pub const PCHIP_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pchip-integer-query-points",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pchip accepts typed-integer query points as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PchipIntegerQueryPointsExtension"),
};
pub const PCHIP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION,
    PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION,
    PCHIP_INTEGER_QUERY_EXTENSION,
];
