use crate::*;

pub const GRIDDED_INTEGER_GRID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "griddedinterpolant-integer-grid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "griddedInterpolant with typed-integer sample grids is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerGridExtension"),
};

pub const GRIDDED_INTEGER_VALUES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "griddedinterpolant-integer-values",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "griddedInterpolant with typed-integer sample values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerValuesExtension"),
    };

pub const GRIDDED_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "griddedinterpolant-integer-query",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "griddedInterpolant with typed-integer query coordinates is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GriddedInterpolantIntegerQueryExtension"),
    };

pub const GRIDDED_INTERPOLANT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    GRIDDED_INTEGER_GRID_EXTENSION,
    GRIDDED_INTEGER_VALUES_EXTENSION,
    GRIDDED_INTEGER_QUERY_EXTENSION,
];
