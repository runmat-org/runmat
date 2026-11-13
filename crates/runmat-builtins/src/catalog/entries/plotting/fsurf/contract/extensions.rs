use crate::*;

pub const FSURF_INTEGER_DOMAIN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsurf-integer-domain",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsurf with a typed-integer domain vector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsurfIntegerDomainExtension"),
};

pub const FSURF_LOGICAL_NUMERIC_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsurf-logical-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "fsurf with a logical numeric domain, control, or callback sample is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsurfLogicalNumericInputExtension"),
    };

pub const FSURF_SINGLE_NUMERIC_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsurf-single-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsurf with a single-precision numeric domain, control, or callback sample is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsurfSingleNumericInputExtension"),
};

pub const FSURF_INTEGER_MESH_DENSITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsurf-integer-mesh-density",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsurf with a typed-integer MeshDensity is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsurfIntegerMeshDensityExtension"),
    };

pub const FSURF_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fsurf-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fsurf with a typed-integer numeric axes alias is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FsurfIntegerAxesHandleExtension"),
};

pub const FSURF_INTEGER_CALLBACK_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsurf-integer-callback-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsurf with a callback returning typed-integer samples is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsurfIntegerCallbackOutputExtension"),
    };

pub const FSURF_INTEGER_STYLE_PROPERTY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsurf-integer-style-property",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsurf with a typed-integer surface style property is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsurfIntegerStylePropertyExtension"),
    };

pub const FSURF_RESIDENT_CALLBACK_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fsurf-resident-callback-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fsurf with a provider-resident callback result is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FsurfResidentCallbackOutputExtension"),
    };

pub const FSURF_EXTENSIONS: [BuiltinExtensionDescriptor; 8] = [
    FSURF_INTEGER_DOMAIN_EXTENSION,
    FSURF_LOGICAL_NUMERIC_INPUT_EXTENSION,
    FSURF_SINGLE_NUMERIC_INPUT_EXTENSION,
    FSURF_INTEGER_MESH_DENSITY_EXTENSION,
    FSURF_INTEGER_AXES_EXTENSION,
    FSURF_INTEGER_CALLBACK_OUTPUT_EXTENSION,
    FSURF_INTEGER_STYLE_PROPERTY_EXTENSION,
    FSURF_RESIDENT_CALLBACK_OUTPUT_EXTENSION,
];
