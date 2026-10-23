use crate::*;

pub const AREA_LINESPEC_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "area-linespec",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "area with a compact positional LineSpec is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AreaLineSpecExtension"),
};

pub const AREA_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AREA_LINESPEC_EXTENSION];
