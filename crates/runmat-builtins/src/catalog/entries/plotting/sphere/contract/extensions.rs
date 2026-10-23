use crate::*;

pub const SPHERE_TYPED_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sphere-typed-face-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sphere with a typed-integer face count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SphereTypedFaceCountExtension"),
};

pub const SPHERE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPHERE_TYPED_N_EXTENSION];
