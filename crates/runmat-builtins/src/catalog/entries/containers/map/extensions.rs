use crate::*;

pub const MAP_RESIDENT_CONSTRUCTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "containers-map-resident-constructor-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "containers.Map gathering resident constructor keys or values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainersMapResidentConstructorExtension"),
    };
pub const MAP_RESIDENT_ISKEY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-iskey-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.isKey gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentIsKeyExtension"),
};
pub const MAP_RESIDENT_VALUES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-values-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.values gathering resident selected keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentValuesExtension"),
};
pub const MAP_RESIDENT_REMOVE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-remove-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.remove gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentRemoveExtension"),
};
pub const MAP_RESIDENT_SUBSREF_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "containers-map-resident-subsref-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "containers.Map.subsref gathering resident keys is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ContainersMapResidentSubsrefExtension"),
};
pub const MAP_RESIDENT_SUBSASGN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "containers-map-resident-subsasgn-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "containers.Map.subsasgn gathering resident keys or rhs values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainersMapResidentSubsasgnExtension"),
    };
pub const CONTAINERS_MAP_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_CONSTRUCTOR_EXTENSION];
pub const CONTAINERS_MAP_ISKEY_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_ISKEY_EXTENSION];
pub const CONTAINERS_MAP_VALUES_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_VALUES_EXTENSION];
pub const CONTAINERS_MAP_REMOVE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_REMOVE_EXTENSION];
pub const CONTAINERS_MAP_SUBSREF_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_SUBSREF_EXTENSION];
pub const CONTAINERS_MAP_SUBSASGN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [MAP_RESIDENT_SUBSASGN_EXTENSION];
