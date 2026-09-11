use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const SETFIELD_TEXTUAL_INDEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "setfield-textual-index",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "setfield with an end or numeric text index is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SetfieldTextualIndexExtension"),
    };
pub const SETFIELD_OBJECT_FAMILY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "setfield-object-family",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "setfield access to RunMat object and handle values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SetfieldObjectFamilyExtension"),
    };
pub const SETFIELD_INDEXED_RESIDENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "setfield-indexed-resident-field", mode: BuiltinExtensionMode::RunMatOnly, description: "setfield indexed mutation of a resident field with typed host gather is a RunMat extension", error_identifier: Some("RunMat:compatibility:SetfieldIndexedResidentFieldExtension") };
pub const SETFIELD_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    SETFIELD_TEXTUAL_INDEX_EXTENSION,
    SETFIELD_OBJECT_FAMILY_EXTENSION,
    SETFIELD_INDEXED_RESIDENT_EXTENSION,
];
