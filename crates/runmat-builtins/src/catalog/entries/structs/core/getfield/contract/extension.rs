use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const GETFIELD_TEXTUAL_INDEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "getfield-textual-index",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "getfield with an end or numeric text index is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GetfieldTextualIndexExtension"),
    };
pub const GETFIELD_OBJECT_FAMILY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "getfield-object-family", mode: BuiltinExtensionMode::RunMatOnly, description: "getfield access to RunMat object, handle, listener, and exception values is a RunMat extension", error_identifier: Some("RunMat:compatibility:GetfieldObjectFamilyExtension") };
pub const GETFIELD_INDEXED_RESIDENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "getfield-indexed-resident-field",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "getfield indexing into a resident field with typed host gather is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GetfieldIndexedResidentFieldExtension"),
    };
pub const GETFIELD_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    GETFIELD_TEXTUAL_INDEX_EXTENSION,
    GETFIELD_OBJECT_FAMILY_EXTENSION,
    GETFIELD_INDEXED_RESIDENT_EXTENSION,
];
