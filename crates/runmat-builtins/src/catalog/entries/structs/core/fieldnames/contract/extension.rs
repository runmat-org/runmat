use crate::{BuiltinExtensionDescriptor, BuiltinExtensionMode};

pub const FIELDNAMES_OBJECT_FAMILY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fieldnames-object-family",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fieldnames introspection for RunMat value, handle, and listener objects is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FieldnamesObjectFamilyExtension"),
    };

pub const FIELDNAMES_EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[FIELDNAMES_OBJECT_FAMILY_EXTENSION];
