use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExtensionMode {
    RunMatOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExtensionDescriptor {
    pub id: &'static str,
    pub mode: BuiltinExtensionMode,
    pub description: &'static str,
    pub error_identifier: Option<&'static str>,
}

pub const SPARSE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sparse-integer-storage",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sparse integer storage is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SparseIntegerExtension"),
};

pub const SPARSE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [SPARSE_INTEGER_EXTENSION];
