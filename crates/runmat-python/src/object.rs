use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PythonObjectHandle(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonObjectMetadata {
    pub handle: PythonObjectHandle,
    pub type_name: String,
    pub module: String,
    pub callable: bool,
    /// A passive iterator-slot capability. `None` is unknown for static
    /// CPython 3.9 types, whose slots cannot be inspected through its stable ABI.
    pub iterable: Option<bool>,
    pub mapping: bool,
    pub sequence: bool,
    pub none: bool,
}

#[derive(Debug, Clone)]
pub struct PythonCallbackInvocation {
    pub callback: u64,
    pub arguments: Vec<crate::PythonValue>,
}
