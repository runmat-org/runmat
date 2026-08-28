use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PythonObjectHandle(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonObjectMetadata {
    pub handle: PythonObjectHandle,
    pub type_name: String,
    pub module: String,
    pub callable: bool,
    pub iterable: bool,
    pub mapping: bool,
    pub sequence: bool,
    pub none: bool,
}

#[derive(Debug, Clone)]
pub struct PythonCallbackInvocation {
    pub callback: u64,
    pub arguments: Vec<crate::PythonValue>,
}

pub(crate) enum PythonCallbackCommand {
    Return(Result<crate::PythonValue, crate::PythonError>),
    Reenter {
        call: crate::PythonCall,
        reply: std::sync::mpsc::SyncSender<Result<Vec<crate::PythonValue>, crate::PythonError>>,
    },
}

pub(crate) enum PythonLaneEvent {
    Complete(Result<Vec<crate::PythonValue>, crate::PythonError>),
    Callback {
        invocation: PythonCallbackInvocation,
        commands: std::sync::mpsc::SyncSender<PythonCallbackCommand>,
    },
}
