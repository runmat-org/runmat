#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PythonFrame {
    pub file: String,
    pub line: Option<u32>,
    pub function: Option<String>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, thiserror::Error)]
#[error("{type_name}: {message}")]
pub struct PythonError {
    pub type_name: String,
    pub message: String,
    pub traceback: Vec<PythonFrame>,
    pub formatted_traceback: String,
    pub cause: Option<Box<PythonError>>,
}

impl PythonError {
    pub fn host(type_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            message: message.into(),
            traceback: Vec::new(),
            formatted_traceback: String::new(),
            cause: None,
        }
    }
}
