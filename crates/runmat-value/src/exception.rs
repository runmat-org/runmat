use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MException {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
    original: Option<Arc<runmat_async::RuntimeError>>,
}

impl MException {
    pub fn new(identifier: String, message: String) -> Self {
        Self {
            identifier,
            message,
            stack: Vec::new(),
            original: None,
        }
    }

    /// Preserve the exact runtime failure caught by the VM or native engine.
    /// Manually constructed exceptions retain the source-less public form.
    pub fn from_runtime_error(
        identifier: String,
        message: String,
        stack: Vec<String>,
        original: Arc<runmat_async::RuntimeError>,
    ) -> Self {
        Self {
            identifier,
            message,
            stack,
            original: Some(original),
        }
    }

    pub fn original(&self) -> Option<&Arc<runmat_async::RuntimeError>> {
        self.original.as_ref()
    }
}

impl PartialEq for MException {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
            && self.message == other.message
            && self.stack == other.stack
    }
}
