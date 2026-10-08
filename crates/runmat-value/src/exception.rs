use runmat_async::RuntimeErrorOwner;

#[derive(Debug, Clone)]
pub struct MException {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
    original: Option<RuntimeErrorOwner>,
}

/// Language fields and the optional exact caught error, moved together.
pub struct MExceptionParts {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
    pub original: Option<RuntimeErrorOwner>,
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
        original: RuntimeErrorOwner,
    ) -> Self {
        Self {
            identifier,
            message,
            stack,
            original: Some(original),
        }
    }

    pub fn original(&self) -> Option<&RuntimeErrorOwner> {
        self.original.as_ref()
    }

    /// Consume the exception without cloning the owner of its caught error.
    pub fn into_parts(self) -> MExceptionParts {
        MExceptionParts {
            identifier: self.identifier,
            message: self.message,
            stack: self.stack,
            original: self.original,
        }
    }
}

impl PartialEq for MException {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
            && self.message == other.message
            && self.stack == other.stack
    }
}
