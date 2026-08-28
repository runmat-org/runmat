#[derive(Debug, Clone, PartialEq)]
pub struct MException {
    pub identifier: String,
    pub message: String,
    pub stack: Vec<String>,
}

impl MException {
    pub fn new(identifier: String, message: String) -> Self {
        Self {
            identifier,
            message,
            stack: Vec::new(),
        }
    }
}
