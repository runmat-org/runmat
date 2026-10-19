use serde::{Deserialize, Serialize};

#[cfg(not(target_family = "wasm"))]
mod capture;

#[cfg(not(target_family = "wasm"))]
pub(crate) use capture::capture_pending_exception;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JavaStackFrame {
    pub class_name: String,
    pub method_name: String,
    pub file_name: Option<String>,
    pub line: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JavaException {
    pub class_name: String,
    pub message: Option<String>,
    pub frames: Vec<JavaStackFrame>,
    pub cause: Option<Box<JavaException>>,
}

impl JavaException {
    pub fn summary(&self) -> String {
        match self.message.as_deref() {
            Some(message) if !message.is_empty() => format!("{}: {message}", self.class_name),
            _ => self.class_name.clone(),
        }
    }

    pub fn cause_chain(&self) -> Vec<&JavaException> {
        let mut causes = Vec::new();
        let mut current = Some(self);
        while let Some(exception) = current {
            causes.push(exception);
            current = exception.cause.as_deref();
        }
        causes
    }
}

impl std::fmt::Display for JavaException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.summary())?;
        for cause in self.cause_chain().into_iter().skip(1) {
            write!(formatter, "\nCaused by: {}", cause.summary())?;
        }
        Ok(())
    }
}

impl std::error::Error for JavaException {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cause_chain_preserves_java_types_and_messages() {
        let exception = JavaException {
            class_name: "java.lang.IllegalStateException".into(),
            message: Some("fixture failure".into()),
            frames: Vec::new(),
            cause: Some(Box::new(JavaException {
                class_name: "java.io.IOException".into(),
                message: Some("fixture cause".into()),
                frames: Vec::new(),
                cause: None,
            })),
        };
        assert_eq!(exception.cause_chain().len(), 2);
        assert!(exception.to_string().contains("java.io.IOException"));
    }
}
