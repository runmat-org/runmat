use crate::Utf8Text;
use runmat_async::RuntimeErrorOwner;

mod stack;
pub use stack::ExceptionStack;

#[derive(Debug, Clone)]
pub struct MException {
    data: crate::immutable_metadata::ImmutableMetadata<MExceptionParts>,
}

/// Language fields and the optional exact caught error, moved together.
#[derive(Debug, Clone)]
pub struct MExceptionParts {
    pub identifier: Utf8Text,
    pub message: Utf8Text,
    pub stack: ExceptionStack,
    pub original: Option<RuntimeErrorOwner>,
}

impl MException {
    pub fn new(identifier: String, message: String) -> Self {
        Self::from_owned_fields(
            identifier.into(),
            message.into(),
            ExceptionStack::default(),
            None,
        )
    }

    /// Preserve the exact runtime failure caught by the VM or native engine.
    /// Manually constructed exceptions retain the source-less public form.
    pub fn from_runtime_error(
        identifier: String,
        message: String,
        stack: Vec<String>,
        original: RuntimeErrorOwner,
    ) -> Self {
        Self::from_owned_fields(
            identifier.into(),
            message.into(),
            Self::stack_from_unbounded(stack),
            Some(original),
        )
    }

    /// Ordinary construction of the original immutable exception generation.
    /// Text, stack, and caught-error owners move without copying their payloads.
    pub fn from_owned_fields(
        identifier: Utf8Text,
        message: Utf8Text,
        stack: ExceptionStack,
        original: Option<RuntimeErrorOwner>,
    ) -> Self {
        Self {
            data: crate::immutable_metadata::ImmutableMetadata::new(MExceptionParts {
                identifier,
                message,
                stack,
                original,
            }),
        }
    }

    pub fn requested_header_bytes() -> usize {
        crate::immutable_metadata::ImmutableMetadata::<MExceptionParts>::requested_header_bytes()
    }

    /// Admit the new header before allocating it. The caller independently
    /// retains/adopts the text, stack, and original caught-error owners; this
    /// header claim does not certify those separately allocated payloads.
    pub fn try_from_owned_fields_with_exact_header(
        fields: MExceptionParts,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<Self, crate::ExactHostNumericAllocationError> {
        crate::immutable_metadata::ImmutableMetadata::try_new_exact(fields, permit)
            .map(|data| Self { data })
    }

    /// Ordinary-policy stack construction. Each String moves into its text
    /// owner; future exception clones retain this original stack generation.
    pub fn stack_from_unbounded(stack: Vec<String>) -> ExceptionStack {
        ExceptionStack::from_unbounded_strings(stack)
    }

    pub fn original(&self) -> Option<&RuntimeErrorOwner> {
        self.data.get().original.as_ref()
    }

    /// Consume the exception without cloning the owner of its caught error.
    pub fn into_parts(self) -> MExceptionParts {
        self.data.into_inner()
    }
}

impl std::ops::Deref for MException {
    type Target = MExceptionParts;

    fn deref(&self) -> &Self::Target {
        self.data.get()
    }
}

impl PartialEq for MException {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
            && self.message == other.message
            && self.stack == other.stack
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_and_parts_retain_the_original_text_and_stack() {
        let original = MException::from_owned_fields(
            "RunMat:Example".into(),
            "detail".into(),
            MException::stack_from_unbounded(vec!["driver".into(), "worker".into()]),
            None,
        );
        let retained = original.clone();
        assert_eq!(original.identifier.as_ptr(), retained.identifier.as_ptr());
        assert_eq!(original.message.as_ptr(), retained.message.as_ptr());
        assert_eq!(original.stack.as_ptr(), retained.stack.as_ptr());
        assert_eq!(original.stack[0].as_ptr(), retained.stack[0].as_ptr());
        let parts = original.into_parts();
        assert_eq!(parts.stack.as_ptr(), retained.stack.as_ptr());
        drop(parts);
        assert_eq!(retained.stack[1].as_str(), "worker");
    }
}
