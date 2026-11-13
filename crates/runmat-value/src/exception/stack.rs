//! Exception stack generations retain their original text and slot owners.

use std::ops::Deref;

use crate::{ExactArrayStorage, Utf8Text};

/// Read-only stack storage. Every constructible state is empty or shared;
/// cloning an exception never copies its stack or its text payloads.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExceptionStack(ExactArrayStorage<Utf8Text>);

impl ExceptionStack {
    pub fn from_unbounded_strings(stack: Vec<String>) -> Self {
        if stack.is_empty() {
            Self::default()
        } else {
            Self(ExactArrayStorage::shared_unbounded(
                stack.into_iter().map(Utf8Text::from).collect(),
            ))
        }
    }

    /// Accept only already-admitted canonical storage. Refusal preserves the
    /// original owner; an ordinary Vec is never silently promoted or copied.
    pub fn try_from_exact_storage(
        stack: ExactArrayStorage<Utf8Text>,
    ) -> Result<Self, ExactArrayStorage<Utf8Text>> {
        if stack.is_exact() {
            Ok(Self(stack))
        } else {
            Err(stack)
        }
    }

    /// Explicit ordinary conversion for consumers whose contract owns Strings.
    pub fn into_strings_unbounded(self) -> Vec<String> {
        match self.0.try_into_owned() {
            Ok(stack) => stack
                .into_iter()
                .map(Utf8Text::into_string_unbounded)
                .collect(),
            Err(stack) => stack
                .iter()
                .map(|frame| frame.as_str().to_owned())
                .collect(),
        }
    }
}

impl Deref for ExceptionStack {
    type Target = [Utf8Text];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
