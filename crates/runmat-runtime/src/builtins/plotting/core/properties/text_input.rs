//! Borrowed UTF-16 validity proof acquired before property mutation callbacks.
//!
//! It does not own a decoded staging payload. The immutable original Value is
//! decoded only when the callback consumes the selected text property.

use runmat_value::{ExactArrayInitializationError, Utf16TextCopyError, Utf8Text, Value};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct ValidatedStyleValue<'a>(&'a Value);

impl<'a> ValidatedStyleValue<'a> {
    pub(super) fn new(value: &'a Value, builtin: &str) -> crate::BuiltinResult<Self> {
        if let Value::CharArray(characters) = value {
            Utf8Text::requested_bytes_for_utf16_with(characters.data(), |_| {
                Ok::<(), std::convert::Infallible>(())
            })
            .map_err(|error| match error {
                ExactArrayInitializationError::Allocation(source) => {
                    crate::builtins::plotting::plotting_error_with_source(
                        builtin,
                        source.to_string(),
                        source,
                    )
                }
                ExactArrayInitializationError::Element(Utf16TextCopyError::Decode(source)) => {
                    crate::builtins::plotting::plotting_error_with_source(
                        builtin,
                        source.to_string(),
                        source,
                    )
                }
                ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(never)) => {
                    match never {}
                }
            })?;
        }
        Ok(Self(value))
    }

    pub(super) fn value(self) -> &'a Value {
        self.0
    }

    pub(super) fn text(self) -> Option<String> {
        // The proof holds the same immutable Value/backing throughout the
        // synchronous mutation. Failure would contradict the canonical quote
        // pass, not classify user data or suppress a decoding error.
        crate::builtins::plotting::style::value_as_string(self.0)
            .expect("the same immutable UTF-16 backing was validated before mutation")
    }
}

pub(super) struct ValidatedStylePairs<'a, 'v>(&'a [(String, &'v Value)]);

impl<'a, 'v> ValidatedStylePairs<'a, 'v> {
    pub(super) fn new(
        pairs: &'a [(String, &'v Value)],
        builtin: &str,
    ) -> crate::BuiltinResult<Self> {
        for (_, value) in pairs {
            ValidatedStyleValue::new(value, builtin)?;
        }
        Ok(Self(pairs))
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&str, ValidatedStyleValue<'v>)> + '_ {
        self.0
            .iter()
            .map(|(key, value)| (key.as_str(), ValidatedStyleValue(value)))
    }
}
