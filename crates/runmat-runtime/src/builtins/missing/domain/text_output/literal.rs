//! Borrowed replacement literal; no intermediate String is needed for chars.

use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_value::{ExactArrayInitializationError, ExactHostAllocationPermit, Utf8Text};

use super::initialization_error;

#[derive(Clone, Copy)]
pub(in crate::builtins::missing::domain) enum TextLiteral<'a> {
    Utf8(&'a str),
    Chars(&'a [char]),
}

impl TextLiteral<'_> {
    pub(super) fn requested_bytes(
        self,
        runtime: &RuntimeContext,
    ) -> Result<usize, HostAllocationError> {
        match self {
            Self::Utf8(text) => Utf8Text::requested_bytes_for_capacity(text.len())
                .map_err(HostAllocationError::Numeric),
            Self::Chars(chars) => Utf8Text::requested_bytes_for_chars_with(chars, |_| {
                if runtime.is_cancelled() {
                    Err(HostAllocationError::Cancelled)
                } else {
                    Ok(())
                }
            })
            .map(|(_, bytes)| bytes)
            .map_err(initialization_error),
        }
    }

    pub(super) fn copy_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
    ) -> Result<Utf8Text, ExactArrayInitializationError<HostAllocationError>> {
        match self {
            Self::Utf8(text) => {
                Utf8Text::try_copy_str_exact_with(text, permit, |_| receipt.checkpoint(runtime))
            }
            Self::Chars(chars) => {
                Utf8Text::try_copy_chars_exact_with(chars, permit, |_| receipt.checkpoint(runtime))
            }
        }
    }

    pub(super) fn is_empty(self) -> bool {
        match self {
            Self::Utf8(text) => text.is_empty(),
            Self::Chars(chars) => chars.is_empty(),
        }
    }
}
