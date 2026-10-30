//! Borrowed replacement literal; no intermediate String is needed for chars.

use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_value::{
    ExactArrayInitializationError, ExactHostAllocationPermit, StringElement, Utf16TextCopyError,
    Utf8Text,
};

#[derive(Clone, Copy)]
pub(in crate::builtins::missing::domain) enum TextLiteral<'a> {
    Utf8(&'a str),
    Missing,
    Chars(&'a [runmat_value::Utf16CodeUnit]),
}

impl TextLiteral<'_> {
    pub(super) fn requested_bytes(
        self,
        runtime: &RuntimeContext,
    ) -> Result<usize, Utf16TextCopyError<HostAllocationError>> {
        match self {
            Self::Missing => Ok(0),
            Self::Utf8(text) => {
                Utf8Text::requested_bytes_for_capacity(text.len()).map_err(|cause| {
                    Utf16TextCopyError::Checkpoint(HostAllocationError::Numeric(cause))
                })
            }
            Self::Chars(chars) => Utf8Text::requested_bytes_for_utf16_with(chars, |_| {
                if runtime.is_cancelled() {
                    Err(HostAllocationError::Cancelled)
                } else {
                    Ok(())
                }
            })
            .map(|(_, bytes)| bytes)
            .map_err(|error| match error {
                ExactArrayInitializationError::Allocation(cause) => {
                    Utf16TextCopyError::Checkpoint(HostAllocationError::Numeric(cause))
                }
                ExactArrayInitializationError::Element(cause) => cause,
            }),
        }
    }

    pub(super) fn copy_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
    ) -> Result<StringElement, ExactArrayInitializationError<Utf16TextCopyError<HostAllocationError>>>
    {
        match self {
            Self::Missing => Ok(StringElement::Missing),
            Self::Utf8(text) => {
                Utf8Text::try_copy_str_exact_with(text, permit, |_| receipt.checkpoint(runtime))
                    .map(StringElement::Text)
                    .map_err(|error| match error {
                        ExactArrayInitializationError::Allocation(cause) => {
                            ExactArrayInitializationError::Allocation(cause)
                        }
                        ExactArrayInitializationError::Element(cause) => {
                            ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(
                                cause,
                            ))
                        }
                    })
            }
            Self::Chars(chars) => {
                Utf8Text::try_copy_utf16_exact_with(chars, permit, |_| receipt.checkpoint(runtime))
                    .map(StringElement::Text)
            }
        }
    }

    pub(super) fn is_empty(self) -> bool {
        match self {
            Self::Missing => true,
            Self::Utf8(text) => text.is_empty(),
            Self::Chars(chars) => chars.is_empty(),
        }
    }
}
