//! Shared numeric upload validation and rejection ownership.
use crate::{AccelProvider, GpuTensorHandle, HostNumericTensorView};

/// A provider returned a malformed or borrowed numeric upload result.
/// This is an execution failure, not an unsupported-representation signal.
#[derive(Debug)]
pub struct InvalidNumericUploadResult {
    detail: String,
    cleanup: Option<crate::allocation::GpuHandleReleaseError>,
}
impl InvalidNumericUploadResult {
    pub(crate) fn reject(handle: GpuTensorHandle, detail: String) -> anyhow::Error {
        let cleanup = handle.release().err();
        Self { detail, cleanup }.into()
    }
    /// Failed retirement retains the exact owner through this error carrier.
    pub fn cleanup_error(&self) -> Option<&crate::allocation::GpuHandleReleaseError> {
        self.cleanup.as_ref()
    }
}

impl std::fmt::Display for InvalidNumericUploadResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.detail)?;
        if let Some(cleanup) = &self.cleanup {
            write!(formatter, "; {cleanup}")?;
        }
        Ok(())
    }
}
impl std::error::Error for InvalidNumericUploadResult {}

pub fn is_invalid_numeric_upload_result(error: &anyhow::Error) -> bool {
    error.is::<InvalidNumericUploadResult>()
}

/// Validate an upload through the raw provider hook with borrowed-handle
/// protection. Unlike a trait convenience method, this authority cannot be
/// overridden by a provider. Internal ownership-sensitive callers use it directly.
pub fn upload_numeric_protecting<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
) -> anyhow::Result<GpuTensorHandle> {
    host.validate()?;
    let _borrowed_metadata: Vec<_> = protected
        .iter()
        .map(|handle| crate::ResidencyRestoreGuard::new(handle))
        .collect();
    let handle = provider.upload_numeric_raw(host)?;
    let aliases = protected
        .iter()
        .any(|input| input.shares_allocation_with(&handle));
    if !aliases
        && handle.device_id() == provider.device_id()
        && handle.shape() == host.shape
        && handle.descriptor().element_type == Some(host.element_type())
        && handle.descriptor().storage == Some(host.storage)
        && !crate::handle_is_logical(&handle)
        && !crate::handle_is_transposed(&handle)
        && crate::handle_class_identity(&handle)
            .is_none_or(|class| class == host.element_type().class_identity())
    {
        return Ok(handle);
    }
    let detail = if aliases {
        "provider returned an upload result aliasing a borrowed input".to_owned()
    } else {
        format!("provider returned an invalid numeric upload descriptor: expected device {}, shape {:?}, element {:?}, storage {:?}; got device {}, shape {:?}, element {:?}, storage {:?}, logical {}, class {:?}, transpose {:?}",
            provider.device_id(), host.shape, host.element_type(), host.storage,
            handle.device_id(), handle.shape(), handle.descriptor().element_type, handle.descriptor().storage,
            crate::handle_is_logical(&handle), crate::handle_class_identity(&handle), crate::handle_transpose_info(&handle))
    };
    // Release only the result's owning reference. Borrowed aliases retain
    // their own leases; no ambient or producing-provider free authority exists.
    Err(InvalidNumericUploadResult::reject(handle, detail))
}

#[cfg(test)]
mod tests;
