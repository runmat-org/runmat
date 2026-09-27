//! Shared numeric upload validation and rejection ownership.
use crate::{AccelProvider, GpuTensorHandle, HostNumericTensorView};

/// A provider returned a malformed or borrowed numeric upload result.
/// This is an execution failure, not an unsupported-representation signal.
#[derive(Debug)]
pub struct InvalidNumericUploadResult(String);

impl std::fmt::Display for InvalidNumericUploadResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
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
    let handle = provider.upload_numeric_raw(host)?;
    let aliases = protected
        .iter()
        .any(|input| (input.device_id, input.buffer_id) == (handle.device_id, handle.buffer_id));
    if !aliases
        && handle.device_id == provider.device_id()
        && handle.shape == host.shape
        && handle.descriptor.element_type == Some(host.element_type())
        && handle.descriptor.storage == Some(host.storage)
    {
        return Ok(handle);
    }
    let detail = if aliases {
        "provider returned an upload result aliasing a borrowed input".to_owned()
    } else {
        format!("provider returned an invalid numeric upload descriptor: expected device {}, shape {:?}, element {:?}, storage {:?}; got device {}, shape {:?}, element {:?}, storage {:?}",
            provider.device_id(), host.shape, host.element_type(), host.storage,
            handle.device_id, handle.shape, handle.descriptor.element_type, handle.descriptor.storage)
    };
    if !aliases {
        // Unknown device ids belong to the invocation that produced the
        // rejected result, never an unrelated ambient default provider.
        let freed = if let Some(owner) = crate::provider_for_device(handle.device_id)
            .filter(|owner| owner.device_id() == handle.device_id)
        {
            owner.free(&handle)
        } else {
            provider.free(&handle)
        };
        if freed.is_ok() {
            crate::clear_handle_metadata(&handle);
        }
    }
    Err(InvalidNumericUploadResult(detail).into())
}

#[cfg(test)]
mod tests;
