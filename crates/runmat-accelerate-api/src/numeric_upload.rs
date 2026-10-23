//! Shared numeric upload validation and rejection ownership.
use crate::{AccelProvider, GpuTensorHandle, HostNumericTensorView, ProviderExecutionError};

mod query;
pub use query::class_preserving_upload_query;

/// Exact operation-owned words for the provider's native integer encoding.
/// Lanes include both complex components; this is not a numeric-value mirror.
pub struct HostNumericUploadScratch<'a> {
    pub words: &'a mut [u32],
}

/// A validated upload may own a thread-affine failed-retirement authority.
/// Keep this typed carrier through local error propagation; converting only its
/// diagnostic to `anyhow` on WASM would discard the sole custom owner.
#[derive(Debug)]
pub enum NumericUploadError {
    Provider(ProviderExecutionError),
    Invalid(InvalidNumericUploadResult),
    Completion(NumericUploadCompletionError),
}
pub type NumericUploadResult<T> = Result<T, NumericUploadError>;

impl NumericUploadError {
    /// Keep the provider's certified effect disposition and original cause.
    /// Rejected results and failed cleanup follow a returned upload result.
    pub fn into_execution_error(self) -> ProviderExecutionError {
        match self {
            Self::Provider(error) => error,
            Self::Invalid(error) => ProviderExecutionError::from_source(error).after_effect(),
            Self::Completion(error) => ProviderExecutionError::from_source(error).after_effect(),
        }
    }
}

impl From<anyhow::Error> for NumericUploadError {
    fn from(error: anyhow::Error) -> Self {
        Self::Provider(error.into())
    }
}
impl From<ProviderExecutionError> for NumericUploadError {
    fn from(error: ProviderExecutionError) -> Self {
        Self::Provider(error)
    }
}
impl std::fmt::Display for NumericUploadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(error) => std::fmt::Display::fmt(error, formatter),
            Self::Invalid(error) => std::fmt::Display::fmt(error, formatter),
            Self::Completion(error) => std::fmt::Display::fmt(error, formatter),
        }
    }
}
impl std::error::Error for NumericUploadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Provider(error) => error,
            Self::Invalid(error) => error,
            Self::Completion(error) => error,
        })
    }
}

/// A later upload failed after a prior upload had to be retired. Preserve the
/// original provider cause and any failed retirement owner independently.
#[derive(Debug)]
pub struct NumericUploadCompletionError {
    primary: Box<NumericUploadError>,
    cleanup: Option<crate::allocation::GpuHandleReleaseError>,
}

impl NumericUploadCompletionError {
    pub fn new(
        primary: NumericUploadError,
        cleanup: Option<crate::allocation::GpuHandleReleaseError>,
    ) -> Self {
        Self {
            primary: Box::new(primary),
            cleanup,
        }
    }

    pub fn primary(&self) -> &NumericUploadError {
        &self.primary
    }

    pub fn cleanup_error(&self) -> Option<&crate::allocation::GpuHandleReleaseError> {
        self.cleanup.as_ref()
    }
}

impl std::fmt::Display for NumericUploadCompletionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "numeric upload failed: {}", self.primary)?;
        if let Some(cleanup) = &self.cleanup {
            write!(formatter, "; prior upload retirement failed: {cleanup}")?;
        }
        Ok(())
    }
}

impl std::error::Error for NumericUploadCompletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.primary)
    }
}

/// A provider returned a malformed or borrowed numeric upload result.
/// This is an execution failure, not an unsupported-representation signal.
#[derive(Debug)]
pub struct InvalidNumericUploadResult {
    detail: String,
    cleanup: Option<crate::allocation::GpuHandleReleaseError>,
}
impl InvalidNumericUploadResult {
    pub(crate) fn reject(handle: GpuTensorHandle, detail: String) -> NumericUploadError {
        let cleanup = handle.release().err();
        NumericUploadError::Invalid(Self { detail, cleanup })
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

pub fn is_invalid_numeric_upload_result(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if error.is::<InvalidNumericUploadResult>() {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

/// Validate an upload through the raw provider hook with borrowed-handle
/// protection. Unlike a trait convenience method, this authority cannot be
/// overridden by a provider. Internal ownership-sensitive callers use it directly.
pub fn upload_numeric_protecting<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
) -> NumericUploadResult<GpuTensorHandle> {
    upload_numeric_protecting_admitted(provider, host, protected, None)
}

/// The same single validation/retirement owner for an upload whose provider
/// registration must consume request-local host metadata headroom.
pub fn upload_numeric_protecting_admitted<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> NumericUploadResult<GpuTensorHandle> {
    upload_numeric_protecting_admitted_with_scratch(provider, host, protected, admission, None)
}

/// Same validation and rejected-output owner, with caller-funded encoding.
pub fn upload_numeric_protecting_admitted_with_scratch<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    scratch: Option<HostNumericUploadScratch<'_>>,
) -> NumericUploadResult<GpuTensorHandle> {
    host.validate()?;
    let _borrowed_metadata = crate::residency::ResidencyRestoreSliceGuard::new(protected);
    let handle = provider.upload_numeric_raw_admitted_with_scratch(host, admission, scratch)?;
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
