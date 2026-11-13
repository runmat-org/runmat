//! Logical producer admission, sharing the ordinary upload ownership boundary.
use crate::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, InvalidNumericUploadResult,
    NumericElementType,
};

/// Borrowed canonical logical bytes. Validation never widens or copies data.
#[derive(Clone, Copy, Debug)]
pub struct HostLogicalTensorView<'a> {
    pub data: &'a [u8],
    pub shape: &'a [usize],
}

/// An operation-owned, already admitted native conversion workspace. The
/// provider fills this borrowed storage; no second logical payload or output
/// ownership is inferred from its presence.
pub enum HostLogicalUploadScratch<'a> {
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
}
impl HostLogicalTensorView<'_> {
    pub fn validate(&self) -> anyhow::Result<()> {
        let count = if self.shape.contains(&0) {
            0
        } else {
            self.shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("logical upload shape overflow"))?
        };
        if count != self.data.len() {
            anyhow::bail!(
                "logical upload shape requires {count} values, got {}",
                self.data.len()
            );
        }
        if self.data.iter().any(|&value| value > 1) {
            anyhow::bail!("logical upload requires canonical 0/1 bytes");
        }
        Ok(())
    }
}

/// Non-overridable admission and owning-result validation. The operation owner,
/// not a wire flag, supplies the admitted logical classification. A malformed
/// result uses the existing typed upload error and exact release mechanism.
pub fn upload_logical_protecting<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostLogicalTensorView<'_>,
    protected: &[&GpuTensorHandle],
) -> crate::NumericUploadResult<GpuTensorHandle> {
    upload_logical_protecting_admitted(provider, host, protected, None, None)
}

/// The same logical classification and rejected-output owner with a
/// request-local provider registration permit. Logical is not numeric U8.
pub fn upload_logical_protecting_admitted<P: AccelProvider + ?Sized>(
    provider: &P,
    host: &HostLogicalTensorView<'_>,
    protected: &[&GpuTensorHandle],
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    scratch: Option<HostLogicalUploadScratch<'_>>,
) -> crate::NumericUploadResult<GpuTensorHandle> {
    host.validate()?;
    let _residency = crate::residency::ResidencyRestoreSliceGuard::new(protected);
    let handle = provider.upload_logical_raw_admitted(host, admission, scratch)?;
    if handle.device_id() == provider.device_id()
        && handle.shape() == host.shape
        && handle.view().is_logical()
        && !crate::handle_is_transposed(&handle)
        && handle.physical().storage() == GpuTensorStorage::Real
        && matches!(
            handle.physical().element_type(),
            NumericElementType::U8 | NumericElementType::F32 | NumericElementType::F64
        )
        && !protected
            .iter()
            .any(|input| input.shares_allocation_with(&handle))
    {
        return Ok(handle);
    }
    Err(InvalidNumericUploadResult::reject(
        handle,
        "provider returned an invalid or borrowed logical upload result".into(),
    ))
}

#[cfg(test)]
mod tests;
