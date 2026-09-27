//! Logical producer admission, sharing the ordinary upload ownership boundary.
use crate::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, InvalidNumericUploadResult,
    NumericElementType, ResidencyRestoreGuard,
};

/// Borrowed canonical logical bytes. Validation never widens or copies data.
#[derive(Clone, Copy, Debug)]
pub struct HostLogicalTensorView<'a> {
    pub data: &'a [u8],
    pub shape: &'a [usize],
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
    host.validate()?;
    let _residency: Vec<_> = protected
        .iter()
        .map(|handle| ResidencyRestoreGuard::new(handle))
        .collect();
    let handle = provider.upload_logical_raw(host)?;
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
