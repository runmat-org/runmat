use super::{real_allocation_mut, PreparedHostNumericalOutput};
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::NumericClass;
use runmat_value::{ComplexTensor, NumericDType, Tensor};

impl PreparedHostNumericalOutput {
    pub(crate) fn matches(&self, dtype: NumericDType, shape: &[usize]) -> bool {
        self.dtype == dtype && self.shape() == shape
    }

    pub(crate) fn matches_floating_cast(
        &self,
        class: NumericClass,
        dtype: NumericDType,
        shape: &[usize],
    ) -> bool {
        self.operation == BuiltinPreparationOperation::FloatingCast(class)
            && self.matches(dtype, shape)
    }

    pub(crate) fn copy_f32_from_f32(
        &mut self,
        context: &RuntimeContext,
        source: &[f32],
    ) -> Result<(), HostAllocationError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        receipt.checkpoint(context)?;
        let target = real_allocation_mut(allocation)?
            .try_as_f32_slice_mut_unique()
            .ok_or(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
            ))?;
        ensure_same_len(target.len(), source.len())?;
        for (target, source) in target
            .chunks_mut(Self::COPY_CHUNK)
            .zip(source.chunks(Self::COPY_CHUNK))
        {
            receipt.checkpoint(context)?;
            target.copy_from_slice(source);
        }
        Ok(())
    }

    pub(crate) fn copy_f32_from_f64(
        &mut self,
        context: &RuntimeContext,
        source: &[f64],
    ) -> Result<(), HostAllocationError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        receipt.checkpoint(context)?;
        let target = real_allocation_mut(allocation)?
            .try_as_f32_slice_mut_unique()
            .ok_or(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
            ))?;
        ensure_same_len(target.len(), source.len())?;
        for (target, source) in target
            .chunks_mut(Self::COPY_CHUNK)
            .zip(source.chunks(Self::COPY_CHUNK))
        {
            receipt.checkpoint(context)?;
            for (target, source) in target.iter_mut().zip(source) {
                *target = *source as f32;
            }
        }
        Ok(())
    }

    pub(crate) fn copy_f64_from_f32(
        &mut self,
        context: &RuntimeContext,
        source: &[f32],
    ) -> Result<(), HostAllocationError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        receipt.checkpoint(context)?;
        let target = real_allocation_mut(allocation)?
            .try_as_f64_slice_mut_unique()
            .ok_or(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
            ))?;
        ensure_same_len(target.len(), source.len())?;
        for (target, source) in target
            .chunks_mut(Self::COPY_CHUNK)
            .zip(source.chunks(Self::COPY_CHUNK))
        {
            receipt.checkpoint(context)?;
            for (target, source) in target.iter_mut().zip(source) {
                *target = f64::from(*source);
            }
        }
        Ok(())
    }

    pub(crate) fn copy_f64_from_f64(
        &mut self,
        context: &RuntimeContext,
        source: &[f64],
    ) -> Result<(), HostAllocationError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        receipt.checkpoint(context)?;
        let target = real_allocation_mut(allocation)?
            .try_as_f64_slice_mut_unique()
            .ok_or(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
            ))?;
        ensure_same_len(target.len(), source.len())?;
        for (target, source) in target
            .chunks_mut(Self::COPY_CHUNK)
            .zip(source.chunks(Self::COPY_CHUNK))
        {
            receipt.checkpoint(context)?;
            target.copy_from_slice(source);
        }
        Ok(())
    }

    pub(crate) fn into_tensor(
        self,
        context: &RuntimeContext,
    ) -> Result<Tensor, HostAllocationError> {
        self.receipt.checkpoint(context)?;
        self.into_real_allocation()
    }

    pub(crate) fn into_complex_tensor(
        self,
        context: &RuntimeContext,
    ) -> Result<ComplexTensor, HostAllocationError> {
        self.receipt.checkpoint(context)?;
        let allocation = self.allocation.complex.ok_or_else(unsupported_class)?;
        Ok(allocation)
    }
}

fn ensure_same_len(left: usize, right: usize) -> Result<(), HostAllocationError> {
    if left == right {
        Ok(())
    } else {
        Err(HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
        ))
    }
}

fn unsupported_class() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::UnsupportedClass)
}
