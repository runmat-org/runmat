//! Exact, cancellable native buffer copies for factor maps and LU row scale.

use super::*;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_execution::value::{DenseValue, ElementType};
use runmat_value::{
    ExactHostIndexBufferAllocation, ExactHostNumericAllocation, NumericDType, NumericStorage,
};

pub(super) struct IndexCopy<'a> {
    cursor: usize,
    exact: Option<ExactHostIndexBufferAllocation>,
    ordinary: Vec<usize>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a crate::context::RuntimeContext>,
}

impl<'a> IndexCopy<'a> {
    pub(super) fn new(
        source: &[u64],
        runtime: Option<&'a crate::context::RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        let raw = ExactHostIndexBufferAllocation::requested_bytes(source.len())
            .map_err(|error| ValueCodecError::allocation(path, error.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| ValueCodecError::allocation(path, HostAllocationError::Overflow))?,
            ExactHostIndexBufferAllocation::retained_claim_count(),
        )
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained)
                .map_err(|error| ValueCodecError::allocation(path, error))?,
        )
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let exact = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed map runtime");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            let result = ExactHostIndexBufferAllocation::new(source.len(), &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            Some(result.map_err(|error| ValueCodecError::allocation(path, error.into()))?)
        } else {
            None
        };
        Ok(Self {
            cursor: 0,
            exact,
            ordinary: if receipt.is_none() {
                Vec::with_capacity(source.len())
            } else {
                Vec::new()
            },
            receipt,
            runtime,
        })
    }

    pub(super) fn advance(
        &mut self,
        source: &[u64],
        maximum_work: usize,
        path: &str,
    ) -> Result<(usize, Option<HostIndexBuffer>), ValueCodecError> {
        self.checkpoint(path)?;
        let end = self.cursor.saturating_add(maximum_work).min(source.len());
        for index in self.cursor..end {
            let value = usize::try_from(source[index]).map_err(|_| {
                ValueCodecError::invalid(path, "factor index exceeds platform limits")
            })?;
            if let Some(exact) = &mut self.exact {
                exact.as_mut_slice()[index] = value;
            } else {
                self.ordinary.push(value);
            }
        }
        let work = end - self.cursor;
        self.cursor = end;
        self.checkpoint(path)?;
        if end != source.len() {
            return Ok((work.max(1), None));
        }
        let result = if let Some(exact) = self.exact.take() {
            exact
                .into_buffer_fallible()
                .map_err(|error| ValueCodecError::allocation(path, error.into()))?
        } else {
            HostIndexBuffer::new(std::mem::take(&mut self.ordinary))
        };
        self.checkpoint(path)?;
        Ok((work.max(1), Some(result)))
    }

    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed map runtime"))
                .map_err(|error| ValueCodecError::allocation(path, error))?;
        }
        Ok(())
    }
}

enum OrdinaryScale {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub(super) struct ScaleCopy<'a> {
    cursor: usize,
    exact: Option<ExactHostNumericAllocation>,
    ordinary: Option<OrdinaryScale>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a crate::context::RuntimeContext>,
}

impl<'a> ScaleCopy<'a> {
    pub(super) fn new(
        source: &DenseValue,
        runtime: Option<&'a crate::context::RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        let dtype = match source.element_type {
            ElementType::F32 => NumericDType::F32,
            ElementType::F64 => NumericDType::F64,
            _ => {
                return Err(ValueCodecError::invalid(
                    path,
                    "factor scale requires real floating storage",
                ))
            }
        };
        if source.shape.len() != 2 || source.shape[1] != 1 {
            return Err(ValueCodecError::invalid(
                path,
                "factor scale requires a column shape",
            ));
        }
        let count = usize::try_from(source.shape[0])
            .map_err(|_| ValueCodecError::invalid(path, "factor scale exceeds platform limits"))?;
        let width = usize::try_from(source.element_type.byte_width())
            .map_err(|_| ValueCodecError::invalid(path, "invalid factor scale width"))?;
        if count.checked_mul(width) != Some(source.little_endian_data.len()) {
            return Err(ValueCodecError::invalid(
                path,
                "factor scale shape and bytes disagree",
            ));
        }
        let raw = ExactHostNumericAllocation::requested_bytes(dtype, count)
            .map_err(|error| ValueCodecError::allocation(path, error.into()))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(raw)
                .map_err(|_| ValueCodecError::allocation(path, HostAllocationError::Overflow))?,
            ExactHostNumericAllocation::retained_claim_count(),
        )
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained)
                .map_err(|error| ValueCodecError::allocation(path, error))?,
        )
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let exact = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed scale runtime");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            let result = ExactHostNumericAllocation::new(dtype, count, &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            Some(result.map_err(|error| ValueCodecError::allocation(path, error.into()))?)
        } else {
            None
        };
        let ordinary = if receipt.is_none() {
            Some(match dtype {
                NumericDType::F32 => OrdinaryScale::F32(Vec::with_capacity(count)),
                NumericDType::F64 => OrdinaryScale::F64(Vec::with_capacity(count)),
                _ => unreachable!("factor scale is floating"),
            })
        } else {
            None
        };
        Ok(Self {
            cursor: 0,
            exact,
            ordinary,
            receipt,
            runtime,
        })
    }

    pub(super) fn advance(
        &mut self,
        source: &DenseValue,
        maximum_work: usize,
        path: &str,
    ) -> Result<(usize, Option<HostNumericBuffer>), ValueCodecError> {
        self.checkpoint(path)?;
        let width = usize::try_from(source.element_type.byte_width())
            .map_err(|_| ValueCodecError::invalid(path, "invalid factor scale width"))?;
        let count = source.little_endian_data.len() / width;
        let end = self.cursor.saturating_add(maximum_work).min(count);
        for index in self.cursor..end {
            let bytes = &source.little_endian_data[index * width..(index + 1) * width];
            match source.element_type {
                ElementType::F32 => {
                    let value = f32::from_bits(u32::from_le_bytes(
                        bytes.try_into().expect("checked width"),
                    ));
                    if let Some(exact) = &mut self.exact {
                        exact.as_f32_slice_mut().expect("F32 scale")[index] = value;
                    } else if let Some(OrdinaryScale::F32(values)) = &mut self.ordinary {
                        values.push(value);
                    }
                }
                ElementType::F64 => {
                    let value = f64::from_bits(u64::from_le_bytes(
                        bytes.try_into().expect("checked width"),
                    ));
                    if let Some(exact) = &mut self.exact {
                        exact.as_f64_slice_mut().expect("F64 scale")[index] = value;
                    } else if let Some(OrdinaryScale::F64(values)) = &mut self.ordinary {
                        values.push(value);
                    }
                }
                _ => unreachable!("preflighted floating scale"),
            }
        }
        let work = end - self.cursor;
        self.cursor = end;
        self.checkpoint(path)?;
        if end != count {
            return Ok((work.max(1), None));
        }
        let result = if let Some(exact) = self.exact.take() {
            exact
                .try_into_host_buffer()
                .map_err(|error| ValueCodecError::allocation(path, error.into()))?
        } else {
            match self.ordinary.take().expect("ordinary factor scale") {
                OrdinaryScale::F32(values) => {
                    HostNumericBuffer::from_numeric_storage(NumericStorage::F32(values))
                }
                OrdinaryScale::F64(values) => {
                    HostNumericBuffer::from_numeric_storage(NumericStorage::F64(values))
                }
            }
        };
        self.checkpoint(path)?;
        Ok((work.max(1), Some(result)))
    }

    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed scale runtime"))
                .map_err(|error| ValueCodecError::allocation(path, error))?;
        }
        Ok(())
    }
}
