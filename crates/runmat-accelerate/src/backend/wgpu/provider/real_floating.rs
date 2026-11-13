//! Selected physical precision for real floating kernels.
//!
//! The provider's preferred precision is not authority to reinterpret an
//! existing allocation. An F64 device may execute an F32 kernel only after
//! the exact live view and physical element type agree.

use super::{
    backend_types::BufferEntry, backend_types::NumericBufferRegistration,
    backend_types::WgpuProvider,
};
use crate::backend::wgpu::{residency::BufferUsageClass, types::NumericPrecision};
use anyhow::{ensure, Result};
use runmat_accelerate_api::{
    unsupported_provider_operation, GpuTensorHandle, GpuTensorStorage, NumericElementType,
    WgpuBufferOwner,
};

fn requires_signed_zero_atan2_kernel(op: crate::backend::wgpu::types::BinaryOpCode) -> bool {
    use crate::backend::wgpu::types::BinaryOpCode;
    match op {
        BinaryOpCode::Atan2 => true,
        BinaryOpCode::Add
        | BinaryOpCode::Sub
        | BinaryOpCode::Mul
        | BinaryOpCode::Div
        | BinaryOpCode::Hypot
        | BinaryOpCode::Pow
        | BinaryOpCode::Max
        | BinaryOpCode::Min => false,
    }
}

impl WgpuProvider {
    pub(super) fn real_floating_binary_inputs(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> Result<(BufferEntry, BufferEntry)> {
        let left = self.real_floating_entry(left)?;
        let right = self.real_floating_entry(right)?;
        ensure!(
            left.precision == right.precision,
            "real binary operands require the same physical precision"
        );
        Ok((left, right))
    }

    pub(super) fn real_floating_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        let entry = self.get_entry_raw(handle)?;
        if entry.precision == self.precision {
            return self.get_entry(handle);
        }
        ensure!(
            entry.storage == GpuTensorStorage::Real
                && handle.descriptor().storage == Some(GpuTensorStorage::Real)
                && !runmat_accelerate_api::handle_is_logical(handle)
                && entry.integer_type().is_none()
                && handle.descriptor().element_type == Some(entry.element_type),
            "real floating kernel requires an exact native real floating view"
        );
        let expected = match entry.precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        ensure!(
            entry.element_type == expected,
            "real floating kernel physical type does not match its precision"
        );
        if entry.precision != self.precision
            && !(self.precision == NumericPrecision::F64
                && entry.precision == NumericPrecision::F32)
        {
            return Err(unsupported_provider_operation(
                "real floating kernel requires an available physical precision",
            ));
        }
        Ok(entry)
    }

    pub(super) fn real_binary_pipeline(
        &self,
        precision: NumericPrecision,
        broadcast: bool,
        op: crate::backend::wgpu::types::BinaryOpCode,
    ) -> Result<&crate::backend::wgpu::pipelines::PipelineBundle> {
        if op == crate::backend::wgpu::types::BinaryOpCode::Max {
            self.require_ieee_words(precision)?;
            if precision == self.precision {
                return Ok(if broadcast {
                    &self.pipelines.binary_broadcast_max
                } else {
                    &self.pipelines.binary_max
                });
            }
            let alternate = self.f32_real_floating.as_ref().ok_or_else(|| {
                unsupported_provider_operation(
                    "F32 real binary Max kernel is unavailable on this provider",
                )
            })?;
            return Ok(if broadcast {
                &alternate.binary_broadcast_max
            } else {
                &alternate.binary_max
            });
        }
        let signed_zero_atan2 = requires_signed_zero_atan2_kernel(op);
        if precision == self.precision {
            if precision == NumericPrecision::F64 && signed_zero_atan2 {
                return (if broadcast {
                    self.pipelines.binary_broadcast_atan2.as_ref()
                } else {
                    self.pipelines.binary_atan2.as_ref()
                })
                .ok_or_else(|| {
                    unsupported_provider_operation("selected F64 atan2 kernel is unavailable")
                });
            }
            return Ok(if broadcast {
                &self.pipelines.binary_broadcast
            } else {
                &self.pipelines.binary
            });
        }
        let alternate = self.f32_real_floating.as_ref().ok_or_else(|| {
            unsupported_provider_operation("F32 real binary kernel is unavailable on this provider")
        })?;
        Ok(if broadcast {
            &alternate.binary_broadcast
        } else {
            &alternate.binary
        })
    }

    pub(super) fn real_scalar_pipeline(
        &self,
        precision: NumericPrecision,
    ) -> Result<&crate::backend::wgpu::pipelines::PipelineBundle> {
        if precision == self.precision {
            return Ok(&self.pipelines.scalar);
        }
        self.f32_real_floating
            .as_ref()
            .map(|alternate| &alternate.scalar)
            .ok_or_else(|| {
                unsupported_provider_operation(
                    "F32 real scalar kernel is unavailable on this provider",
                )
            })
    }

    /// Both ordinary execution and physical proposals must select this owner.
    /// Max's fixed IEEE policy requires enabled INT64 only for native F64.
    pub(super) fn real_scalar_pipeline_for_operation(
        &self,
        op: crate::backend::wgpu::types::ScalarOpCode,
        precision: NumericPrecision,
    ) -> Result<&crate::backend::wgpu::pipelines::PipelineBundle> {
        use crate::backend::wgpu::types::ScalarOpCode;
        let pipeline = match op {
            ScalarOpCode::Max => {
                self.require_ieee_words(precision)?;
                if precision == self.precision {
                    &self.pipelines.scalar_max
                } else {
                    &self
                        .f32_real_floating
                        .as_ref()
                        .ok_or_else(|| {
                            unsupported_provider_operation(
                                "F32 real scalar Max kernel is unavailable",
                            )
                        })?
                        .scalar_max
                }
            }
            ScalarOpCode::Add
            | ScalarOpCode::Sub
            | ScalarOpCode::Mul
            | ScalarOpCode::Div
            | ScalarOpCode::RSub
            | ScalarOpCode::RDiv
            | ScalarOpCode::Min => self.real_scalar_pipeline(precision)?,
        };
        pipeline.preflight()?;
        Ok(pipeline)
    }

    pub(super) fn real_nd_pipeline(
        &self,
        precision: NumericPrecision,
        moments: bool,
    ) -> Result<&crate::backend::wgpu::pipelines::PipelineBundle> {
        if precision == self.precision {
            return Ok(if moments {
                &self.pipelines.reduce_nd_moments
            } else {
                &self.pipelines.reduce_nd_mean
            });
        }
        let alternate = self.f32_real_floating.as_ref().ok_or_else(|| {
            unsupported_provider_operation("F32 real N-D reduction kernel is unavailable")
        })?;
        Ok(if moments {
            &alternate.reduce_nd_moments
        } else {
            &alternate.reduce_nd_mean
        })
    }

    pub(super) fn register_real_floating_output(
        &self,
        buffer: WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        precision: NumericPrecision,
    ) -> Result<GpuTensorHandle> {
        self.register_real_floating_output_admitted(buffer, shape, len, precision, None)
    }

    pub(super) fn register_real_floating_output_admitted(
        &self,
        buffer: WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        precision: NumericPrecision,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let physical_element_type = match precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        let requested_bytes = u64::try_from(len)?
            .checked_mul(physical_element_type.element_size() as u64)
            .ok_or_else(|| anyhow::anyhow!("real floating output byte extent overflow"))?;
        let allocated_bytes = buffer.size();
        ensure!(
            allocated_bytes >= requested_bytes,
            "real floating output allocation is smaller than its physical type"
        );
        self.register_numeric_buffer_produced_admitted(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
            false,
            admission,
        )
    }

    pub(super) fn validate_real_floating_output_extent(
        &self,
        len: usize,
        precision: NumericPrecision,
    ) -> Result<u64> {
        let element_bytes = match precision {
            NumericPrecision::F32 => 4u64,
            NumericPrecision::F64 => 8u64,
        };
        let bytes = u64::try_from(len.max(1))?
            .checked_mul(element_bytes)
            .ok_or_else(|| anyhow::anyhow!("real floating output byte extent overflow"))?;
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(unsupported_provider_operation(
                "real floating output exceeds the selected device buffer limit",
            ));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod ieee_route_tests {
    use super::*;
    use crate::backend::wgpu::types::BinaryOpCode;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn only_atan2_uses_the_integer_word_binary_route() {
        assert!(requires_signed_zero_atan2_kernel(BinaryOpCode::Atan2));
        for op in [
            BinaryOpCode::Add,
            BinaryOpCode::Sub,
            BinaryOpCode::Mul,
            BinaryOpCode::Div,
            BinaryOpCode::Hypot,
            BinaryOpCode::Pow,
            BinaryOpCode::Max,
            BinaryOpCode::Min,
        ] {
            assert!(!requires_signed_zero_atan2_kernel(op));
        }
    }
}
