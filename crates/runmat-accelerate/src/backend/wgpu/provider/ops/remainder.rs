//! Exact-class floating remainder on the provider's validated native layouts.
use super::backend_types::WgpuProvider;
use crate::backend::wgpu::types::NumericPrecision;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_types::FloatingRemainderKind;

impl WgpuProvider {
    pub(crate) fn floating_remainder_exec(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        kind: FloatingRemainderKind,
    ) -> Result<GpuTensorHandle> {
        for handle in [a, b] {
            let entry = self.get_entry(handle)?;
            let physical = match self.precision {
                NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
            };
            if entry.element_type != physical {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "floating remainder requires the provider's native physical precision",
                ));
            }
            ensure!(
                entry.integer_type().is_none() && !runmat_accelerate_api::handle_is_logical(handle),
                "floating remainder requires floating operands"
            );
            ensure!(
                handle.descriptor().element_type == Some(entry.element_type)
                    && runmat_accelerate_api::handle_class_identity(handle)
                        .is_none_or(|class| class == entry.element_type.class_identity()),
                "floating remainder class metadata mismatch"
            );
            ensure!(
                entry.storage == GpuTensorStorage::Real
                    && handle.descriptor().storage == Some(GpuTensorStorage::Real),
                "floating remainder requires real storage"
            );
            ensure!(
                entry.shape == handle.shape(),
                "floating remainder shape metadata mismatch"
            );
            let count = checked_count(handle.shape())?;
            ensure!(
                entry.len == count,
                "floating remainder storage length mismatch"
            );
        }
        let shape = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
            "remainder",
            a.shape(),
            b.shape(),
        )
        .map_err(|error| anyhow!(error))?;
        ensure!(
            shape.len() <= crate::backend::wgpu::params::BCAST_MAX_RANK,
            "remainder exceeds provider rank limit"
        );
        let count = checked_count(&shape)?;
        ensure!(
            count <= u32::MAX as usize && shape.iter().all(|extent| *extent <= u32::MAX as usize),
            "remainder exceeds provider indexing limit"
        );
        for handle in [a, b] {
            super::backend_shared::fusion_operand_layout(
                handle,
                &self.get_entry_raw(handle)?,
                &shape,
            )?;
        }
        if count == 0 {
            let buffer = self.with_emission(|emission, _| {
                self.create_storage_buffer(emission, 0, "remainder-empty")
            })??;
            return self.register_existing_buffer_with_storage(
                buffer,
                shape,
                0,
                GpuTensorStorage::Real,
            );
        }
        self.require_ieee_words(self.precision)?;
        self.fused_elementwise_with_telemetry_and_storage_exec(
            &crate::backend::wgpu::shaders::remainder::shader(self.precision, kind),
            &[a.clone(), b.clone()],
            &shape,
            count,
            GpuTensorStorage::Real,
        )
    }
}

fn checked_count(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| anyhow!("remainder element count overflow"))
}
