//! Frozen triangular-solve inputs and recipe before scoped pipeline creation.
use super::*;

pub(super) struct TriangularSolveFrame {
    pub(super) lhs: super::super::backend_types::BufferEntry,
    pub(super) rhs: super::super::backend_types::BufferEntry,
    pub(super) rows: usize,
    pub(super) rhs_cols: usize,
    pub(super) len: usize,
    pub(super) recipe: super::super::pipeline_recipe::PipelineRecipe,
}

fn check_triangular_limits(
    limits: &wgpu::Limits,
    lhs_bytes: u64,
    rhs_bytes: u64,
    output_bytes: u64,
    workgroup: u32,
    len: u32,
) -> Result<()> {
    super::super::backend_shared::validate_compute_binding_counts(
        "linsolve_triangular",
        4,
        5,
        limits,
    )?;
    ensure!(
        limits.max_uniform_buffers_per_shader_stage >= 1
            && limits.max_uniform_buffer_binding_size
                >= std::mem::size_of::<crate::backend::wgpu::params::TriangularSolveParams>()
                    as u32,
        "linsolve: triangular uniform binding unavailable"
    );
    ensure!(
        workgroup != 0
            && workgroup <= limits.max_compute_workgroup_size_x
            && workgroup <= limits.max_compute_invocations_per_workgroup,
        "linsolve: triangular workgroup exceeds enabled limits"
    );
    let max_bytes = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    ensure!(
        lhs_bytes <= max_bytes && rhs_bytes <= max_bytes && output_bytes <= max_bytes,
        "linsolve: triangular storage binding exceeds enabled limits"
    );
    ensure!(
        crate::backend::wgpu::dispatch::common::dispatch_size(len, workgroup)
            <= limits.max_compute_workgroups_per_dimension,
        "linsolve: triangular dispatch exceeds enabled limits"
    );
    Ok(())
}

impl TriangularSolveFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        transpose: bool,
        lower: bool,
    ) -> Result<Self> {
        let lhs_entry = provider.get_entry(lhs)?;
        let rhs_entry = provider.get_entry(rhs)?;
        let (rows, cols) = WgpuProvider::matrix_dims_for_solve(lhs.shape())?;
        let (rhs_rows, rhs_cols) = WgpuProvider::matrix_dims_for_solve(rhs.shape())?;
        ensure!(
            rows == cols && rhs_rows == rows,
            "linsolve: triangular device path requires square lhs and matching rhs rows"
        );
        let len = rhs_rows
            .checked_mul(rhs_cols)
            .ok_or_else(|| anyhow!("linsolve: rhs dimensions overflow"))?;
        ensure!(
            rows.checked_mul(cols) == Some(lhs_entry.len) && len == rhs_entry.len,
            "linsolve: triangular input shape differs from physical storage"
        );
        if !(lhs_entry.storage == runmat_accelerate_api::GpuTensorStorage::Real
            && rhs_entry.storage == runmat_accelerate_api::GpuTensorStorage::Real
            && provider.supports_native_real_element_type(lhs_entry.element_type)
            && provider.supports_native_real_element_type(rhs_entry.element_type)
            && lhs_entry.precision == provider.precision
            && rhs_entry.precision == provider.precision
            && !provider.allocations.requires_logical_encoding(lhs)?
            && !provider.allocations.requires_logical_encoding(rhs)?)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "linsolve: triangular device path requires native real floating owners",
            ));
        }
        let workgroup = provider.workgroup_config.scalar;
        if len != 0 {
            u32::try_from(rows).map_err(|_| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "linsolve: triangular rows exceed GPU indexing limits",
                )
            })?;
            u32::try_from(rhs_cols).map_err(|_| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "linsolve: triangular RHS columns exceed GPU indexing limits",
                )
            })?;
            let len_u32 = u32::try_from(len).map_err(|_| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "linsolve: triangular output exceeds GPU indexing limits",
                )
            })?;
            let output_bytes = (len as u64)
                .checked_mul(provider.element_size as u64)
                .ok_or_else(|| anyhow!("linsolve: triangular output bytes overflow"))?;
            check_triangular_limits(
                &provider.adapter_limits,
                lhs_entry.allocated_bytes,
                rhs_entry.allocated_bytes,
                output_bytes,
                workgroup,
                len_u32,
            )
            .map_err(|error| {
                runmat_accelerate_api::unsupported_provider_operation(error.to_string())
            })?;
        }
        let recipe = super::super::pipeline_recipe::PipelineRecipe {
            shader: crate::backend::wgpu::shaders::solve::triangular_linsolve_shader(
                provider.precision,
                transpose,
                lower,
                workgroup,
            ),
            layout: crate::backend::wgpu::bindings::TRIANGULAR_SOLVE_LAYOUT_TAG.into(),
            workgroup,
        };
        Ok(Self {
            lhs: lhs_entry,
            rhs: rhs_entry,
            rows,
            rhs_cols,
            len,
            recipe,
        })
    }

    pub(super) fn require_live_inputs(
        &self,
        provider: &WgpuProvider,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) -> Result<()> {
        let live_lhs = provider.get_entry(lhs)?;
        let live_rhs = provider.get_entry(rhs)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live_lhs.buffer, &self.lhs.buffer)
                && runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                    &live_rhs.buffer,
                    &self.rhs.buffer,
                )
                && !provider.allocations.requires_logical_encoding(lhs)?
                && !provider.allocations.requires_logical_encoding(rhs)?,
            "linsolve: triangular borrowed operands changed during preparation"
        );
        Ok(())
    }
}

#[cfg(test)]
mod limits_tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn triangular_limits_follow_selected_workgroup_and_dispatch() {
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            ..Default::default()
        };
        assert!(check_triangular_limits(&limits, 32, 2064, 2064, 256, 258).is_ok());
        assert!(check_triangular_limits(&limits, 32, 2064, 2064, 128, 258).is_ok());
        assert!(check_triangular_limits(&limits, 32, 2064, 2064, 512, 258)
            .unwrap_err()
            .to_string()
            .contains("workgroup"));
        let mut no_storage = limits.clone();
        no_storage.max_storage_buffers_per_shader_stage = 3;
        assert!(
            check_triangular_limits(&no_storage, 32, 2064, 2064, 256, 258)
                .unwrap_err()
                .to_string()
                .contains("storage buffers")
        );
        let mut no_extent = limits.clone();
        no_extent.max_storage_buffer_binding_size = 2048;
        assert!(
            check_triangular_limits(&no_extent, 32, 2064, 2064, 256, 258)
                .unwrap_err()
                .to_string()
                .contains("storage binding")
        );
        let mut one_group = limits;
        one_group.max_compute_workgroups_per_dimension = 1;
        assert!(
            check_triangular_limits(&one_group, 32, 2064, 2064, 256, 258)
                .unwrap_err()
                .to_string()
                .contains("dispatch")
        );
    }
}

#[cfg(test)]
#[path = "validated_triangular/actual_tests.rs"]
mod actual_tests;
