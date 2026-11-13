//! Frozen matmul-epilogue owner, geometry, and shader before cold validation.
use super::super::super::{backend_types::BufferEntry, pipeline_recipe::PipelineRecipe};
use super::*;
use runmat_accelerate_api::{GpuTensorStorage, MatmulEpilogue};

pub(super) struct MatmulEpilogueFrame {
    pub(super) a: BufferEntry,
    pub(super) b: BufferEntry,
    pub(super) row: Option<BufferEntry>,
    pub(super) column: Option<BufferEntry>,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) len: usize,
    pub(super) params: crate::backend::wgpu::params::MatmulParams,
    pub(super) groups: [u32; 3],
    pub(super) recipe: PipelineRecipe,
}

fn native_real_input(
    provider: &WgpuProvider,
    handle: &GpuTensorHandle,
    entry: &BufferEntry,
) -> Result<()> {
    if entry.storage != GpuTensorStorage::Real
        || !provider.supports_native_real_element_type(entry.element_type)
        || entry.precision != provider.precision
        || provider.allocations.requires_logical_encoding(handle)?
    {
        return Err(runmat_accelerate_api::unsupported_provider_operation(
            "matmul_epilogue: native real floating owner required",
        ));
    }
    Ok(())
}

fn checked_epilogue_limits(
    limits: &wgpu::Limits,
    input_bytes: &[u64],
    output_bytes: u64,
    tile: u32,
    groups: [u32; 3],
    epilogue_uniform_bytes: u32,
) -> Result<()> {
    super::super::super::backend_shared::validate_compute_binding_counts(
        "matmul_epilogue",
        5,
        7,
        limits,
    )?;
    ensure!(
        limits.max_uniform_buffers_per_shader_stage >= 2
            && limits.max_uniform_buffer_binding_size
                >= std::mem::size_of::<crate::backend::wgpu::params::MatmulParams>() as u32
            && limits.max_uniform_buffer_binding_size >= epilogue_uniform_bytes,
        "matmul_epilogue: uniform binding unavailable"
    );
    ensure!(
        tile != 0
            && tile <= limits.max_compute_workgroup_size_x
            && tile <= limits.max_compute_workgroup_size_y
            && tile
                .checked_mul(tile)
                .is_some_and(|lanes| lanes <= limits.max_compute_invocations_per_workgroup),
        "matmul_epilogue: matrix tile exceeds enabled limits"
    );
    ensure!(
        groups[0] <= limits.max_compute_workgroups_per_dimension
            && groups[1] <= limits.max_compute_workgroups_per_dimension,
        "matmul_epilogue: dispatch exceeds enabled limits"
    );
    let max_bytes = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    ensure!(
        input_bytes.iter().all(|bytes| *bytes <= max_bytes) && output_bytes <= max_bytes,
        "matmul_epilogue: storage binding exceeds enabled limits"
    );
    Ok(())
}

impl MatmulEpilogueFrame {
    pub(super) fn plan(
        provider: &WgpuProvider,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        ep: &MatmulEpilogue,
    ) -> Result<Self> {
        let entry_a = provider.get_entry(a)?;
        let entry_b = provider.get_entry(b)?;
        native_real_input(provider, a, &entry_a)?;
        native_real_input(provider, b, &entry_b)?;
        ensure!(
            entry_a.shape.len() == 2 && entry_b.shape.len() == 2,
            "matmul_epilogue: only 2D tensors supported"
        );
        let view_a = build_matrix_operand_view(a, &entry_a)?;
        let view_b = build_matrix_operand_view(b, &entry_b)?;
        ensure!(
            view_a.cols == view_b.rows,
            "matmul_epilogue: inner dimensions must match"
        );
        ensure!(
            view_a.rows.checked_mul(view_a.cols) == Some(entry_a.len)
                && view_b.rows.checked_mul(view_b.cols) == Some(entry_b.len),
            "matmul_epilogue: input shape differs from physical storage"
        );
        if ep.diag_output.is_some() {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: diag_output is not supported by the WGPU provider yet",
            ));
        }
        let m = view_a.rows;
        let n = view_b.cols;
        let k = view_a.cols;
        let len = m
            .checked_mul(n)
            .ok_or_else(|| anyhow!("matmul_epilogue: output cardinality overflows"))?;
        let row = ep
            .row_scale
            .as_ref()
            .map(|handle| {
                let entry = provider.get_entry(handle)?;
                native_real_input(provider, handle, &entry)?;
                ensure!(
                    entry.len == m,
                    "matmul_epilogue: row scale length must equal output rows"
                );
                Ok::<_, anyhow::Error>(entry)
            })
            .transpose()?;
        let column = ep
            .col_scale
            .as_ref()
            .map(|handle| {
                let entry = provider.get_entry(handle)?;
                native_real_input(provider, handle, &entry)?;
                ensure!(
                    entry.len == n,
                    "matmul_epilogue: column scale length must equal output columns"
                );
                Ok::<_, anyhow::Error>(entry)
            })
            .transpose()?;
        if len == 0 {
            return Ok(Self {
                a: entry_a,
                b: entry_b,
                row,
                column,
                m,
                n,
                len,
                params: crate::backend::wgpu::params::MatmulParams {
                    m: 0,
                    n: 0,
                    k: 0,
                    lda: 0,
                    ldb: 0,
                    ldc: 0,
                    offset_a: 0,
                    offset_b: 0,
                    offset_out: 0,
                    flags: 0,
                },
                groups: [0, 0, 0],
                recipe: PipelineRecipe {
                    shader: String::new(),
                    layout: crate::backend::wgpu::bindings::MATMUL_EPILOGUE_LAYOUT_TAG.into(),
                    workgroup: 1,
                },
            });
        }
        let m_u32 = u32::try_from(m).map_err(|_| {
            runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: m exceeds GPU indexing limits",
            )
        })?;
        let n_u32 = u32::try_from(n).map_err(|_| {
            runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: n exceeds GPU indexing limits",
            )
        })?;
        let k_u32 = u32::try_from(k).map_err(|_| {
            runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: k exceeds GPU indexing limits",
            )
        })?;
        u32::try_from(len).map_err(|_| {
            runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: output exceeds GPU indexing limits",
            )
        })?;
        let mut flags = 0;
        if view_a.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_A;
        }
        if view_b.transpose {
            flags |= crate::backend::wgpu::params::MATMUL_FLAG_TRANSPOSE_B;
        }
        let params = crate::backend::wgpu::params::MatmulParams {
            m: m_u32,
            n: n_u32,
            k: k_u32,
            lda: view_a.lda,
            ldb: view_b.lda,
            ldc: m_u32,
            offset_a: 0,
            offset_b: 0,
            offset_out: 0,
            flags,
        };
        let geometry = provider.matmul_geometry().ok_or_else(|| {
            runmat_accelerate_api::unsupported_provider_operation(
                "matmul_epilogue: matrix geometry unavailable",
            )
        })?;
        let groups = geometry
            .dispatch_groups(m as u64, n as u64, false)
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(
                    "matmul_epilogue: dispatch geometry overflows",
                )
            })?;
        let tile = geometry.tile_size;
        let output_bytes = (len as u64)
            .checked_mul(provider.element_size as u64)
            .ok_or_else(|| anyhow!("matmul_epilogue: output byte extent overflows"))?;
        let mut input_bytes = vec![entry_a.allocated_bytes, entry_b.allocated_bytes];
        if let Some(entry) = &row {
            input_bytes.push(entry.allocated_bytes);
        }
        if let Some(entry) = &column {
            input_bytes.push(entry.allocated_bytes);
        }
        checked_epilogue_limits(
            &provider.adapter_limits,
            &input_bytes,
            output_bytes,
            tile,
            groups,
            match provider.precision {
                crate::backend::wgpu::types::NumericPrecision::F32 => std::mem::size_of::<
                    crate::backend::wgpu::params::MatmulEpilogueParamsF32,
                >() as u32,
                crate::backend::wgpu::types::NumericPrecision::F64 => std::mem::size_of::<
                    crate::backend::wgpu::params::MatmulEpilogueParamsF64,
                >() as u32,
            },
        )
        .map_err(|error| {
            runmat_accelerate_api::unsupported_provider_operation(error.to_string())
        })?;
        provider.require_ieee_words(provider.precision)?;
        let recipe = PipelineRecipe {
            shader: crate::backend::wgpu::shaders::matmul::epilogue_shader(
                provider.provider_precision_exec(),
                tile,
            ),
            layout: crate::backend::wgpu::bindings::MATMUL_EPILOGUE_LAYOUT_TAG.into(),
            workgroup: tile,
        };
        Ok(Self {
            a: entry_a,
            b: entry_b,
            row,
            column,
            m,
            n,
            len,
            params,
            groups,
            recipe,
        })
    }

    pub(super) fn require_live_inputs(
        &self,
        provider: &WgpuProvider,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        ep: &MatmulEpilogue,
    ) -> Result<()> {
        let required = [(a, &self.a), (b, &self.b)];
        for (handle, frozen) in required {
            let live = provider.get_entry(handle)?;
            ensure!(
                runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &frozen.buffer)
                    && !provider.allocations.requires_logical_encoding(handle)?,
                "matmul_epilogue: borrowed operand changed during preparation"
            );
        }
        for (handle, frozen) in [
            (ep.row_scale.as_ref(), self.row.as_ref()),
            (ep.col_scale.as_ref(), self.column.as_ref()),
        ] {
            if let (Some(handle), Some(frozen)) = (handle, frozen) {
                let live = provider.get_entry(handle)?;
                ensure!(
                    runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&live.buffer, &frozen.buffer)
                        && !provider.allocations.requires_logical_encoding(handle)?,
                    "matmul_epilogue: borrowed scale changed during preparation"
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn epilogue_limits_reject_binding_tile_and_dispatch_before_creation() {
        let mut limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 8,
            ..Default::default()
        };
        let ep_bytes =
            std::mem::size_of::<crate::backend::wgpu::params::MatmulEpilogueParamsF32>() as u32;
        assert!(checked_epilogue_limits(&limits, &[32, 32], 32, 8, [1, 1, 0], ep_bytes).is_ok());
        limits.max_storage_buffers_per_shader_stage = 4;
        assert!(checked_epilogue_limits(&limits, &[32, 32], 32, 8, [1, 1, 0], ep_bytes).is_err());
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_compute_workgroups_per_dimension = 1;
        assert!(checked_epilogue_limits(&limits, &[32, 32], 32, 8, [2, 1, 0], ep_bytes).is_err());
    }
}

#[cfg(test)]
#[path = "validated_epilogue/actual_tests.rs"]
mod actual_tests;
