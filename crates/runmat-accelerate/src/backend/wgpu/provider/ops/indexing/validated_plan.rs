//! Floating scatter facts frozen before transpose materialization or scoped compilation.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum ScatterAxis {
    Column,
    Row,
}

pub(super) struct ScatterPlan {
    matrix: super::super::backend_types::BufferEntry,
    values: super::super::backend_types::BufferEntry,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) recipe: super::super::pipeline_recipe::PipelineRecipe,
    pub(super) kind: super::super::validated_fusion::IndexingPipelineKind,
}

impl ScatterPlan {
    pub(super) fn new(
        provider: &WgpuProvider,
        matrix: &GpuTensorHandle,
        values: &GpuTensorHandle,
        index: usize,
        axis: ScatterAxis,
    ) -> Result<Option<Self>> {
        let matrix_entry = provider.get_entry_raw(matrix)?;
        let values_entry = provider.get_entry_raw(values)?;
        for (handle, entry) in [(matrix, &matrix_entry), (values, &values_entry)] {
            if let Some(info) = runmat_accelerate_api::handle_transpose_info(handle) {
                ensure!(
                    entry.shape.len() == 2
                        && entry.shape[0] == info.base_cols
                        && entry.shape[1] == info.base_rows,
                    "scatter: transpose metadata mismatch before preparation"
                );
            }
        }
        if matrix_entry.integer_type().is_some() {
            ensure!(
                matrix_entry.shape.len() == 2,
                "scatter: only 2D tensors supported"
            );
            let rows = matrix_entry.shape[0];
            let cols = matrix_entry.shape[1];
            let valid = match axis {
                ScatterAxis::Column => {
                    index < cols
                        && (values_entry.shape.as_slice() == [rows]
                            || values_entry.shape.as_slice() == [rows, 1])
                        && values_entry.len == rows
                }
                ScatterAxis::Row => {
                    index < rows
                        && (values_entry.shape.as_slice() == [cols]
                            || values_entry.shape.as_slice() == [1, cols])
                        && values_entry.len == cols
                }
            };
            ensure!(valid, "scatter: integer index or value shape mismatch");
            ensure!(
                values_entry.integer_type() == matrix_entry.integer_type(),
                "scatter: integer storage mismatch"
            );
            ensure!(
                rows.checked_mul(cols)
                    .is_some_and(|length| length <= u32::MAX as usize),
                "scatter: integer output index exceeds GPU limits"
            );
            return Ok(None);
        }
        if matrix.view().is_logical() && !provider.allocations.requires_logical_encoding(values)? {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "logical scatter row/column requires canonical logical values",
            ));
        }
        let label = match axis {
            ScatterAxis::Column => "scatter_column",
            ScatterAxis::Row => "scatter_row",
        };
        if matrix_entry.precision != NumericPrecision::F32
            || values_entry.precision != NumericPrecision::F32
            || matrix_entry.storage != GpuTensorStorage::Real
            || values_entry.storage != GpuTensorStorage::Real
            || matrix_entry.integer_type().is_some()
            || values_entry.integer_type().is_some()
            || !provider.supports_native_real_element_type(matrix_entry.element_type)
            || !provider.supports_native_real_element_type(values_entry.element_type)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                format!("{label}: floating kernel requires real F32 storage"),
            ));
        }
        ensure!(
            matrix_entry.shape.len() == 2,
            "{label}: only 2D tensors supported"
        );
        let rows = matrix_entry.shape[0];
        let cols = matrix_entry.shape[1];
        let (extent, kind, layout, shader) = match axis {
            ScatterAxis::Column => {
                ensure!(index < cols, "scatter_column: column index out of bounds");
                ensure!(
                    values_entry.shape.as_slice() == [rows]
                        || values_entry.shape.as_slice() == [rows, 1],
                    "scatter_column: values must be a length-rows vector"
                );
                (
                    rows,
                    super::super::validated_fusion::IndexingPipelineKind::ScatterColumn,
                    "runmat-scatter-col-bgl",
                    crate::backend::wgpu::shaders::scatter::SCATTER_COL_SHADER_F32,
                )
            }
            ScatterAxis::Row => {
                ensure!(index < rows, "scatter_row: row index out of bounds");
                ensure!(
                    values_entry.shape.as_slice() == [cols]
                        || values_entry.shape.as_slice() == [1, cols],
                    "scatter_row: values must be a length-cols vector"
                );
                (
                    cols,
                    super::super::validated_fusion::IndexingPipelineKind::ScatterRow,
                    "runmat-scatter-row-bgl",
                    crate::backend::wgpu::shaders::scatter::SCATTER_ROW_SHADER_F32,
                )
            }
        };
        ensure!(values_entry.len == extent, "{label}: length mismatch");
        let output_len = rows.checked_mul(cols).ok_or_else(|| {
            runmat_accelerate_api::unsupported_provider_operation(format!(
                "{label}: output shape exceeds GPU limits"
            ))
        })?;
        let output_bytes = (output_len as u64)
            .checked_mul(provider.element_size as u64)
            .ok_or_else(|| {
                runmat_accelerate_api::unsupported_provider_operation(format!(
                    "{label}: output bytes exceed GPU limits"
                ))
            })?;
        let workgroup = provider.workgroup_config.scalar;
        let limits = &provider.adapter_limits;
        let max_binding =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        super::super::backend_shared::validate_compute_binding_counts(label, 3, 4, limits)
            .map_err(|error| {
                runmat_accelerate_api::unsupported_provider_operation(error.to_string())
            })?;
        if rows > u32::MAX as usize
            || cols > u32::MAX as usize
            || index > u32::MAX as usize
            || output_len > u32::MAX as usize
            || workgroup == 0
            || workgroup > limits.max_compute_workgroup_size_x
            || workgroup > limits.max_compute_invocations_per_workgroup
            || limits.max_uniform_buffers_per_shader_stage == 0
            || limits.max_uniform_buffer_binding_size < 12
            || matrix_entry.allocated_bytes > max_binding
            || values_entry.allocated_bytes > max_binding
            || output_bytes > max_binding
            || crate::backend::wgpu::dispatch::common::dispatch_size(extent as u32, workgroup)
                > limits.max_compute_workgroups_per_dimension
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                format!("{label}: enabled device limits do not admit the scatter recipe"),
            ));
        }
        Ok(Some(Self {
            matrix: matrix_entry,
            values: values_entry,
            rows,
            cols,
            recipe: super::super::pipeline_recipe::PipelineRecipe {
                shader: shader.replace("@WG@", &workgroup.to_string()),
                layout: layout.into(),
                workgroup,
            },
            kind,
        }))
    }

    pub(super) fn require_live_inputs(
        &self,
        provider: &WgpuProvider,
        matrix: &GpuTensorHandle,
        values: &GpuTensorHandle,
    ) -> Result<()> {
        let matrix_now = provider.get_entry_raw(matrix)?;
        let values_now = provider.get_entry_raw(values)?;
        ensure!(
            runmat_accelerate_api::WgpuBufferOwner::ptr_eq(&matrix_now.buffer, &self.matrix.buffer)
                && runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
                    &values_now.buffer,
                    &self.values.buffer
                )
                && (!matrix.view().is_logical()
                    || provider.allocations.requires_logical_encoding(values)?),
            "scatter: borrowed input owner changed during preparation"
        );
        Ok(())
    }
}
