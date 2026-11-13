use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, GpuTensorHandle, GpuTensorStorage, SparseFeasibilityQuery,
    SparseOperation, SparseSpmvRequest, SparseTranspose,
};

use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};

pub(crate) fn execute<'a>(
    provider: &'a WgpuProvider,
    request: &'a SparseSpmvRequest<'a>,
) -> AccelProviderFuture<'a, GpuTensorHandle> {
    Box::pin(async move {
        request.validate(provider.runtime_device_id)?;
        ensure!(
            request.transpose == SparseTranspose::None,
            "WGPU sparse provider does not support transpose SpMV"
        );
        let matrix = provider.sparse.matrix(provider, request.matrix)?;
        let input = provider.get_entry(request.input)?;
        ensure!(
            input.storage == GpuTensorStorage::Real
                && input.element_type == matrix.descriptor.element_type,
            "WGPU sparse SpMV input representation does not match the matrix"
        );
        let feasibility = super::feasibility(
            provider,
            &SparseFeasibilityQuery {
                operation: SparseOperation::Spmv(request.transpose),
                pattern: matrix.pattern.descriptor,
                element_type: matrix.descriptor.element_type,
                storage: matrix.descriptor.storage,
                vector_count: 1,
                vector_length: matrix.pattern.descriptor.columns,
                max_iterations: None,
            },
        );
        ensure!(
            feasibility.is_supported(),
            "WGPU sparse SpMV exceeds the advertised provider envelope"
        );
        ensure!(
            input.buffer.size() <= super::storage_binding_limit(provider),
            "WGPU sparse SpMV input exceeds the storage-buffer binding limit"
        );
        let row_count = usize::try_from(matrix.pattern.descriptor.rows)
            .map_err(|_| anyhow!("sparse row count exceeds host address space"))?;
        let groups = super::dispatch_groups(provider, row_count)?;
        let output_buffer = provider.with_emission(|emission, pending| {
            let output_buffer = provider.create_storage_buffer_checked(
                emission,
                row_count,
                "runmat-sparse-spmv-output",
            )?;
            if row_count > 0 {
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sparse-spmv-encoder"),
                    },
                );
                super::dispatch(
                    emission,
                    &mut encoder,
                    &provider.sparse_pipelines.spmv,
                    "runmat-sparse-spmv-pass",
                    &[
                        matrix.pattern.row_offsets.as_ref(),
                        matrix.pattern.column_indices.as_ref(),
                        matrix.values.as_ref(),
                        input.buffer.as_ref(),
                        output_buffer.as_ref(),
                    ],
                    groups,
                )?;
                provider.submit(emission, pending, encoder)?;
            }
            anyhow::Ok(output_buffer)
        })??;
        if row_count > 0 {
            provider.record_kernel_launch_basic(
                "sparse_spmv",
                &[
                    ("rows", matrix.pattern.descriptor.rows),
                    ("columns", matrix.pattern.descriptor.columns),
                    ("nnz", matrix.pattern.descriptor.nnz),
                ],
                &[(
                    "workgroup_size",
                    u64::from(provider.workgroup_config.scalar.max(1)),
                )],
            );
        }
        let allocated_bytes = output_buffer.size();
        provider.register_numeric_buffer(
            output_buffer,
            NumericBufferRegistration {
                shape: vec![row_count],
                len: row_count,
                physical_element_type: matrix.descriptor.element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )
    })
}

pub(super) fn shader_source(scalar: &str, workgroup_size: u32) -> String {
    format!(
        r#"
struct IndexBuffer {{ data: array<u32>, }};
struct ScalarBuffer {{ data: array<{scalar}>, }};

@group(0) @binding(0) var<storage, read> RowOffsets: IndexBuffer;
@group(0) @binding(1) var<storage, read> ColumnIndices: IndexBuffer;
@group(0) @binding(2) var<storage, read> Values: ScalarBuffer;
@group(0) @binding(3) var<storage, read> Input: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Output: ScalarBuffer;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let rows = arrayLength(&RowOffsets.data) - 1u;
    if (row >= rows) {{ return; }}
    let start = RowOffsets.data[row];
    let end = RowOffsets.data[row + 1u];
    var sum: {scalar} = 0.0;
    for (var index = start; index < end; index = index + 1u) {{
        sum = sum + Values.data[index] * Input.data[ColumnIndices.data[index]];
    }}
    Output.data[row] = sum;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::shader_source;

    #[test]
    fn sparse_spmv_shader_is_precision_and_workgroup_specific() {
        let f32_shader = shader_source("f32", 64);
        assert!(f32_shader.contains("array<f32>"));
        assert!(f32_shader.contains("@workgroup_size(64)"));
        let f64_shader = shader_source("f64", 128);
        assert!(f64_shader.contains("array<f64>"));
        assert!(f64_shader.contains("@workgroup_size(128)"));
    }
}
