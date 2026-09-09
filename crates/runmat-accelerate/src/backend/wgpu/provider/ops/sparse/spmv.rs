use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, GpuTensorHandle, GpuTensorStorage, SparseFeasibilityQuery,
    SparseOperation, SparseSpmvRequest, SparseTranspose,
};

use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};

pub(crate) struct SparseSpmvPipeline {
    layout: Arc<wgpu::BindGroupLayout>,
    pipeline: wgpu::ComputePipeline,
}

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
        let output_buffer =
            provider.create_storage_buffer_checked(row_count, "runmat-sparse-spmv-output")?;
        if row_count > 0 {
            let pipeline = pipeline(provider);
            let bind_group = provider
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-sparse-spmv-bind-group"),
                    layout: pipeline.layout.as_ref(),
                    entries: &[
                        binding(0, matrix.pattern.row_offsets.as_ref()),
                        binding(1, matrix.pattern.column_indices.as_ref()),
                        binding(2, matrix.values.as_ref()),
                        binding(3, input.buffer.as_ref()),
                        binding(4, output_buffer.as_ref()),
                    ],
                });
            let workgroup_size = provider.workgroup_config.scalar.max(1);
            let groups = u32::try_from(row_count)
                .ok()
                .and_then(|rows| rows.checked_add(workgroup_size - 1))
                .map(|rows| rows / workgroup_size)
                .ok_or_else(|| anyhow!("sparse SpMV dispatch count exceeds WGPU limits"))?;
            ensure!(
                groups <= provider.adapter_limits.max_compute_workgroups_per_dimension,
                "sparse SpMV dispatch exceeds the adapter workgroup limit"
            );
            let mut encoder =
                provider
                    .device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sparse-spmv-encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-sparse-spmv-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            provider.submit(encoder);
            provider.record_kernel_launch_basic(
                "sparse_spmv",
                &[
                    ("rows", matrix.pattern.descriptor.rows),
                    ("columns", matrix.pattern.descriptor.columns),
                    ("nnz", matrix.pattern.descriptor.nnz),
                ],
                &[("workgroup_size", u64::from(workgroup_size))],
            );
        }
        let allocated_bytes = output_buffer.size();
        Ok(provider.register_numeric_buffer(
            output_buffer,
            NumericBufferRegistration {
                shape: vec![row_count],
                len: row_count,
                physical_element_type: matrix.descriptor.element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        ))
    })
}

fn pipeline(provider: &WgpuProvider) -> Arc<SparseSpmvPipeline> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .spmv
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create_pipeline(provider)))
        .clone()
}

fn create_pipeline(provider: &WgpuProvider) -> SparseSpmvPipeline {
    let entries = [
        crate::backend::wgpu::bindings::storage_read_entry(0),
        crate::backend::wgpu::bindings::storage_read_entry(1),
        crate::backend::wgpu::bindings::storage_read_entry(2),
        crate::backend::wgpu::bindings::storage_read_entry(3),
        crate::backend::wgpu::bindings::storage_read_write_entry(4),
    ];
    let layout = Arc::new(provider.device_ref().create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("runmat-sparse-spmv-layout"),
            entries: &entries,
        },
    ));
    let pipeline_layout =
        provider
            .device_ref()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("runmat-sparse-spmv-pipeline-layout"),
                bind_group_layouts: &[layout.as_ref()],
                push_constant_ranges: &[],
            });
    let source = shader_source(
        provider.precision.as_str(),
        provider.workgroup_config.scalar.max(1),
    );
    let module = provider
        .device_ref()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("runmat-sparse-spmv-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    let pipeline =
        provider
            .device_ref()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("runmat-sparse-spmv-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });
    SparseSpmvPipeline { layout, pipeline }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn shader_source(scalar: &str, workgroup_size: u32) -> String {
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
