use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, GpuTensorHandle, GpuTensorStorage, SparseFeasibilityQuery,
    SparseOperation, SparsePreconditionerApplyRequest, SparsePreconditionerDescriptor,
    SparsePreconditionerHandle, SparsePreconditionerKind, SparsePreconditionerRequest,
};
use wgpu::util::DeviceExt;

use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};
use super::storage::DeviceSparsePreconditioner;

pub(crate) struct SparseJacobiPreparePipeline {
    layout: Arc<wgpu::BindGroupLayout>,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) struct SparseJacobiApplyPipeline {
    layout: Arc<wgpu::BindGroupLayout>,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) fn prepare<'a>(
    provider: &'a WgpuProvider,
    request: &'a SparsePreconditionerRequest<'a>,
) -> AccelProviderFuture<'a, SparsePreconditionerHandle> {
    Box::pin(async move {
        request.validate(provider.runtime_device_id)?;
        ensure!(
            request.kind == SparsePreconditionerKind::Jacobi,
            "WGPU sparse provider supports Jacobi preparation only"
        );
        let matrix = provider.sparse.matrix(provider, request.matrix)?;
        ensure!(
            super::feasibility(
                provider,
                &SparseFeasibilityQuery {
                    operation: SparseOperation::PreparePreconditioner(request.kind),
                    pattern: matrix.pattern.descriptor,
                    element_type: matrix.descriptor.element_type,
                    storage: matrix.descriptor.storage,
                    vector_count: 0,
                    vector_length: 0,
                    max_iterations: None,
                },
            )
            .is_supported(),
            "WGPU Jacobi preparation exceeds the advertised provider envelope"
        );
        let dimension = usize::try_from(matrix.pattern.descriptor.rows)
            .map_err(|_| anyhow!("Jacobi dimension exceeds host address space"))?;
        let inverse_diagonal =
            provider.create_storage_buffer_checked(dimension, "runmat-sparse-jacobi-inverse")?;
        let status = Arc::new(provider.device_ref().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("runmat-sparse-jacobi-status"),
                contents: bytemuck::cast_slice(&[0_u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        ));
        if dimension > 0 {
            let pipeline = prepare_pipeline(provider);
            let bind_group = provider
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-sparse-jacobi-prepare-bind-group"),
                    layout: pipeline.layout.as_ref(),
                    entries: &[
                        binding(0, matrix.pattern.row_offsets.as_ref()),
                        binding(1, matrix.pattern.column_indices.as_ref()),
                        binding(2, matrix.values.as_ref()),
                        binding(3, inverse_diagonal.as_ref()),
                        binding(4, status.as_ref()),
                    ],
                });
            let groups = dispatch_groups(provider, dimension)?;
            let staging =
                provider.create_readback_buffer(4, "runmat-sparse-jacobi-status-readback");
            let mut encoder =
                provider
                    .device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sparse-jacobi-prepare-encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-sparse-jacobi-prepare-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(status.as_ref(), 0, staging.as_ref(), 0, 4);
            provider.submit(encoder);
            let bytes = provider
                .map_readback_bytes(staging, 4, "sparse Jacobi preparation")
                .await?;
            let state = u32::from_le_bytes(bytes[..4].try_into().expect("four status bytes"));
            ensure!(
                state == 0,
                "sparse Jacobi preparation requires one finite nonzero diagonal entry per row"
            );
            provider.record_kernel_launch_basic(
                "sparse_jacobi_prepare",
                &[
                    ("rows", matrix.pattern.descriptor.rows),
                    ("nnz", matrix.pattern.descriptor.nnz),
                ],
                &[(
                    "workgroup_size",
                    u64::from(provider.workgroup_config.scalar.max(1)),
                )],
            );
        }
        let preconditioner_id = provider
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let descriptor = SparsePreconditionerDescriptor {
            matrix_id: request.matrix.matrix_id,
            kind: request.kind,
            dimension: matrix.pattern.descriptor.rows,
            element_type: matrix.descriptor.element_type,
            storage: matrix.descriptor.storage,
        };
        provider.sparse.insert_preconditioner(
            preconditioner_id,
            DeviceSparsePreconditioner {
                descriptor,
                pattern: matrix.pattern.descriptor,
                inverse_diagonal,
            },
        )?;
        Ok(SparsePreconditionerHandle {
            device_id: provider.runtime_device_id,
            preconditioner_id,
            descriptor,
        })
    })
}

pub(crate) fn apply<'a>(
    provider: &'a WgpuProvider,
    request: &'a SparsePreconditionerApplyRequest<'a>,
) -> AccelProviderFuture<'a, GpuTensorHandle> {
    Box::pin(async move {
        request.validate(provider.runtime_device_id)?;
        let preconditioner = provider
            .sparse
            .preconditioner(provider, request.preconditioner)?;
        let input = provider.get_entry(request.input)?;
        ensure!(
            input.storage == GpuTensorStorage::Real
                && input.element_type == preconditioner.descriptor.element_type,
            "WGPU Jacobi input representation does not match its preconditioner"
        );
        ensure!(
            super::feasibility(
                provider,
                &SparseFeasibilityQuery {
                    operation: SparseOperation::ApplyPreconditioner(
                        preconditioner.descriptor.kind,
                    ),
                    pattern: preconditioner.pattern,
                    element_type: preconditioner.descriptor.element_type,
                    storage: preconditioner.descriptor.storage,
                    vector_count: 1,
                    vector_length: preconditioner.descriptor.dimension,
                    max_iterations: None,
                },
            )
            .is_supported(),
            "WGPU Jacobi application exceeds the advertised provider envelope"
        );
        let dimension = usize::try_from(preconditioner.descriptor.dimension)
            .map_err(|_| anyhow!("Jacobi dimension exceeds host address space"))?;
        let output =
            provider.create_storage_buffer_checked(dimension, "runmat-sparse-jacobi-output")?;
        if dimension > 0 {
            let pipeline = apply_pipeline(provider);
            let bind_group = provider
                .device_ref()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("runmat-sparse-jacobi-apply-bind-group"),
                    layout: pipeline.layout.as_ref(),
                    entries: &[
                        binding(0, preconditioner.inverse_diagonal.as_ref()),
                        binding(1, input.buffer.as_ref()),
                        binding(2, output.as_ref()),
                    ],
                });
            let mut encoder =
                provider
                    .device_ref()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sparse-jacobi-apply-encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runmat-sparse-jacobi-apply-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_groups(provider, dimension)?, 1, 1);
            }
            provider.submit(encoder);
            provider.record_kernel_launch_basic(
                "sparse_jacobi_apply",
                &[("rows", preconditioner.descriptor.dimension)],
                &[(
                    "workgroup_size",
                    u64::from(provider.workgroup_config.scalar.max(1)),
                )],
            );
        }
        Ok(provider.register_numeric_buffer(
            output.clone(),
            NumericBufferRegistration {
                shape: request.input.shape.clone(),
                len: dimension,
                physical_element_type: preconditioner.descriptor.element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes: output.size(),
                usage: BufferUsageClass::Generic,
            },
        ))
    })
}

fn prepare_pipeline(provider: &WgpuProvider) -> Arc<SparseJacobiPreparePipeline> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .jacobi_prepare
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create_prepare_pipeline(provider)))
        .clone()
}

fn apply_pipeline(provider: &WgpuProvider) -> Arc<SparseJacobiApplyPipeline> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .jacobi_apply
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create_apply_pipeline(provider)))
        .clone()
}

fn create_prepare_pipeline(provider: &WgpuProvider) -> SparseJacobiPreparePipeline {
    let entries = [
        crate::backend::wgpu::bindings::storage_read_entry(0),
        crate::backend::wgpu::bindings::storage_read_entry(1),
        crate::backend::wgpu::bindings::storage_read_entry(2),
        crate::backend::wgpu::bindings::storage_read_write_entry(3),
        crate::backend::wgpu::bindings::storage_read_write_entry(4),
    ];
    let (layout, pipeline) = create_pipeline(
        provider,
        "runmat-sparse-jacobi-prepare",
        &entries,
        prepare_shader(provider),
    );
    SparseJacobiPreparePipeline { layout, pipeline }
}

fn create_apply_pipeline(provider: &WgpuProvider) -> SparseJacobiApplyPipeline {
    let entries = [
        crate::backend::wgpu::bindings::storage_read_entry(0),
        crate::backend::wgpu::bindings::storage_read_entry(1),
        crate::backend::wgpu::bindings::storage_read_write_entry(2),
    ];
    let (layout, pipeline) = create_pipeline(
        provider,
        "runmat-sparse-jacobi-apply",
        &entries,
        apply_shader(provider),
    );
    SparseJacobiApplyPipeline { layout, pipeline }
}

fn create_pipeline(
    provider: &WgpuProvider,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
    source: String,
) -> (Arc<wgpu::BindGroupLayout>, wgpu::ComputePipeline) {
    let layout = Arc::new(provider.device_ref().create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        },
    ));
    let pipeline_layout =
        provider
            .device_ref()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout.as_ref()],
                push_constant_ranges: &[],
            });
    let module = provider
        .device_ref()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    let pipeline =
        provider
            .device_ref()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });
    (layout, pipeline)
}

fn prepare_shader(provider: &WgpuProvider) -> String {
    let scalar = provider.precision.as_str();
    let finite_limit = match provider.precision {
        super::super::NumericPrecision::F32 => "3.402823466e38",
        super::super::NumericPrecision::F64 => "1.7976931348623157e308",
    };
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    format!(
        r#"
struct IndexBuffer {{ data: array<u32>, }};
struct ScalarBuffer {{ data: array<{scalar}>, }};
struct StatusBuffer {{ data: array<atomic<u32>>, }};
@group(0) @binding(0) var<storage, read> RowOffsets: IndexBuffer;
@group(0) @binding(1) var<storage, read> ColumnIndices: IndexBuffer;
@group(0) @binding(2) var<storage, read> Values: ScalarBuffer;
@group(0) @binding(3) var<storage, read_write> InverseDiagonal: ScalarBuffer;
@group(0) @binding(4) var<storage, read_write> Status: StatusBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    let rows = arrayLength(&RowOffsets.data) - 1u;
    if (row >= rows) {{ return; }}
    var diagonal: {scalar} = 0.0;
    var found = false;
    for (var index = RowOffsets.data[row]; index < RowOffsets.data[row + 1u]; index = index + 1u) {{
        if (ColumnIndices.data[index] == row) {{
            diagonal = Values.data[index];
            found = true;
        }}
    }}
    let inverse = 1.0 / diagonal;
    if (!found || diagonal == 0.0 || inverse != inverse || abs(inverse) > {finite_limit}) {{
        atomicOr(&Status.data[0], 1u);
        InverseDiagonal.data[row] = 0.0;
        return;
    }}
    InverseDiagonal.data[row] = inverse;
}}
"#
    )
}

fn apply_shader(provider: &WgpuProvider) -> String {
    let scalar = provider.precision.as_str();
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    format!(
        r#"
struct ScalarBuffer {{ data: array<{scalar}>, }};
@group(0) @binding(0) var<storage, read> InverseDiagonal: ScalarBuffer;
@group(0) @binding(1) var<storage, read> Input: ScalarBuffer;
@group(0) @binding(2) var<storage, read_write> Output: ScalarBuffer;
@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < arrayLength(&Input.data)) {{
        Output.data[index] = InverseDiagonal.data[index] * Input.data[index];
    }}
}}
"#
    )
}

fn dispatch_groups(provider: &WgpuProvider, dimension: usize) -> anyhow::Result<u32> {
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    let groups = u32::try_from(dimension)
        .ok()
        .and_then(|count| count.checked_add(workgroup_size - 1))
        .map(|count| count / workgroup_size)
        .ok_or_else(|| anyhow!("Jacobi dispatch count exceeds WGPU limits"))?;
    ensure!(
        groups <= provider.adapter_limits.max_compute_workgroups_per_dimension,
        "Jacobi dispatch exceeds the adapter workgroup limit"
    );
    Ok(groups)
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
