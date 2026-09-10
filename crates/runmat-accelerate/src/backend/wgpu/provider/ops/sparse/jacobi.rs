use std::sync::Arc;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, GpuTensorHandle, GpuTensorStorage, SparseFeasibilityQuery,
    SparseOperation, SparsePreconditionerApplyRequest, SparsePreconditionerDescriptor,
    SparsePreconditionerHandle, SparsePreconditionerKind, SparsePreconditionerRequest,
};
use wgpu::util::DeviceExt;

use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};
use super::jacobi_kernels::{apply_pipeline, prepare_pipeline};
use super::storage::DeviceSparsePreconditioner;

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
