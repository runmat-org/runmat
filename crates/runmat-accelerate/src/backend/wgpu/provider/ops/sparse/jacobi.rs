use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProviderFuture, GpuTensorHandle, GpuTensorStorage, SparseFeasibilityQuery,
    SparseOperation, SparsePreconditionerApplyRequest, SparsePreconditionerDescriptor,
    SparsePreconditionerHandle, SparsePreconditionerKind, SparsePreconditionerRequest,
};

use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};
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
        let groups = super::dispatch_groups(provider, dimension)?;
        let (inverse_diagonal, staging) = provider.with_emission(|emission, pending| {
            let inverse_diagonal = provider.create_storage_buffer_checked(
                emission,
                dimension,
                "runmat-sparse-jacobi-inverse",
            )?;
            if dimension == 0 {
                return anyhow::Ok((inverse_diagonal, None));
            }
            let status = super::readback_source(
                provider,
                emission,
                "runmat-sparse-jacobi-status",
                bytemuck::cast_slice(&[0_u32]),
            )?;
            let mut encoder = pending.profiled_encoder(
                emission,
                &wgpu::CommandEncoderDescriptor {
                    label: Some("runmat-sparse-jacobi-prepare-encoder"),
                },
            );
            super::dispatch(
                emission,
                &mut encoder,
                &provider.sparse_pipelines.jacobi_prepare,
                "runmat-sparse-jacobi-prepare-pass",
                &[
                    matrix.pattern.row_offsets.as_ref(),
                    matrix.pattern.column_indices.as_ref(),
                    matrix.values.as_ref(),
                    inverse_diagonal.as_ref(),
                    status.as_ref(),
                ],
                groups,
            )?;
            let staging = super::stage_readback(
                emission,
                &mut encoder,
                status.as_ref(),
                4,
                "runmat-sparse-jacobi-status-readback",
            );
            provider.submit(emission, pending, encoder)?;
            Ok((inverse_diagonal, Some(staging)))
        })??;
        if let Some(staging) = staging {
            let state = provider
                .map_readback_with(staging, "sparse Jacobi preparation", |bytes| {
                    Ok(u32::from_le_bytes(
                        bytes
                            .get(..4)
                            .and_then(|status| status.try_into().ok())
                            .ok_or_else(|| anyhow!("sparse Jacobi status is truncated"))?,
                    ))
                })
                .await?;
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
        let preconditioner_id = provider.sparse.next_id();
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
        let groups = super::dispatch_groups(provider, dimension)?;
        let output = provider.with_emission(|emission, pending| {
            let output = provider.create_storage_buffer_checked(
                emission,
                dimension,
                "runmat-sparse-jacobi-output",
            )?;
            if dimension > 0 {
                let mut encoder = pending.profiled_encoder(
                    emission,
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("runmat-sparse-jacobi-apply-encoder"),
                    },
                );
                super::dispatch(
                    emission,
                    &mut encoder,
                    &provider.sparse_pipelines.jacobi_apply,
                    "runmat-sparse-jacobi-apply-pass",
                    &[
                        preconditioner.inverse_diagonal.as_ref(),
                        input.buffer.as_ref(),
                        output.as_ref(),
                    ],
                    groups,
                )?;
                provider.submit(emission, pending, encoder)?;
            }
            anyhow::Ok(output)
        })??;
        if dimension > 0 {
            provider.record_kernel_launch_basic(
                "sparse_jacobi_apply",
                &[("rows", preconditioner.descriptor.dimension)],
                &[(
                    "workgroup_size",
                    u64::from(provider.workgroup_config.scalar.max(1)),
                )],
            );
        }
        let allocated_bytes = output.size();
        provider.register_numeric_buffer(
            output,
            NumericBufferRegistration {
                shape: request.input.shape().to_vec(),
                len: dimension,
                physical_element_type: preconditioner.descriptor.element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )
    })
}
