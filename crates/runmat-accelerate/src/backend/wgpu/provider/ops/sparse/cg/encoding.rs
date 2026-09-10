use std::sync::Arc;

use super::super::super::{BufferEntry, WgpuProvider};
use super::super::storage::{DeviceSparseMatrix, DeviceSparsePreconditioner};
use super::pipelines::{ComputeStage, SparseCgPipelines};
use super::workspace::CgWorkspace;

pub(super) struct CgInputs<'a> {
    pub(super) matrix: &'a DeviceSparseMatrix,
    pub(super) preconditioner: &'a DeviceSparsePreconditioner,
    pub(super) rhs: &'a BufferEntry,
    pub(super) initial_guess: Option<&'a BufferEntry>,
}

pub(super) fn encode_initialization(
    provider: &WgpuProvider,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SparseCgPipelines,
    inputs: CgInputs<'_>,
    workspace: &CgWorkspace,
    vector_groups: u32,
) {
    if let Some(initial_guess) = inputs.initial_guess {
        encoder.copy_buffer_to_buffer(
            initial_guess.buffer.as_ref(),
            0,
            workspace.solution.as_ref(),
            0,
            workspace.solution.size(),
        );
    } else {
        encoder.clear_buffer(workspace.solution.as_ref(), 0, None);
    }
    dispatch(
        provider,
        encoder,
        &pipelines.spmv,
        "runmat-sparse-cg-initial-spmv",
        &[
            inputs.matrix.pattern.row_offsets.as_ref(),
            inputs.matrix.pattern.column_indices.as_ref(),
            inputs.matrix.values.as_ref(),
            workspace.solution.as_ref(),
            workspace.product.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.initialize_vectors,
        "runmat-sparse-cg-initialize-vectors",
        &[
            inputs.rhs.buffer.as_ref(),
            workspace.product.as_ref(),
            inputs.preconditioner.inverse_diagonal.as_ref(),
            workspace.residual.as_ref(),
            workspace.preconditioned.as_ref(),
            workspace.direction.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
    encode_dot(
        provider,
        encoder,
        pipelines,
        workspace.residual.as_ref(),
        workspace.residual.as_ref(),
        workspace.residual_squared.as_ref(),
        workspace,
    );
    encode_dot(
        provider,
        encoder,
        pipelines,
        workspace.residual.as_ref(),
        workspace.preconditioned.as_ref(),
        workspace.rho.as_ref(),
        workspace,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.initialize_state,
        "runmat-sparse-cg-initialize-state",
        &[
            workspace.residual_squared.as_ref(),
            workspace.rho.as_ref(),
            workspace.state.as_ref(),
        ],
        1,
    );
}

pub(super) fn encode_iteration(
    provider: &WgpuProvider,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SparseCgPipelines,
    matrix: &DeviceSparseMatrix,
    preconditioner: &DeviceSparsePreconditioner,
    workspace: &CgWorkspace,
    vector_groups: u32,
) {
    dispatch(
        provider,
        encoder,
        &pipelines.spmv,
        "runmat-sparse-cg-spmv",
        &[
            matrix.pattern.row_offsets.as_ref(),
            matrix.pattern.column_indices.as_ref(),
            matrix.values.as_ref(),
            workspace.direction.as_ref(),
            workspace.product.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
    encode_dot(
        provider,
        encoder,
        pipelines,
        workspace.direction.as_ref(),
        workspace.product.as_ref(),
        workspace.denominator.as_ref(),
        workspace,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.alpha,
        "runmat-sparse-cg-alpha",
        &[workspace.denominator.as_ref(), workspace.state.as_ref()],
        1,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.update_solution_residual,
        "runmat-sparse-cg-update-solution-residual",
        &[
            workspace.solution.as_ref(),
            workspace.residual.as_ref(),
            workspace.direction.as_ref(),
            workspace.product.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
    encode_dot(
        provider,
        encoder,
        pipelines,
        workspace.residual.as_ref(),
        workspace.residual.as_ref(),
        workspace.residual_squared.as_ref(),
        workspace,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.apply_preconditioner,
        "runmat-sparse-cg-apply-preconditioner",
        &[
            preconditioner.inverse_diagonal.as_ref(),
            workspace.residual.as_ref(),
            workspace.preconditioned.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
    encode_dot(
        provider,
        encoder,
        pipelines,
        workspace.residual.as_ref(),
        workspace.preconditioned.as_ref(),
        workspace.rho.as_ref(),
        workspace,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.update_state,
        "runmat-sparse-cg-update-state",
        &[
            workspace.residual_squared.as_ref(),
            workspace.rho.as_ref(),
            workspace.state.as_ref(),
        ],
        1,
    );
    dispatch(
        provider,
        encoder,
        &pipelines.update_direction,
        "runmat-sparse-cg-update-direction",
        &[
            workspace.preconditioned.as_ref(),
            workspace.direction.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    );
}

fn encode_dot(
    provider: &WgpuProvider,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &SparseCgPipelines,
    left: &wgpu::Buffer,
    right: &wgpu::Buffer,
    output: &wgpu::Buffer,
    workspace: &CgWorkspace,
) {
    dispatch(
        provider,
        encoder,
        &pipelines.dot_partials,
        "runmat-sparse-cg-dot-partials",
        &[
            left,
            right,
            workspace.dot_partials.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups(provider, left),
    );
    dispatch(
        provider,
        encoder,
        &pipelines.dot_finalize,
        "runmat-sparse-cg-dot-finalize",
        &[
            workspace.dot_partials.as_ref(),
            output,
            workspace.state.as_ref(),
        ],
        1,
    );
}

fn vector_groups(provider: &WgpuProvider, buffer: &wgpu::Buffer) -> u32 {
    let scalar_bytes = provider.element_size as u64;
    let count = buffer.size() / scalar_bytes;
    count
        .div_ceil(u64::from(provider.workgroup_config.scalar.max(1)))
        .try_into()
        .expect("validated sparse vector dispatch fits u32")
}

fn dispatch(
    provider: &WgpuProvider,
    encoder: &mut wgpu::CommandEncoder,
    stage: &ComputeStage,
    label: &'static str,
    buffers: &[&wgpu::Buffer],
    groups: u32,
) {
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
    let bind_group = Arc::new(provider.device_ref().create_bind_group(
        &wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: stage.layout.as_ref(),
            entries: &entries,
        },
    ));
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(&stage.pipeline);
    pass.set_bind_group(0, bind_group.as_ref(), &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
