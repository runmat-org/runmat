use runmat_accelerate_api::WgpuEmission;

use super::super::super::{BufferEntry, WgpuProvider};
use super::super::dispatch;
use super::super::storage::{DeviceSparseMatrix, DeviceSparsePreconditioner};
use super::pipelines::SparseCgPipelines;
use super::workspace::CgWorkspace;
use crate::backend::wgpu::dispatch::observation::ProfiledEncoder;

pub(super) struct CgOperator<'a> {
    pub(super) matrix: &'a DeviceSparseMatrix,
    pub(super) preconditioner: &'a DeviceSparsePreconditioner,
}

pub(super) struct CgInputs<'a> {
    pub(super) operator: CgOperator<'a>,
    pub(super) rhs: &'a BufferEntry,
    pub(super) initial_guess: Option<&'a BufferEntry>,
}

pub(super) fn encode_initialization(
    provider: &WgpuProvider,
    emission: &WgpuEmission<'_>,
    encoder: &mut ProfiledEncoder<'_>,
    pipelines: &SparseCgPipelines,
    inputs: CgInputs<'_>,
    workspace: &CgWorkspace,
    vector_groups: u32,
) -> anyhow::Result<()> {
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
        emission,
        encoder,
        &pipelines.spmv,
        "runmat-sparse-cg-initial-spmv",
        &[
            inputs.operator.matrix.pattern.row_offsets.as_ref(),
            inputs.operator.matrix.pattern.column_indices.as_ref(),
            inputs.operator.matrix.values.as_ref(),
            workspace.solution.as_ref(),
            workspace.product.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.initialize_vectors,
        "runmat-sparse-cg-initialize-vectors",
        &[
            inputs.rhs.buffer.as_ref(),
            workspace.product.as_ref(),
            inputs.operator.preconditioner.inverse_diagonal.as_ref(),
            workspace.residual.as_ref(),
            workspace.preconditioned.as_ref(),
            workspace.direction.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    )?;
    encode_dot(
        provider,
        emission,
        encoder,
        pipelines,
        DotProduct {
            left: workspace.residual.as_ref(),
            right: workspace.residual.as_ref(),
            output: workspace.residual_squared.as_ref(),
        },
        workspace,
    )?;
    encode_dot(
        provider,
        emission,
        encoder,
        pipelines,
        DotProduct {
            left: workspace.residual.as_ref(),
            right: workspace.preconditioned.as_ref(),
            output: workspace.rho.as_ref(),
        },
        workspace,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.initialize_state,
        "runmat-sparse-cg-initialize-state",
        &[
            workspace.residual_squared.as_ref(),
            workspace.rho.as_ref(),
            workspace.state.as_ref(),
        ],
        1,
    )?;
    Ok(())
}

pub(super) fn encode_iteration(
    provider: &WgpuProvider,
    emission: &WgpuEmission<'_>,
    encoder: &mut ProfiledEncoder<'_>,
    pipelines: &SparseCgPipelines,
    operator: CgOperator<'_>,
    workspace: &CgWorkspace,
    vector_groups: u32,
) -> anyhow::Result<()> {
    dispatch(
        emission,
        encoder,
        &pipelines.spmv,
        "runmat-sparse-cg-spmv",
        &[
            operator.matrix.pattern.row_offsets.as_ref(),
            operator.matrix.pattern.column_indices.as_ref(),
            operator.matrix.values.as_ref(),
            workspace.direction.as_ref(),
            workspace.product.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    )?;
    encode_dot(
        provider,
        emission,
        encoder,
        pipelines,
        DotProduct {
            left: workspace.direction.as_ref(),
            right: workspace.product.as_ref(),
            output: workspace.denominator.as_ref(),
        },
        workspace,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.alpha,
        "runmat-sparse-cg-alpha",
        &[workspace.denominator.as_ref(), workspace.state.as_ref()],
        1,
    )?;
    dispatch(
        emission,
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
    )?;
    encode_dot(
        provider,
        emission,
        encoder,
        pipelines,
        DotProduct {
            left: workspace.residual.as_ref(),
            right: workspace.residual.as_ref(),
            output: workspace.residual_squared.as_ref(),
        },
        workspace,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.apply_preconditioner,
        "runmat-sparse-cg-apply-preconditioner",
        &[
            operator.preconditioner.inverse_diagonal.as_ref(),
            workspace.residual.as_ref(),
            workspace.preconditioned.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    )?;
    encode_dot(
        provider,
        emission,
        encoder,
        pipelines,
        DotProduct {
            left: workspace.residual.as_ref(),
            right: workspace.preconditioned.as_ref(),
            output: workspace.rho.as_ref(),
        },
        workspace,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.update_state,
        "runmat-sparse-cg-update-state",
        &[
            workspace.residual_squared.as_ref(),
            workspace.rho.as_ref(),
            workspace.state.as_ref(),
        ],
        1,
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.update_direction,
        "runmat-sparse-cg-update-direction",
        &[
            workspace.preconditioned.as_ref(),
            workspace.direction.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups,
    )?;
    Ok(())
}

struct DotProduct<'a> {
    left: &'a wgpu::Buffer,
    right: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
}

fn encode_dot(
    provider: &WgpuProvider,
    emission: &WgpuEmission<'_>,
    encoder: &mut ProfiledEncoder<'_>,
    pipelines: &SparseCgPipelines,
    product: DotProduct<'_>,
    workspace: &CgWorkspace,
) -> anyhow::Result<()> {
    dispatch(
        emission,
        encoder,
        &pipelines.dot_partials,
        "runmat-sparse-cg-dot-partials",
        &[
            product.left,
            product.right,
            workspace.dot_partials.as_ref(),
            workspace.state.as_ref(),
        ],
        vector_groups(provider, product.left),
    )?;
    dispatch(
        emission,
        encoder,
        &pipelines.dot_finalize,
        "runmat-sparse-cg-dot-finalize",
        &[
            workspace.dot_partials.as_ref(),
            product.output,
            workspace.state.as_ref(),
        ],
        1,
    )?;
    Ok(())
}

fn vector_groups(provider: &WgpuProvider, buffer: &wgpu::Buffer) -> u32 {
    let scalar_bytes = provider.element_size as u64;
    let count = buffer.size() / scalar_bytes;
    count
        .div_ceil(u64::from(provider.workgroup_config.scalar.max(1)))
        .try_into()
        .expect("validated sparse vector dispatch fits u32")
}
