use runmat_accelerate_api::WgpuDeviceAccess;

use crate::backend::wgpu::pipelines::PipelineBundle;

use super::super::pipelines::bundle;
use super::shaders;

pub(crate) struct SparseCgPipelines {
    pub(super) spmv: PipelineBundle,
    pub(super) initialize_vectors: PipelineBundle,
    pub(super) dot_partials: PipelineBundle,
    pub(super) dot_finalize: PipelineBundle,
    pub(super) initialize_state: PipelineBundle,
    pub(super) alpha: PipelineBundle,
    pub(super) update_solution_residual: PipelineBundle,
    pub(super) apply_preconditioner: PipelineBundle,
    pub(super) update_state: PipelineBundle,
    pub(super) update_direction: PipelineBundle,
}

impl SparseCgPipelines {
    pub(in super::super) fn new(
        access: &WgpuDeviceAccess,
        scalar: &str,
        finite_limit: &str,
        workgroup_size: u32,
    ) -> Self {
        let sources = shaders::sources(scalar, finite_limit, workgroup_size);
        Self {
            spmv: bundle(
                access,
                "runmat-sparse-cg-spmv",
                &[false, false, false, false, true, false],
                sources.spmv,
            ),
            initialize_vectors: bundle(
                access,
                "runmat-sparse-cg-initialize-vectors",
                &[false, false, false, true, true, true, false],
                sources.initialize_vectors,
            ),
            dot_partials: bundle(
                access,
                "runmat-sparse-cg-dot-partials",
                &[false, false, true, false],
                sources.dot_partials,
            ),
            dot_finalize: bundle(
                access,
                "runmat-sparse-cg-dot-finalize",
                &[false, true, false],
                sources.dot_finalize,
            ),
            initialize_state: bundle(
                access,
                "runmat-sparse-cg-initialize-state",
                &[false, false, true],
                sources.initialize_state,
            ),
            alpha: bundle(
                access,
                "runmat-sparse-cg-alpha",
                &[false, true],
                sources.alpha,
            ),
            update_solution_residual: bundle(
                access,
                "runmat-sparse-cg-update-solution-residual",
                &[true, true, false, false, false],
                sources.update_solution_residual,
            ),
            apply_preconditioner: bundle(
                access,
                "runmat-sparse-cg-apply-preconditioner",
                &[false, false, true, false],
                sources.apply_preconditioner,
            ),
            update_state: bundle(
                access,
                "runmat-sparse-cg-update-state",
                &[false, false, true],
                sources.update_state,
            ),
            update_direction: bundle(
                access,
                "runmat-sparse-cg-update-direction",
                &[false, true, false],
                sources.update_direction,
            ),
        }
    }
}
