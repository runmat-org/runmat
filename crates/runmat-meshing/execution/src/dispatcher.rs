use runmat_meshing_core::MeshingPartitionKind;

use crate::{
    ExactCurveJoinKernel, ExactCurveStageKernel, MeshingStageInvocation, MeshingStageKernel,
    ValidatedMeshingStageOutput,
};

/// Dispatches admitted meshing stage semantics without acquiring any scheduling responsibility.
/// Execution hosts use this one production entry point while meshing remains the owner of stage
/// and partition meaning.
#[derive(Default)]
pub struct MeshingKernelDispatcher;

impl MeshingStageKernel for MeshingKernelDispatcher {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<runmat_meshing_core::MeshingFailure>> {
        match invocation.host.workload.partition.kind {
            MeshingPartitionKind::DeterministicJoin => {
                ExactCurveJoinKernel::default().execute(invocation)
            }
            _ => ExactCurveStageKernel::default().execute(invocation),
        }
    }
}
