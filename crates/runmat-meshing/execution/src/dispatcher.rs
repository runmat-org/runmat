use runmat_meshing_core::MeshingPartitionKind;

use crate::{
    ExactCurveJoinKernel, ExactCurveRefinementKernel, ExactCurveStageKernel,
    ExactSurfaceJoinKernel, ExactSurfacePartitionKernel, MeshingStageInvocation,
    MeshingStageKernel, ValidatedMeshingStageOutput,
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
        match (
            invocation.host.workload.stage,
            invocation.host.workload.partition.kind,
        ) {
            (
                runmat_meshing_core::MeshingStageKind::CurveMesh,
                MeshingPartitionKind::DeterministicJoin,
            ) => ExactCurveJoinKernel::default().execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::CurveMesh,
                MeshingPartitionKind::WholeStage,
            ) => ExactCurveRefinementKernel::default().execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::CurveMesh,
                MeshingPartitionKind::CanonicalEntityBatch,
            ) => ExactCurveStageKernel::default().execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::SurfaceMesh,
                MeshingPartitionKind::DeterministicJoin,
            ) => ExactSurfaceJoinKernel.execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::SurfaceMesh,
                MeshingPartitionKind::CanonicalEntityBatch,
            ) => ExactSurfacePartitionKernel::default().execute(invocation),
            _ => Err(Box::new(runmat_meshing_core::MeshingFailure {
                schema_version: runmat_meshing_core::MESHING_FAILURE_SCHEMA_VERSION,
                category: runmat_meshing_core::MeshingFailureCategory::InternalInvariantViolation,
                stage: invocation.host.workload.stage,
                operation: invocation.host.workload.stage.operation(),
                entity_ids: Vec::new(),
                witnesses: Vec::new(),
                request_values: Vec::new(),
                achieved_values: Vec::new(),
                remediation: "register a production kernel for this meshing stage shape".into(),
            })),
        }
    }
}
