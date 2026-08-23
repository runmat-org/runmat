use runmat_meshing_core::MeshingPartitionKind;

use crate::{
    ExactCurveJoinKernel, ExactCurveRefinementKernel, ExactCurveStageKernel,
    ExactMeshingEvaluatorProvider, ExactSolverProjectionKernel, ExactSurfaceJoinKernel,
    ExactSurfacePartitionKernel, ExactVolumeKernel, MeshingStageInvocation, MeshingStageKernel,
    PortableMeshingEvaluatorProvider, SolverPublicationKernel, SolverSerializationKernel,
    SolverValidationKernel, ValidatedMeshingStageOutput,
};

/// Dispatches admitted meshing stage semantics without acquiring any scheduling responsibility.
/// Execution hosts use this one production entry point while meshing remains the owner of stage
/// and partition meaning.
#[derive(Clone, Debug)]
pub struct MeshingKernelDispatcher<P = PortableMeshingEvaluatorProvider> {
    evaluator_provider: P,
}

impl Default for MeshingKernelDispatcher<PortableMeshingEvaluatorProvider> {
    fn default() -> Self {
        Self::new(PortableMeshingEvaluatorProvider)
    }
}

impl<P> MeshingKernelDispatcher<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
    }
}

impl<P: ExactMeshingEvaluatorProvider> MeshingStageKernel for MeshingKernelDispatcher<P> {
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
            ) => ExactCurveJoinKernel::new(&self.evaluator_provider).execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::CurveMesh,
                MeshingPartitionKind::WholeStage,
            ) => ExactCurveRefinementKernel::new(&self.evaluator_provider).execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::CurveMesh,
                MeshingPartitionKind::CanonicalEntityBatch,
            ) => ExactCurveStageKernel::new(&self.evaluator_provider).execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::SurfaceMesh,
                MeshingPartitionKind::DeterministicJoin,
            ) => ExactSurfaceJoinKernel.execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::SurfaceMesh,
                MeshingPartitionKind::CanonicalEntityBatch,
            ) => ExactSurfacePartitionKernel::new(&self.evaluator_provider).execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::Tetrahedralization,
                MeshingPartitionKind::WholeStage,
            ) => ExactVolumeKernel.execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::OrderElevation,
                MeshingPartitionKind::WholeStage,
            ) => ExactSolverProjectionKernel::new(&self.evaluator_provider).execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::Validation,
                MeshingPartitionKind::WholeStage,
            ) => SolverValidationKernel.execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::Serialization,
                MeshingPartitionKind::WholeStage,
            ) => SolverSerializationKernel.execute(invocation),
            (
                runmat_meshing_core::MeshingStageKind::Publication,
                MeshingPartitionKind::WholeStage,
            ) => SolverPublicationKernel.execute(invocation),
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
