use runmat_execution::value::ValueRef;
use runmat_meshing_core::{MeshingInputKind, MeshingPartitionKind, MeshingStageKind};

use super::stage::whole_partition;
use super::{invalid, ExactMeshingDagPlanner, PlannedMeshingStage};
use crate::MeshingExecutionResult;

impl ExactMeshingDagPlanner {
    /// Plans the canonical Tet4/Tet10 solver projection from independently admitted terminal
    /// geometry, surface, volume, and domain-model roots.
    pub fn solver_projection(
        &self,
        surface_root: ValueRef,
        volume_root: ValueRef,
        domain_model_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if surface_root.logical_digest == volume_root.logical_digest
            || [surface_root.logical_digest, volume_root.logical_digest]
                .contains(&domain_model_root.logical_digest)
        {
            return Err(invalid(
                "solver projection requires distinct surface, volume, and domain-model roots",
            ));
        }
        self.build_stage_with_dependencies(
            MeshingStageKind::OrderElevation,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![
                (MeshingInputKind::ExactGeometry, self.geometry_root.clone()),
                (MeshingInputKind::StageArtifact, surface_root),
                (MeshingInputKind::StageArtifact, volume_root),
                (MeshingInputKind::DomainModel, domain_model_root),
            ],
        )
    }

    pub fn solver_validation(
        &self,
        projection_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        self.build_stage_with_dependencies(
            MeshingStageKind::Validation,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![(MeshingInputKind::StageArtifact, projection_root)],
        )
    }

    pub fn solver_serialization(
        &self,
        projection_root: ValueRef,
        validation_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if projection_root.logical_digest == validation_root.logical_digest {
            return Err(invalid(
                "solver serialization requires distinct projection and validation roots",
            ));
        }
        self.build_stage_with_dependencies(
            MeshingStageKind::Serialization,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![
                (MeshingInputKind::StageArtifact, projection_root),
                (MeshingInputKind::StageArtifact, validation_root),
            ],
        )
    }

    pub fn solver_publication(
        &self,
        serialization_root: ValueRef,
        evidence_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if serialization_root.logical_digest == evidence_root.logical_digest {
            return Err(invalid(
                "solver publication requires distinct serialization and evidence roots",
            ));
        }
        self.build_stage_with_dependencies(
            MeshingStageKind::Publication,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![
                (MeshingInputKind::StageArtifact, serialization_root),
                (MeshingInputKind::Evidence, evidence_root),
            ],
        )
    }
}
