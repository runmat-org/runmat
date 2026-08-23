use runmat_execution::value::ValueRef;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingChunkMediaType, MeshingStageEvidence};

use crate::{import_result_publication, ExactMeshingDagPlanner};

use super::stage::{execute_planned_stage, ExactMeshingDagExecutor};
use super::{execute_partitions, ExactMeshingDagRun, ExactMeshingDagRunError};

pub(super) fn execute_surface_dag<E: ExactMeshingDagExecutor>(
    executor: &mut E,
    planner: &ExactMeshingDagPlanner,
    topology: &ExactBRepTopology,
    initial_curve_root: ValueRef,
    run: &ExactMeshingDagRun<'_>,
    observations: &mut Vec<MeshingStageEvidence>,
) -> Result<ValueRef, ExactMeshingDagRunError> {
    let mut pass = planner.begin_surface_pass(
        topology,
        initial_curve_root,
        run.preferred_faces_per_partition,
    )?;
    loop {
        let partition_roots = execute_partitions(
            executor,
            pass.partitions(),
            run.program_revision.clone(),
            run.inventory_limits,
            observations,
        )?;
        let join = planner.surface_join(&pass, partition_roots.clone())?;
        let joined = execute_planned_stage(
            executor,
            &join,
            run.program_revision.clone(),
            run.inventory_limits,
        )?;
        observations.push(joined.evidence);
        if surface_converged(executor, &join, &joined.root, run.inventory_limits)? {
            return Ok(joined.root);
        }
        let refinement = planner.curve_refinement(&pass, partition_roots, joined.root)?;
        let refined = execute_planned_stage(
            executor,
            &refinement,
            run.program_revision.clone(),
            run.inventory_limits,
        )?;
        observations.push(refined.evidence);
        pass = planner.next_surface_pass(
            &pass,
            topology,
            refined.root,
            run.preferred_faces_per_partition,
        )?;
    }
}

fn surface_converged(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    stage: &crate::PlannedMeshingStage,
    root: &ValueRef,
    limits: ObjectInventoryLimits,
) -> Result<bool, ExactMeshingDagRunError> {
    let publication =
        import_result_publication(source, root, stage.host().artifact_access.clone(), limits)?;
    let streams = publication.stage_objects().decoded_streams()?;
    match streams.as_slice() {
        [pass] if pass.media_type == MeshingChunkMediaType::SurfacePartitions => Ok(false),
        [pass, surface]
            if pass.media_type == MeshingChunkMediaType::SurfacePartitions
                && surface.media_type == MeshingChunkMediaType::SurfaceMesh =>
        {
            Ok(true)
        }
        _ => Err(ExactMeshingDagRunError::Invalid(
            "surface join published an invalid convergence stream shape".into(),
        )),
    }
}
