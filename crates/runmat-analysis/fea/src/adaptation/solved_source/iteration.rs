//! Feed actual original solved fields into the existing shared accuracy adapter.
use super::*;
use crate::adaptation::{
    build_structural_adaptation_iteration, estimate_structural_recovery_error,
    measure_solver_field_transfer_error, transfer_solver_field, StructuralAdaptationIteration,
    StructuralAdaptationIterationInput, StructuralRecoveryEstimatorOptions,
};
use runmat_analysis_core::AccuracyPlan;
use runmat_meshing_core::{
    FieldTopologyLocation, SolverMeshAdaptationLineage, SolverMeshTransferMap,
};

#[derive(Clone, Copy)]
pub struct StructuralAdaptationSolvedIterationInput<'a> {
    pub source: StructuralAdaptationSolvedSourceInput<'a>,
    pub target: StructuralAdaptationSolvedSourceInput<'a>,
    pub transfer_map: &'a SolverMeshTransferMap,
    pub lineage: &'a SolverMeshAdaptationLineage,
    pub estimator_options: StructuralRecoveryEstimatorOptions,
    pub previous: Option<&'a StructuralAdaptationSolvedIteration>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralAdaptationSolvedIteration {
    pub source: StructuralAdaptationSolvedSource,
    pub target: StructuralAdaptationSolvedSource,
    pub iteration: StructuralAdaptationIteration,
}

pub fn build_structural_adaptation_iteration_from_solved_results(
    input: StructuralAdaptationSolvedIterationInput<'_>,
    accuracy: &AccuracyPlan,
) -> Result<StructuralAdaptationSolvedIteration, Error> {
    if input.source.model != input.target.model
        || input.source.observation_id != input.target.observation_id
    {
        return Err(invalid(
            "adaptation must retain the same authored physical problem and target observation",
        ));
    }
    input
        .lineage
        .validate_against(input.source.mesh, input.target.mesh, input.transfer_map)
        .map_err(|e| invalid(format!("{e:?}")))?;
    let source = StructuralAdaptationSolvedSource::derive(input.source)?;
    let target = StructuralAdaptationSolvedSource::derive(input.target)?;
    if input
        .previous
        .is_some_and(|previous| previous.target != source)
    {
        return Err(invalid(
            "adaptation source is not the previous original solved target",
        ));
    }
    let estimate = |source: StructuralAdaptationSolvedSourceInput<'_>| {
        let topology = source
            .mesh
            .topology
            .field_topologies
            .iter()
            .find(|t| t.location == FieldTopologyLocation::VolumeElement)
            .ok_or_else(|| invalid("original solid stress topology is missing"))?;
        let stress = source
            .run
            .field(crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .ok_or_else(|| invalid("original integration-point stress is missing"))?;
        estimate_structural_recovery_error(
            source.mesh,
            &topology.topology_id,
            stress,
            input.estimator_options,
        )
        .map_err(|e| invalid(format!("{e:?}")))
    };
    let source_estimator = estimate(input.source)?;
    let estimator = estimate(input.target)?;
    let node_topology = input
        .source
        .mesh
        .topology
        .field_topologies
        .iter()
        .find(|t| t.location == FieldTopologyLocation::Node)
        .ok_or_else(|| invalid("original nodal topology is missing"))?;
    let displacement = input
        .source
        .run
        .field(crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .ok_or_else(|| invalid("original displacement is missing"))?;
    let reference = input
        .target
        .run
        .field(crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .ok_or_else(|| invalid("original target displacement is missing"))?;
    let transferred = transfer_solver_field(
        input.source.mesh,
        input.target.mesh,
        Some(input.transfer_map),
        &node_topology.topology_id,
        displacement,
    )
    .map_err(|e| invalid(format!("{e:?}")))?;
    let transfer_errors = [measure_solver_field_transfer_error(&transferred, reference)
        .map_err(|e| invalid(format!("{e:?}")))?];
    let iteration = build_structural_adaptation_iteration(
        StructuralAdaptationIterationInput {
            model: input.source.model,
            source_artifact: input.source.mesh,
            target_artifact: input.target.mesh,
            transfer_map: input.transfer_map,
            lineage: input.lineage,
            source_estimator: &source_estimator,
            estimator: &estimator,
            transfer_errors: &transfer_errors,
            source_solver_result: &source.solver,
            solver_result: &target.solver,
            source_target_quantity: &source.target_quantity,
            target_quantity: &target.target_quantity,
            previous: input.previous.map(|p| &p.iteration),
        },
        accuracy,
    )
    .map_err(|e| invalid(format!("{e:?}")))?;
    Ok(StructuralAdaptationSolvedIteration {
        source,
        target,
        iteration,
    })
}
impl StructuralAdaptationSolvedIteration {
    pub fn validate_against(
        &self,
        input: StructuralAdaptationSolvedIterationInput<'_>,
        accuracy: &AccuracyPlan,
    ) -> Result<(), Error> {
        if *self != build_structural_adaptation_iteration_from_solved_results(input, accuracy)? {
            return Err(invalid(
                "retained iteration differs from original solved sources",
            ));
        }
        Ok(())
    }
}
