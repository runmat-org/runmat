//! Shared receipt-based accuracy evaluation admission, independent of physics family.
use super::{
    build_resource_plan, AccuracyEvaluationResourceDemand, ByteCount, Count, FeaResourceDemand,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, PreparedPhaseResourceFacts,
    ResourcePlanningError, RESOURCE_ESTIMATION_MODEL_REVISION,
};
use crate::{
    execution::AccuracyEvaluationRequest,
    results::accuracy::{accuracy_evaluation_resource_profile, AccuracyEvidence},
};

#[cfg(test)]
mod tests;

/// Borrowed original count/profile admission before finite-volume projections or
/// numerical stencils expand. Uses the same source/property estimators as prepare;
/// validity and the complete authority reservation are still independently rederived.
pub fn preflight_fluid_accuracy_source(
    workload: &crate::execution::FluidSteadyWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    maximum: u64,
) -> Result<u64, ResourcePlanningError> {
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    let geometry =
        crate::topology::FiniteVolumeTopology::reserve_mesh(mesh, workload.geometry_policy)
            .map_err(|e| invalid(&e))?;
    let numerical = crate::physics::cfd::canonical::CanonicalFluidReservation::for_counts(
        &geometry,
        geometry.face_incidence_count,
        workload.policy,
    )
    .map_err(|e| invalid(&e))?;
    let projection = crate::topology::AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|e| invalid(&e))?;
    let metadata = runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
        workload,
        runmat_canonical_codec::CanonicalLimits::new(64 * 1024 * 1024, 2_000_000, 65_536, 64),
    )
    .map_err(|e| invalid(&e))?;
    let add = |a: u64, b: u64| {
        a.checked_add(b).ok_or(ResourcePlanningError::Overflow(
            "fluid accuracy source preflight",
        ))
    };
    let preliminary = add(
        add(projection, geometry.scratch_bytes as u64)?,
        add(
            add(metadata, metadata)?,
            numerical.resident_bytes_upper_bound,
        )?,
    )?;
    if preliminary > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "fluid accuracy borrowed source expansion",
            required: preliminary,
            maximum,
        });
    }
    let properties = crate::properties::PreparedProperties::prepare(
        &workload.model,
        Some(mesh),
        maximum - preliminary,
    )
    .map_err(|e| invalid(&e))?;
    let full = add(
        preliminary,
        add(
            properties.estimated_resident_bytes,
            properties.estimated_resident_bytes,
        )?,
    )?;
    if full > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "fluid accuracy borrowed source expansion",
            required: full,
            maximum,
        });
    }
    Ok(full)
}

pub fn build_accuracy_resource_plan(
    request: &AccuracyEvaluationRequest,
    evidence: &[AccuracyEvidence],
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_accuracy_resource_plan_with_source_workspace(request, evidence, facts, None)
}

/// Fluid passes an independently derived original numerical/source workspace;
/// legacy families retain their existing metadata decode envelope unchanged.
pub fn build_accuracy_resource_plan_with_source_workspace(
    request: &AccuracyEvaluationRequest,
    evidence: &[AccuracyEvidence],
    facts: PreparedPhaseResourceFacts,
    source_workspace: Option<u64>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let profile = accuracy_evaluation_resource_profile(request, evidence)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    super::post_process::validate_projected_output(
        &facts.output,
        profile.output_encoded_bytes_upper_bound,
        "accuracy evaluation",
    )?;
    // Source verification may reconstruct mesh projections and authored observation plans.
    // Reserve that transient workspace separately from resident decoded transport inputs.
    let context_workspace = match source_workspace {
        Some(bytes) => bytes
            .checked_add(profile.workspace_bytes_upper_bound)
            .ok_or(ResourcePlanningError::Overflow(
                "accuracy original source workspace",
            ))?,
        None => facts
            .inputs
            .decoded_resident_bytes_admission_budget
            .0
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(profile.workspace_bytes_upper_bound))
            .ok_or(ResourcePlanningError::Overflow(
                "accuracy source verification workspace",
            ))?,
    };
    let mut worker = super::post_process::post_process_worker();
    worker.maximum_wall_milliseconds = worker
        .maximum_wall_milliseconds
        .min(request.plan.resources.maximum_wall_time_milliseconds);
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AccuracyEvaluation(request.identity()),
        demand: FeaResourceDemand::AccuracyEvaluation(AccuracyEvaluationResourceDemand {
            request: request.identity(),
            source_result: request.source_result,
            criterion_count: Count(
                u64::try_from(request.plan.criteria.len())
                    .map_err(|_| ResourcePlanningError::Overflow("accuracy criteria"))?,
            ),
            evidence_count: Count(
                u64::try_from(evidence.len())
                    .map_err(|_| ResourcePlanningError::Overflow("accuracy evidence"))?,
            ),
            inputs: facts.inputs,
            evaluator_workspace_bytes_upper_bound: ByteCount(context_workspace),
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker,
        accuracy_ceiling: request.plan.resources,
    })
}
