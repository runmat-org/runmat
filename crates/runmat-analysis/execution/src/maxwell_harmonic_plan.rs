//! Shared transport reservations for harmonic electric-phasor Maxwell work.
//! This prepares inputs, not a native executable task or semantic publication.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, AnalysisTransportProfiler, MaxwellHarmonicWorkload},
    planning::resource::{
        build_maxwell_harmonic_resource_plan, maxwell_harmonic_result_payload_bytes_upper_bound,
        FeaResourcePlan, PreparedMaxwellHarmonicResourceFacts,
    },
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct PreparedMaxwellHarmonicInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

pub fn prepare_maxwell_harmonic_inputs(
    workload: &MaxwellHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
) -> AnalysisBridgeResult<PreparedMaxwellHarmonicInputs> {
    let result_output = projected_result(workload, mesh, policy)?;
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
        workload.identity()?.digest(),
        &workload.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    let mesh_bytes = mesh.canonical_encode().map_err(invalid)?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh_bytes,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    let inputs = vec![workload_objects, mesh_objects];
    let resource_plan = build_maxwell_harmonic_resource_plan(
        workload,
        mesh,
        PreparedMaxwellHarmonicResourceFacts {
            local_execution,
            inputs: prepared_input_resource_facts(&inputs)?,
            result_output,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::MAXWELL_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            state_profiles_output: projected_state_profiles(workload, mesh, policy)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?;
    Ok(PreparedMaxwellHarmonicInputs {
        inputs,
        resource_plan,
    })
}

/// Rebuild the worker's resource authority using imported workload/mesh and actual input
/// manifests. The supplied plan is itself checked, not simply omitted from accounting.
pub fn rederive_maxwell_harmonic_resource_plan(
    workload: &MaxwellHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    // Original raw count/profile ceiling precedes source encoding/projection.
    maxwell_harmonic_result_payload_bytes_upper_bound(workload, mesh)?;
    use AnalysisNumericalObjectKind::{MaxwellHarmonicWorkload, Mesh, ResourcePlan};
    if inputs.len() != 3
        || [MaxwellHarmonicWorkload, Mesh, ResourcePlan]
            .iter()
            .any(|kind| {
                inputs
                    .iter()
                    .filter(|input| input.manifest.kind == *kind)
                    .count()
                    != 1
            })
    {
        return Err(invalid(
            "Maxwell inputs require exactly workload, mesh and resource plan",
        ));
    }
    for input in inputs {
        input.manifest.validate()?;
        let (expected, bytes) = match input.manifest.kind {
            MaxwellHarmonicWorkload => (workload.identity()?.digest(), workload.canonical_bytes()?),
            Mesh => (
                Digest::from_bytes(*mesh.canonical_digest.bytes()),
                mesh.canonical_encode().map_err(invalid)?,
            ),
            ResourcePlan => continue,
            _ => return Err(invalid("unexpected Maxwell input")),
        };
        if input.manifest.logical_identity != expected {
            return Err(invalid("Maxwell imported source identity differs"));
        }
        let canonical = prepare_analysis_objects(
            input.manifest.kind,
            expected,
            &bytes,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?;
        if input != &canonical {
            return Err(invalid(
                "Maxwell input closure differs from canonical source or chunk policy",
            ));
        }
    }
    let input = inputs
        .iter()
        .find(|input| input.manifest.kind == ResourcePlan)
        .ok_or_else(|| invalid("Maxwell resource plan missing"))?;
    let imported = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
    let plan = build_maxwell_harmonic_resource_plan(
        workload,
        mesh,
        PreparedMaxwellHarmonicResourceFacts {
            local_execution: LocalExecutionPlan::from_cpu_millicores(
                imported.request.cpu_millicores,
            )
            .map_err(invalid)?,
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            result_output: projected_result(workload, mesh, policy)?,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::MAXWELL_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            state_profiles_output: projected_state_profiles(workload, mesh, policy)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?;
    if imported != plan || input.manifest.logical_identity != plan.identity.digest() {
        return Err(invalid(
            "Maxwell resource plan disagrees with imported inputs or output policy",
        ));
    }
    let canonical = prepare_analysis_objects(
        ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    if input != &canonical {
        return Err(invalid(
            "Maxwell resource-plan closure differs from canonical plan",
        ));
    }
    Ok(plan)
}

fn projected_result(
    workload: &MaxwellHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    projected_output(
        AnalysisNumericalObjectKind::SolverResult,
        maxwell_harmonic_result_payload_bytes_upper_bound(workload, mesh)?,
        policy,
    )
}

fn projected_state_profiles(
    workload: &MaxwellHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    projected_output(
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        workload
            .state_publication_profiles_encoding_profile(Digest::from_bytes(
                *mesh.canonical_digest.bytes(),
            ))?
            .encoded_bytes_upper_bound,
        policy,
    )
}

fn projected_output(
    kind: AnalysisNumericalObjectKind,
    bytes: u64,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(bytes)?;
    let profile = profiler.finish(kind)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    projected_transport_output_resource_facts(&profile)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests;
