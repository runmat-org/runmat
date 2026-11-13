//! Shared transport reservations for static scalar electrical numerical work.
//! This prepares inputs, not a native executable task or semantic publication.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, AnalysisTransportProfiler, ElectricalStaticWorkload},
    planning::resource::{
        build_electrical_static_resource_plan, electrical_static_result_payload_bytes_upper_bound,
        FeaResourcePlan, LocalExecutionPlan, PreparedElectricalStaticResourceFacts,
    },
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct PreparedElectricalStaticInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

pub fn prepare_electrical_static_inputs(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
    machine_memory_bytes: u64,
) -> AnalysisBridgeResult<PreparedElectricalStaticInputs> {
    let result_output = projected_result(workload, mesh, policy)?;
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ElectricalStaticWorkload,
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
    let resource_plan = build_electrical_static_resource_plan(
        workload,
        mesh,
        PreparedElectricalStaticResourceFacts {
            local_execution,
            inputs: prepared_input_resource_facts(&inputs)?,
            result_output,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            state_profiles_output: projected_state_profiles(workload, mesh, policy)?,
            output_set: projected_output_set_resource_facts(),
            machine_memory_bytes,
        },
    )?;
    Ok(PreparedElectricalStaticInputs {
        inputs,
        resource_plan,
    })
}

/// Rebuild the worker's resource authority using imported workload/mesh and actual input
/// manifests. The supplied plan is itself checked, not simply omitted from accounting.
pub fn rederive_electrical_static_resource_plan(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    use AnalysisNumericalObjectKind::{ElectricalStaticWorkload, Mesh, ResourcePlan};
    if inputs.len() != 3
        || [ElectricalStaticWorkload, Mesh, ResourcePlan]
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
            "electrical inputs require exactly workload, mesh and resource plan",
        ));
    }
    for input in inputs {
        input.manifest.validate()?;
        let (expected, bytes) = match input.manifest.kind {
            ElectricalStaticWorkload => {
                (workload.identity()?.digest(), workload.canonical_bytes()?)
            }
            Mesh => (
                Digest::from_bytes(*mesh.canonical_digest.bytes()),
                mesh.canonical_encode().map_err(invalid)?,
            ),
            ResourcePlan => continue,
            _ => return Err(invalid("unexpected electrical input")),
        };
        if input.manifest.logical_identity != expected {
            return Err(invalid("electrical imported source identity differs"));
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
                "electrical input closure differs from canonical source or chunk policy",
            ));
        }
    }
    let input = inputs
        .iter()
        .find(|input| input.manifest.kind == ResourcePlan)
        .ok_or_else(|| invalid("electrical resource plan missing"))?;
    let imported = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
    // A replay solves under the budget the plan recorded, not this host's memory.
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::ElectricalStatic(demand) =
        &imported.input.demand
    else {
        return Err(invalid("electrical resource plan has another demand"));
    };
    let recorded_budget = demand.solver_memory_budget.0;
    let plan = build_electrical_static_resource_plan(
        workload,
        mesh,
        PreparedElectricalStaticResourceFacts {
            local_execution: LocalExecutionPlan::from_cpu_millicores(
                imported.request.cpu_millicores,
            )
            .map_err(invalid)?,
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            result_output: projected_result(workload, mesh, policy)?,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            state_profiles_output: projected_state_profiles(workload, mesh, policy)?,
            output_set: projected_output_set_resource_facts(),
            machine_memory_bytes: recorded_budget,
        },
    )?;
    if imported != plan || input.manifest.logical_identity != plan.identity.digest() {
        return Err(invalid(
            "electrical resource plan disagrees with imported inputs or output policy",
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
            "electrical resource-plan closure differs from canonical plan",
        ));
    }
    Ok(plan)
}

fn projected_result(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    projected_output(
        AnalysisNumericalObjectKind::SolverResult,
        electrical_static_result_payload_bytes_upper_bound(workload, mesh)?,
        policy,
    )
}

fn projected_state_profiles(
    workload: &ElectricalStaticWorkload,
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
