//! Shared transport reservations for original steady finite-volume fluid work.
//! This prepares inputs, not a native executable task or semantic publication.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, AnalysisTransportProfiler, FluidSteadyWorkload},
    planning::resource::{
        build_fluid_steady_resource_plan_with, FeaResourcePlan, FluidSteadyResourceProjection,
        PreparedFluidSteadyResourceFacts,
    },
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct PreparedFluidSteadyInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

pub fn prepare_fluid_steady_inputs(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
) -> AnalysisBridgeResult<PreparedFluidSteadyInputs> {
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
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
    let resource_plan = build_fluid_steady_resource_plan_with(workload, mesh, |projection| {
        Ok::<_, AnalysisBridgeError>(PreparedFluidSteadyResourceFacts {
            local_execution,
            inputs: prepared_input_resource_facts(&inputs)?,
            result_output: projected_result(projection, policy)?,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::FLUID_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            output_set: projected_output_set_resource_facts(),
        })
    })?;
    Ok(PreparedFluidSteadyInputs {
        inputs,
        resource_plan,
    })
}

/// Rebuild the worker's resource authority using imported workload/mesh and actual input
/// manifests. The supplied plan is itself checked, not simply omitted from accounting.
pub fn rederive_fluid_steady_resource_plan(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    // The projection (the original raw count/profile ceiling) precedes source encoding.
    use AnalysisNumericalObjectKind::{FluidSteadyWorkload, Mesh, ResourcePlan};
    let mut imported_plan = None;
    let plan = build_fluid_steady_resource_plan_with(workload, mesh, |projection| {
        if inputs.len() != 3
            || [FluidSteadyWorkload, Mesh, ResourcePlan]
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
                "Fluid inputs require exactly workload, mesh and resource plan",
            ));
        }
        for input in inputs {
            input.manifest.validate()?;
            let (expected, bytes) = match input.manifest.kind {
                FluidSteadyWorkload => (workload.identity()?.digest(), workload.canonical_bytes()?),
                Mesh => (
                    Digest::from_bytes(*mesh.canonical_digest.bytes()),
                    mesh.canonical_encode().map_err(invalid)?,
                ),
                ResourcePlan => continue,
                _ => return Err(invalid("unexpected Fluid input")),
            };
            if input.manifest.logical_identity != expected {
                return Err(invalid("Fluid imported source identity differs"));
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
                    "Fluid input closure differs from canonical source or chunk policy",
                ));
            }
        }
        let input = inputs
            .iter()
            .find(|input| input.manifest.kind == ResourcePlan)
            .ok_or_else(|| invalid("Fluid resource plan missing"))?;
        let imported = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
        let facts = PreparedFluidSteadyResourceFacts {
            local_execution: LocalExecutionPlan::from_cpu_millicores(
                imported.request.cpu_millicores,
            )
            .map_err(invalid)?,
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            result_output: projected_result(projection, policy)?,
            receipt_output: projected_output(
                AnalysisNumericalObjectKind::NumericalReceipt,
                runmat_analysis_fea::execution::FLUID_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            output_set: projected_output_set_resource_facts(),
        };
        imported_plan = Some((input, imported));
        Ok::<_, AnalysisBridgeError>(facts)
    })?;
    let (input, imported) = imported_plan.expect("the plan builder prepared its facts");
    if imported != plan || input.manifest.logical_identity != plan.identity.digest() {
        return Err(invalid(
            "Fluid resource plan disagrees with imported inputs or output policy",
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
            "Fluid resource-plan closure differs from canonical plan",
        ));
    }
    Ok(plan)
}

fn projected_result(
    projection: &FluidSteadyResourceProjection,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let mut facts = projected_output(
        AnalysisNumericalObjectKind::SolverResult,
        projection.result_payload_bytes_upper_bound.0,
        policy,
    )?;
    // Transport staging is not the typed original-result allocation envelope.
    facts.construction_resident_bytes_admission_budget = facts
        .construction_resident_bytes_admission_budget
        .max(projection.result_materialization_resident_bytes);
    Ok(facts)
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
    let mut facts = projected_transport_output_resource_facts(&profile)?;
    if kind == AnalysisNumericalObjectKind::NumericalReceipt {
        facts.construction_resident_bytes_admission_budget.0 =
            facts.construction_resident_bytes_admission_budget.0.max(
                runmat_analysis_fea::execution::fluid_receipt_encoding_profile()?
                    .materialization_resident_bytes_estimate,
            );
    }
    Ok(facts)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}

#[cfg(test)]
mod tests;
