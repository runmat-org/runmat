//! Shared original source closures and exact transient worker resource rederivation.
mod source;
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, AnalysisTransportProfiler, FluidTransientBlockWorkload,
    },
    planning::resource::{
        prepare_fluid_transient_resources, FeaResourcePlan, LocalExecutionPlan,
        PreparedPhaseResourceFacts,
    },
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
pub use source::{decode_fluid_transient_task_source, DecodedFluidTransientTaskSource};
pub struct PreparedFluidTransientInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
pub fn project_fluid_transient_checkpoint_output(
    bytes: u64,
    resident: runmat_analysis_fea::planning::resource::ByteCount,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(bytes)?;
    let profile = profiler.finish(Kind::FluidTransientCheckpoint)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    let mut facts = projected_transport_output_resource_facts(&profile)?;
    facts.construction_resident_bytes_admission_budget = facts
        .construction_resident_bytes_admission_budget
        .max(resident);
    Ok(facts)
}
pub fn prepare_fluid_transient_inputs(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
) -> AnalysisBridgeResult<PreparedFluidTransientInputs> {
    let resources = prepare_fluid_transient_resources(request, mesh)?;
    let projection = resources.projection();
    let output = project_fluid_transient_checkpoint_output(
        projection.checkpoint_encoded_bytes.0,
        projection.checkpoint_resident_bytes,
        policy,
    )?;
    let inputs = vec![
        prepare_analysis_objects(
            Kind::FluidTransientBlockWorkload,
            request.identity()?.digest(),
            &request.canonical_bytes()?,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?,
        prepare_analysis_objects(
            Kind::Mesh,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            &mesh.canonical_encode().map_err(invalid)?,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?,
    ];
    let resource_plan = resources.build(
        request,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts(&inputs)?,
            output,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?;
    Ok(PreparedFluidTransientInputs {
        inputs,
        resource_plan,
    })
}
pub fn rederive_fluid_transient_resource_plan(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let resources = prepare_fluid_transient_resources(request, mesh)?;
    require_fluid_transient_source_closure(request, mesh, inputs, policy)?;
    let projection = resources.projection();
    let output = project_fluid_transient_checkpoint_output(
        projection.checkpoint_encoded_bytes.0,
        projection.checkpoint_resident_bytes,
        policy,
    )?;
    let actual = inputs
        .iter()
        .find(|i| i.manifest.kind == Kind::ResourcePlan)
        .ok_or_else(|| invalid("fluid transient resource plan missing"))?;
    let declared = FeaResourcePlan::from_canonical_bytes(&actual.payload())?;
    let resource_plan = resources.build(
        request,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output,
            output_set: projected_output_set_resource_facts(),
        },
        LocalExecutionPlan::from_cpu_millicores(declared.request.cpu_millicores)
            .map_err(invalid)?,
    )?;
    let expected = prepare_analysis_objects(
        Kind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    if actual != &expected {
        return Err(invalid("fluid transient resource source differs"));
    }
    Ok(resource_plan)
}

pub fn require_fluid_transient_source_closure(
    request: &FluidTransientBlockWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<()> {
    let kinds = [
        Kind::FluidTransientBlockWorkload,
        Kind::Mesh,
        Kind::ResourcePlan,
    ];
    if inputs.len() != 3
        || kinds
            .iter()
            .any(|kind| inputs.iter().filter(|i| i.manifest.kind == *kind).count() != 1)
    {
        return Err(invalid(
            "fluid transient requires exact original request/mesh/plan closure",
        ));
    }
    for input in inputs {
        let (identity, bytes) = match input.manifest.kind {
            Kind::FluidTransientBlockWorkload => {
                (request.identity()?.digest(), request.canonical_bytes()?)
            }
            Kind::Mesh => (
                Digest::from_bytes(*mesh.canonical_digest.bytes()),
                mesh.canonical_encode().map_err(invalid)?,
            ),
            Kind::ResourcePlan => continue,
            _ => return Err(invalid("unexpected transient source")),
        };
        let expected = prepare_analysis_objects(
            input.manifest.kind,
            identity,
            &bytes,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?;
        if input != &expected {
            return Err(invalid("fluid transient source or chunk policy differs"));
        }
    }
    let actual = inputs
        .iter()
        .find(|i| i.manifest.kind == Kind::ResourcePlan)
        .unwrap();
    let plan = crate::decode_admitted_resource_plan(actual, policy.inventory_limits())?;
    let expected = prepare_analysis_objects(
        Kind::ResourcePlan,
        plan.identity.digest(),
        &plan.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    if actual != &expected {
        return Err(invalid("fluid transient resource-plan closure differs"));
    }
    Ok(())
}
