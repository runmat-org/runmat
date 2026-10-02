//! Import canonical Fluid fields through the shared output-set/inventory authority.
use crate::{
    import_analysis_objects, import_analysis_output_set, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects, PreparedFluidSteadyTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, FluidNumericalReceiptArtifact, FluidSteadySolverResult,
    FluidSteadyWorkload,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct ImportedFluidSteadyResult {
    pub result: FluidSteadySolverResult,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: FluidNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
}

pub fn import_fluid_steady_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedFluidSteadyTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedFluidSteadyResult> {
    prepared.host.workload.validate()?;
    if prepared.host.workload.kind
        != runmat_analysis_fea::execution::AnalysisWorkKind::SolveFluidSteady
        || prepared.host.workload.outputs
            != [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
            ]
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid task kind or declared output set differs",
        ));
    }
    if prepared.inputs.len() != 3 || prepared.host.workload.inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "Fluid source input inventory differs",
        ));
    }
    let expected_inputs = prepared
        .inputs
        .iter()
        .map(|input| crate::AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        })
        .collect::<Vec<_>>();
    if expected_inputs != prepared.host.workload.inputs {
        return Err(AnalysisBridgeError::Identity(
            "Fluid inputs differ from their immutable task",
        ));
    }
    crate::require_fluid_source_decode_budget(&prepared.inputs, &prepared.resource_plan, limits)?;
    let input = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = prepared
            .inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        let first = matches
            .next()
            .ok_or_else(|| AnalysisBridgeError::Invalid("Fluid task input absent".into()))?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid("duplicate Fluid input".into()));
        }
        Ok(first)
    };
    let workload_input = input(AnalysisNumericalObjectKind::FluidSteadyWorkload)?;
    let mesh_input = input(AnalysisNumericalObjectKind::Mesh)?;
    let plan_input = input(AnalysisNumericalObjectKind::ResourcePlan)?;
    let workload = FluidSteadyWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let rederived = crate::rederive_fluid_steady_resource_plan(
        &workload,
        &mesh,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if rederived != prepared.resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "Fluid task resource policy differs from canonical source authority",
        ));
    }
    let plan = runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
        &plan_input.payload(),
    )?;
    let numerical =
        runmat_analysis_fea::execution::NumericalPlanIdentity::from_fluid_steady_workload(
            workload.identity()?,
            runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?;
    let execution = runmat_analysis_fea::execution::ExecutionPlanIdentity::new(
        numerical,
        plan.identity.digest(),
    )?;
    if plan != prepared.resource_plan
        || plan_input.manifest.logical_identity != plan.identity.digest()
        || prepared.host.workload.execution_plan != execution
        || prepared.host.workload.step != workload.producer_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid plan or producer differs from its task",
        ));
    }
    if workload.identity()?.digest() != workload_input.manifest.logical_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_input.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid source input identity differs",
        ));
    }

    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output, receipt] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "Fluid task requires exact result and receipt outputs".into(),
        ));
    };
    if output.kind != AnalysisNumericalObjectKind::SolverResult
        || receipt.kind != AnalysisNumericalObjectKind::NumericalReceipt
    {
        return Err(AnalysisBridgeError::Identity("Fluid output kinds differ"));
    }
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidSteadySolve(demand) =
        &rederived.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "Fluid output resource family differs",
        ));
    };
    let bounded_limits =
        |facts: &runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts| {
            let mut bounded = limits;
            bounded.max_total_bytes = bounded
                .max_total_bytes
                .min(facts.transport_closure_bytes_upper_bound.0);
            bounded
        };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::SolverResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        bounded_limits(&demand.result_output),
    )?;
    let numerical_receipt_objects = import_analysis_objects(
        source,
        &receipt.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.logical_identity,
        receipt.payload_digest,
        &prepared.host.artifact_access,
        bounded_limits(&demand.receipt_output),
    )?;
    if objects.manifest.encoded_length > demand.result_output.encoded_payload_bytes_upper_bound.0
        || numerical_receipt_objects.manifest.encoded_length
            > demand.receipt_output.encoded_payload_bytes_upper_bound.0
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid outputs exceed admitted payload bounds",
        ));
    }
    let mut references =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    references.extend(
        numerical_receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut references);
    references.push(output_set.root);
    if references != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "Fluid output inventory differs from canonical closure",
        ));
    }
    let result = FluidSteadySolverResult::from_canonical_bytes_for(
        &objects.payload(),
        objects.manifest.payload_digest,
        &workload,
        &mesh,
    )?;
    if result.identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "Fluid result logical identity differs",
        ));
    }
    let numerical_receipt = FluidNumericalReceiptArtifact::from_canonical_bytes_for(
        &numerical_receipt_objects.payload(),
        numerical_receipt_objects.manifest.payload_digest,
        &workload,
        &result,
        &mesh,
    )?;
    if numerical_receipt.identity().digest() != numerical_receipt_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "Fluid receipt logical identity differs",
        ));
    }
    Ok(ImportedFluidSteadyResult {
        result,
        objects,
        numerical_receipt,
        numerical_receipt_objects,
    })
}
