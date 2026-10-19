//! Original thermal-solve execution accounting; not whole-study resource evidence.
use crate::AnalysisPhaseExecutionEvidence;
use crate::{
    import_thermal_result, prepare_planned_resource_graph_usage, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisOutputPolicy, ImportedThermalResult, PreparedResourceUsage,
    PreparedThermalTask,
};
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind, planning::resource::FeaResourcePhase,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};

pub struct ImportedThermalCompletion {
    pub result: ImportedThermalResult,
    /// Every original attempt of the admitted thermal solve, excluding subsequent post-processing tasks.
    pub solve_usage: PreparedResourceUsage,
}

/// Import a scheduler-returned response and its associated accounting together.
///
/// Complete original task history binds the delivered response to its invocation.
/// Unknown earlier consumption blocks complete solve usage; this is internal lineage,
/// not trusted worker attestation or measured utilization/RSS.
pub fn import_thermal_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    accounting: AnalysisPhaseExecutionEvidence,
    prepared: &PreparedThermalTask,
    output_policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<ImportedThermalCompletion> {
    let roots = prepared
        .inputs
        .iter()
        .map(|input| {
            input.root_reference(&prepared.host.artifact_access, ValueRefKind::DriverObject)
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    prepared.host.validate_inputs(&roots)?;
    accounting.validate_for(&prepared.host, &prepared.resource_plan, &response)?;
    let result =
        import_thermal_result(source, response, prepared, output_policy.inventory_limits())?;
    let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = &result.response
    else {
        return Err(AnalysisBridgeError::Invalid(
            "thermal completion requires externalized results".into(),
        ));
    };
    let bytes = result_objects
        .iter()
        .try_fold(0_u64, |total, reference| {
            total.checked_add(reference.encoded_length)
        })
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("thermal result byte count overflowed".into())
        })?;
    if bytes != accounting.winning_accounting().resources.artifact_bytes {
        return Err(AnalysisBridgeError::Identity(
            "thermal accounting artifact bytes differ from the validated result closure",
        ));
    }
    let plans = prepared
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    let [plan] = plans.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "thermal completion requires one admitted plan".into(),
        ));
    };
    if prepared.resource_plan.input.phase
        != FeaResourcePhase::ThermalSolve(result.manifest.source_workload)
        || prepared.resource_plan.input.topology_digest() != result.manifest.topology_digest
        || plan.manifest.logical_identity != prepared.resource_plan.identity.digest()
        || plan.payload() != prepared.resource_plan.canonical_bytes()?
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal completion resource plan differs from the admitted solve",
        ));
    }
    let graph = accounting.resource_graph(&prepared.resource_plan)?;
    let solve_usage = prepare_planned_resource_graph_usage(
        &prepared.resource_plan,
        prepared.host.workload.identity.digest(),
        result.manifest.identity().digest(),
        &graph,
        &prepared.host.artifact_access,
        output_policy.chunk_bytes(),
        output_policy.inventory_limits(),
    )?;
    Ok(ImportedThermalCompletion {
        result,
        solve_usage,
    })
}
