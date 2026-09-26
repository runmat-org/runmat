//! Solve-attempt accounting for thermal results; not whole-study resource evidence.
use crate::{
    import_thermal_result, prepare_planned_resource_graph_usage, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisOutputPolicy, ImportedThermalResult, PlannedResourceUsageAttempt,
    PlannedResourceUsageGraph, PreparedResourceUsage, PreparedThermalTask,
};
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind, planning::resource::FeaResourcePhase,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};
use runmat_execution_runner::AttemptResourceAccounting;

pub struct ImportedThermalCompletion {
    pub result: ImportedThermalResult,
    /// Only the admitted thermal solve attempt, excluding subsequent post-processing tasks.
    pub solve_usage: PreparedResourceUsage,
}

/// Import a scheduler-returned response and its associated accounting together.
///
/// The caller must preserve the scheduler/supervisor's response-accounting association.
/// Accounting contains attempt and worker identities, not a task identity, so this adapter
/// cannot independently authenticate that association. It validates the closed result inventory,
/// reconciles its encoded bytes, and binds usage to the admitted plan and semantic result.
pub fn import_thermal_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    accounting: AttemptResourceAccounting,
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
    accounting.validate().map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid thermal attempt accounting: {error}"))
    })?;
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
    if bytes != accounting.resources.artifact_bytes {
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
    let wall_millis = accounting.resources.usage.wall_millis;
    let graph = PlannedResourceUsageGraph::new(
        vec![PlannedResourceUsageAttempt {
            plan: prepared.resource_plan.clone(),
            accounting,
        }],
        wall_millis,
    )?;
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
