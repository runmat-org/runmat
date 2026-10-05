//! Fenced, bounded original checkpoint import; no solve or reconstructed fields.
use crate::{
    decode_fluid_transient_task_source, import_analysis_objects, import_analysis_output_set,
    AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects, PreparedFluidTransientTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind as Kind, FluidTransientCheckpoint,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};

pub struct ImportedFluidTransientCheckpoint {
    pub checkpoint: FluidTransientCheckpoint,
    pub objects: PreparedAnalysisObjects,
}

pub fn import_fluid_transient_checkpoint(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    task: &PreparedFluidTransientTask,
) -> AnalysisBridgeResult<ImportedFluidTransientCheckpoint> {
    let original =
        decode_fluid_transient_task_source(&task.host, &task.inputs, task.output_policy)?;
    if original.resource_plan != task.resource_plan {
        return Err(invalid("fluid transient imported resource source differs"));
    }
    // Reuse physical preparation for exact plan/output rederivation. It never solves.
    let prepared = original.prepare(task.output_policy)?;
    response.validate_against(&task.program.request)?;
    let output_set = import_analysis_output_set(source, response, &task.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(invalid(
            "fluid transient requires exactly one checkpoint output",
        ));
    };
    if output.kind != Kind::FluidTransientCheckpoint {
        return Err(invalid("fluid transient output kind differs"));
    }
    let facts = task
        .resource_plan
        .input
        .demand
        .output_facts(Kind::FluidTransientCheckpoint)
        .ok_or_else(|| invalid("fluid transient output budget missing"))?;
    let mut limits = task.output_policy.inventory_limits();
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(facts.transport_closure_bytes_upper_bound.0);
    let objects = import_analysis_objects(
        source,
        &output.root,
        Kind::FluidTransientCheckpoint,
        output.logical_identity,
        output.payload_digest,
        &task.host.artifact_access,
        limits,
    )?;
    if objects.manifest.encoded_length > facts.encoded_payload_bytes_upper_bound.0 {
        return Err(invalid(
            "fluid transient checkpoint exceeds admitted payload",
        ));
    }
    let mut references =
        objects.unique_references(&task.host.artifact_access, ValueRefKind::ResultObject)?;
    references.push(output_set.root);
    if references != output_set.result_objects {
        return Err(invalid(
            "fluid transient output inventory differs from exact closure",
        ));
    }
    let checkpoint = FluidTransientCheckpoint::from_canonical_bytes(&objects.payload())?;
    if checkpoint.identity()? != objects.manifest.logical_identity {
        return Err(invalid("fluid transient checkpoint identity differs"));
    }
    prepared.recover_output(&checkpoint)?;
    Ok(ImportedFluidTransientCheckpoint {
        checkpoint,
        objects,
    })
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
