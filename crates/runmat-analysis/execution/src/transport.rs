use std::collections::BTreeMap;

use runmat_execution::identity::ValueId;
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{
    ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionRequest,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

/// A portable analysis request and the complete input-object inventory it needs. The input bytes
/// belong to encrypted transport; the executable request retains only immutable references.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAnalysisProgram {
    pub request: ProgramExecutionRequest,
    pub inputs: ProgramExecutionInputs,
}

pub fn prepare_analysis_program(
    host: &AnalysisHostWorkload,
    program_revision: ProgramRevision,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<PreparedAnalysisProgram> {
    host.validate()?;
    if inputs.len() != host.workload.inputs.len() {
        return Err(AnalysisBridgeError::Invalid(
            "analysis object count differs from the workload input contract".into(),
        ));
    }

    let mut roots = Vec::with_capacity(inputs.len());
    let mut inventory = BTreeMap::<ValueId, ProgramExecutionObject>::new();
    for (prepared, contract) in inputs.iter().zip(&host.workload.inputs) {
        if prepared.manifest.kind != contract.kind
            || prepared.manifest.logical_identity != contract.logical_identity
            || prepared.manifest.payload_digest != contract.payload_digest
            || prepared.root.digest != contract.manifest_digest
            || prepared.root.encoded_length != contract.manifest_encoded_length
            || contract.reference_kind != ValueRefKind::DriverObject
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis object closure differs from the workload input contract",
            ));
        }
        let references = prepared.references(&host.artifact_access, ValueRefKind::DriverObject)?;
        let root = references
            .first()
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "analysis input closure has no manifest root",
            ))?;
        roots.push(root);
        for (reference, object) in references.into_iter().zip(&prepared.objects) {
            let transported = ProgramExecutionObject {
                reference,
                bytes: object.bytes.clone(),
            };
            transported.validate()?;
            match inventory.get(&transported.reference.id) {
                Some(existing) if existing != &transported => {
                    return Err(AnalysisBridgeError::Identity(
                        "analysis input object id collides with different content",
                    ))
                }
                Some(_) => {}
                None => {
                    inventory.insert(transported.reference.id, transported);
                }
            }
        }
    }

    let request = host.program_request(program_revision, &roots)?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: request.arguments.clone(),
        objects: inventory.into_values().collect(),
    };
    inputs.validate()?;
    Ok(PreparedAnalysisProgram { request, inputs })
}
