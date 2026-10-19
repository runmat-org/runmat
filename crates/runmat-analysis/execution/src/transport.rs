use std::collections::BTreeMap;

use runmat_execution::identity::ValueId;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{
    LogicalObject, ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionObjectLocation,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

/// A portable analysis request and the complete input-object inventory it needs. The input bytes
/// belong to encrypted transport; the executable request retains only immutable references.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAnalysisProgram {
    pub request: ProgramExecutionRequest,
    pub objects: Vec<ProgramExecutionObject>,
    logical_objects: Vec<LogicalObject>,
}

impl PreparedAnalysisProgram {
    /// Requested storage of both owned transport inventories and immutable request metadata.
    /// These buffers coexist on a coordinating driver; this is not measured process memory.
    pub(crate) fn retained_storage_bytes(&self) -> AnalysisBridgeResult<u64> {
        let metadata = runmat_canonical_codec::profile_serialized(
            &self.request,
            runmat_canonical_codec::CanonicalLimits::new(
                1024 * 1024 * 1024,
                100_000_000,
                65_536,
                64,
            ),
        )?;
        let mut bytes = crate::object::logical_objects_storage(&self.logical_objects)?
            .checked_add(metadata.materialization_resident_bytes_estimate)
            .and_then(|n| {
                n.checked_add(
                    (self.objects.capacity() as u64)
                        .checked_mul(std::mem::size_of::<ProgramExecutionObject>() as u64)?,
                )
            })
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("retained program storage overflow".into())
            })?;
        for object in &self.objects {
            let reference = runmat_canonical_codec::profile_serialized(
                &object.reference,
                runmat_canonical_codec::CanonicalLimits::new(
                    1024 * 1024 * 1024,
                    100_000_000,
                    65_536,
                    64,
                ),
            )?;
            bytes = bytes
                .checked_add(object.bytes.capacity() as u64)
                .and_then(|n| n.checked_add(reference.materialization_resident_bytes_estimate))
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("retained program storage overflow".into())
                })?;
        }
        Ok(bytes)
    }
    pub fn write_to(
        &self,
        store: &mut impl runmat_execution_artifact::cache::CacheExport,
    ) -> AnalysisBridgeResult<()> {
        for object in &self.logical_objects {
            store.write_verified(object)?;
        }
        Ok(())
    }

    pub fn inputs(
        &self,
        locations: Vec<ProgramExecutionObjectLocation>,
    ) -> AnalysisBridgeResult<ProgramExecutionInputs> {
        if locations.len() != self.objects.len()
            || locations
                .iter()
                .zip(&self.objects)
                .any(|(location, object)| location.reference != object.reference)
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis input locations differ from the prepared object inventory",
            ));
        }
        let inputs = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
            arguments: self.request.arguments.clone(),
            objects: locations,
        };
        inputs.validate()?;
        Ok(inputs)
    }
}

/// Shared immutable input binding for preparation, admission and cold accuracy import.
/// It compares borrowed descriptors only; payload authentication belongs to the object owner.
pub(crate) fn validate_prepared_analysis_inputs(
    host: &AnalysisHostWorkload,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    host.validate()?;
    if inputs.len() != host.workload.inputs.len() {
        return Err(AnalysisBridgeError::Invalid(
            "analysis object count differs from the workload input contract".into(),
        ));
    }
    if inputs
        .iter()
        .zip(&host.workload.inputs)
        .any(|(prepared, contract)| {
            prepared.manifest.kind != contract.kind
                || prepared.manifest.logical_identity != contract.logical_identity
                || prepared.manifest.payload_digest != contract.payload_digest
                || prepared.root.digest != contract.manifest_digest
                || prepared.root.encoded_length != contract.manifest_encoded_length
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis object closure differs from the workload input contract",
        ));
    }
    Ok(())
}

pub fn prepare_analysis_program(
    host: &AnalysisHostWorkload,
    program_revision: ProgramRevision,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<PreparedAnalysisProgram> {
    validate_prepared_analysis_inputs(host, inputs)?;

    let mut roots = Vec::with_capacity(inputs.len());
    let mut inventory = BTreeMap::<ValueId, ProgramExecutionObject>::new();
    let mut logical_inventory = BTreeMap::<runmat_execution::Digest, LogicalObject>::new();
    for (prepared, contract) in inputs.iter().zip(&host.workload.inputs) {
        let references = prepared.references(&host.artifact_access, contract.reference_kind)?;
        let root = references
            .first()
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "analysis input closure has no manifest root",
            ))?;
        roots.push(root);
        for (reference, object) in references.into_iter().zip(&prepared.objects) {
            logical_inventory
                .entry(object.descriptor.digest)
                .or_insert_with(|| object.clone());
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

    let invocation = host.prepare_program_request(program_revision, &roots)?;
    let (request, descriptor) = invocation.into_parts();
    let descriptor_reference = match request.arguments.first() {
        Some(runmat_execution::value::ValuePayload::Object(reference)) => reference.as_ref(),
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "analysis host invocation has no descriptor root",
            ))
        }
    };
    let transported = ProgramExecutionObject {
        reference: descriptor_reference.clone(),
        bytes: descriptor.bytes.clone(),
    };
    transported.validate()?;
    inventory.insert(transported.reference.id, transported);
    logical_inventory.insert(descriptor.descriptor.digest, descriptor);
    Ok(PreparedAnalysisProgram {
        request,
        objects: inventory.into_values().collect(),
        logical_objects: logical_inventory.into_values().collect(),
    })
}
