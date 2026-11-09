use std::collections::BTreeMap;

use runmat_execution::identity::ValueId;
use runmat_execution::value::ValueRef;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{
    LogicalObject, ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionObjectLocation,
    ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

/// A portable analysis request and the complete input-object inventory it needs. The input bytes
/// belong to encrypted transport; the executable request retains only immutable references.
/// Each object's bytes are held once, in the logical inventory; a transported reference names
/// its bytes by position there.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAnalysisProgram {
    pub request: ProgramExecutionRequest,
    references: Vec<(ValueRef, usize)>,
    logical_objects: Vec<LogicalObject>,
}

impl PreparedAnalysisProgram {
    /// Every transported input object, in reference-id order: its reference and its bytes.
    pub fn objects(&self) -> impl ExactSizeIterator<Item = (&ValueRef, &[u8])> + '_ {
        self.references
            .iter()
            .map(|(reference, index)| (reference, self.logical_objects[*index].bytes.as_slice()))
    }

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
                    (self.references.capacity() as u64).checked_mul(std::mem::size_of::<(
                        ValueRef,
                        usize,
                    )>()
                        as u64)?,
                )
            })
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid("retained program storage overflow".into())
            })?;
        for (reference, _) in &self.references {
            let reference = runmat_canonical_codec::profile_serialized(
                reference,
                runmat_canonical_codec::CanonicalLimits::new(
                    1024 * 1024 * 1024,
                    100_000_000,
                    65_536,
                    64,
                ),
            )?;
            bytes = bytes
                .checked_add(reference.materialization_resident_bytes_estimate)
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
        store.write_verified_batch(&self.logical_objects)?;
        Ok(())
    }

    pub fn inputs(
        &self,
        locations: Vec<ProgramExecutionObjectLocation>,
    ) -> AnalysisBridgeResult<ProgramExecutionInputs> {
        if locations.len() != self.references.len()
            || locations
                .iter()
                .zip(&self.references)
                .any(|(location, (reference, _))| &location.reference != reference)
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis input locations differ from the prepared object inventory",
            ));
        }
        let inputs = ProgramExecutionInputs {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: self.request.context.clone(),
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
    // Transported references by id, each naming its bytes by logical digest; the bytes are
    // borrowed from the prepared inputs until the one owned inventory is built below.
    let mut inventory = BTreeMap::<ValueId, (ValueRef, runmat_execution::Digest)>::new();
    let mut logical_inventory = BTreeMap::<runmat_execution::Digest, &LogicalObject>::new();
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
            let digest = object.descriptor.digest;
            let logical = *logical_inventory.entry(digest).or_insert(object);
            validate_transported(&reference, &object.bytes)?;
            match inventory.get(&reference.id) {
                Some((existing, existing_digest))
                    if existing != &reference
                        || logical_inventory[existing_digest].bytes != logical.bytes =>
                {
                    return Err(AnalysisBridgeError::Identity(
                        "analysis input object id collides with different content",
                    ))
                }
                Some(_) => {}
                None => {
                    inventory.insert(reference.id, (reference, digest));
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
    validate_transported(descriptor_reference, &descriptor.bytes)?;
    let descriptor_digest = descriptor.descriptor.digest;
    inventory.insert(
        descriptor_reference.id,
        (descriptor_reference.clone(), descriptor_digest),
    );
    // The one owned copy of every object's bytes.
    let mut descriptor = Some(descriptor);
    let mut positions = BTreeMap::new();
    let mut logical_objects = Vec::with_capacity(logical_inventory.len() + 1);
    let mut owned = logical_inventory
        .into_iter()
        .map(|(digest, object)| (digest, Some(object)))
        .collect::<BTreeMap<_, _>>();
    owned.entry(descriptor_digest).or_insert(None);
    for (digest, object) in owned {
        positions.insert(digest, logical_objects.len());
        logical_objects.push(match object {
            Some(object) => object.clone(),
            None => descriptor.take().expect("descriptor inserted once"),
        });
    }
    Ok(PreparedAnalysisProgram {
        request,
        references: inventory
            .into_values()
            .map(|(reference, digest)| (reference, positions[&digest]))
            .collect(),
        logical_objects,
    })
}

/// Validate one transported object against its reference. Objects are chunks, so the
/// temporary copy is one chunk and is dropped at once.
fn validate_transported(reference: &ValueRef, bytes: &[u8]) -> AnalysisBridgeResult<()> {
    ProgramExecutionObject {
        reference: reference.clone(),
        bytes: bytes.to_vec(),
    }
    .validate()?;
    Ok(())
}
