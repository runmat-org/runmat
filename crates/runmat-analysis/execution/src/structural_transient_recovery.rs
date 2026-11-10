use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionResponse,
};
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, StructuralTransientSequencePreparation};

const CHECKPOINT_PREFIX: &[u8] = b"runmat-structural-transient-execution-checkpoint\0";
const CHECKPOINT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);
pub const STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE: &str =
    "application/vnd.runmat.structural-transient-execution-checkpoint+cbor";
pub const STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA: &str =
    "runmat.structural-transient-execution-checkpoint.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StructuralTransientExecutionCheckpointIdentity(Digest);

impl StructuralTransientExecutionCheckpointIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientExecutionCheckpointWire"
)]
pub struct StructuralTransientExecutionCheckpoint {
    schema_version: u16,
    pub identity: StructuralTransientExecutionCheckpointIdentity,
    preparation_digest: Digest,
    completed_block_responses: Vec<ProgramExecutionResponse>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralTransientRecoveryClosure {
    pub checkpoint: StructuralTransientExecutionCheckpoint,
    pub root: runmat_execution::value::ValueRef,
    pub objects: Vec<ProgramExecutionObject>,
}

impl StructuralTransientRecoveryClosure {
    pub fn from_objects(
        artifact_access: &ArtifactAccess,
        root: ValueRef,
        mut objects: Vec<ProgramExecutionObject>,
    ) -> AnalysisBridgeResult<Self> {
        artifact_access.validate_reference(
            &root,
            ValueRefKind::ResultObject,
            STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE,
            STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
        )?;
        let mut by_digest = std::collections::BTreeMap::new();
        for object in &objects {
            object.validate()?;
            match by_digest.insert(object.reference.logical_digest, object.bytes.as_slice()) {
                Some(existing) if existing != object.bytes.as_slice() => {
                    return Err(AnalysisBridgeError::Identity(
                        "transient recovery closure reuses a digest for different content",
                    ))
                }
                _ => {}
            }
        }
        let checkpoint_bytes =
            by_digest
                .get(&root.logical_digest)
                .ok_or(AnalysisBridgeError::Identity(
                    "transient recovery closure omits its checkpoint root",
                ))?;
        let checkpoint =
            StructuralTransientExecutionCheckpoint::from_canonical_bytes(checkpoint_bytes)?;
        objects.sort_by_key(|object| object.reference.id);
        Ok(Self {
            checkpoint,
            root,
            objects,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientExecutionCheckpointWire {
    schema_version: u16,
    identity: StructuralTransientExecutionCheckpointIdentity,
    preparation_digest: Digest,
    completed_block_responses: Vec<ProgramExecutionResponse>,
}

impl TryFrom<StructuralTransientExecutionCheckpointWire>
    for StructuralTransientExecutionCheckpoint
{
    type Error = AnalysisBridgeError;

    fn try_from(wire: StructuralTransientExecutionCheckpointWire) -> Result<Self, Self::Error> {
        let checkpoint = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            preparation_digest: wire.preparation_digest,
            completed_block_responses: wire.completed_block_responses,
        };
        checkpoint.validate_content()?;
        Ok(checkpoint)
    }
}

impl StructuralTransientExecutionCheckpoint {
    pub fn completed_block_count(&self) -> usize {
        self.completed_block_responses.len()
    }

    pub(crate) fn new(
        preparation: &StructuralTransientSequencePreparation,
        completed_block_responses: Vec<ProgramExecutionResponse>,
    ) -> AnalysisBridgeResult<Self> {
        let preparation_digest = preparation_digest(preparation)?;
        let identity = checkpoint_identity(1, preparation_digest, &completed_block_responses)?;
        let checkpoint = Self {
            schema_version: 1,
            identity,
            preparation_digest,
            completed_block_responses,
        };
        checkpoint.validate_for(preparation)?;
        Ok(checkpoint)
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        self.validate_content()?;
        encode_contract(
            CHECKPOINT_PREFIX,
            "structural-transient-execution-checkpoint",
            self,
            CHECKPOINT_LIMITS,
        )
        .map_err(Into::into)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisBridgeResult<Self> {
        let checkpoint: Self = decode_contract(
            CHECKPOINT_PREFIX,
            "structural-transient-execution-checkpoint",
            bytes,
            CHECKPOINT_LIMITS,
        )?;
        checkpoint.validate_content()?;
        Ok(checkpoint)
    }

    pub(crate) fn validate_for(
        &self,
        preparation: &StructuralTransientSequencePreparation,
    ) -> AnalysisBridgeResult<()> {
        self.validate_content()?;
        if self.preparation_digest != preparation_digest(preparation)?
            || self.completed_block_responses.len() > preparation.terminal_schedule_ordinals.len()
        {
            return Err(AnalysisBridgeError::Identity(
                "transient execution checkpoint differs from its exact preparation",
            ));
        }
        Ok(())
    }

    pub(crate) fn responses(&self) -> &[ProgramExecutionResponse] {
        &self.completed_block_responses
    }

    pub(crate) fn prepare_closure(
        &self,
        source: &impl CacheImport,
        preparation: &StructuralTransientSequencePreparation,
    ) -> AnalysisBridgeResult<StructuralTransientRecoveryClosure> {
        self.validate_for(preparation)?;
        let checkpoint_object = LogicalObject::new(
            ObjectNamespace::ResultValue,
            "analysis/structural-transient-execution-checkpoint",
            STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_MEDIA_TYPE,
            self.canonical_bytes()?,
        )?;
        let root = preparation.artifact_access.reference(
            &checkpoint_object,
            ValueRefKind::ResultObject,
            STRUCTURAL_TRANSIENT_EXECUTION_CHECKPOINT_VALUE_SCHEMA,
        )?;
        let mut by_id = std::collections::BTreeMap::new();
        by_id.insert(
            root.id,
            ProgramExecutionObject {
                reference: root.clone(),
                bytes: checkpoint_object.bytes,
            },
        );
        for response in &self.completed_block_responses {
            let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = response
            else {
                return Err(invalid(
                    "transient recovery closure contains a non-externalized block response",
                ));
            };
            for reference in result_objects {
                let bytes = source.read_verified(reference.logical_digest)?.ok_or(
                    AnalysisBridgeError::Identity(
                        "transient recovery closure omits a block result object",
                    ),
                )?;
                let object = ProgramExecutionObject {
                    reference: reference.clone(),
                    bytes,
                };
                object.validate()?;
                match by_id.get(&reference.id) {
                    Some(existing) if existing != &object => {
                        return Err(AnalysisBridgeError::Identity(
                            "transient recovery closure reuses an object id for different content",
                        ))
                    }
                    Some(_) => {}
                    None => {
                        by_id.insert(reference.id, object);
                    }
                }
            }
        }
        Ok(StructuralTransientRecoveryClosure {
            checkpoint: self.clone(),
            root,
            objects: by_id.into_values().collect(),
        })
    }

    fn validate_content(&self) -> AnalysisBridgeResult<()> {
        if self.schema_version != 1
            || self.preparation_digest.bytes() == &[0; 32]
            || self.completed_block_responses.is_empty()
            || self.completed_block_responses.iter().any(|response| {
                let ProgramExecutionResponse::ExternalizedSuccess {
                    outputs,
                    result_objects,
                } = response
                else {
                    return true;
                };
                outputs.is_empty()
                    || result_objects.is_empty()
                    || result_objects
                        .iter()
                        .map(|reference| reference.id)
                        .collect::<BTreeSet<_>>()
                        .len()
                        != result_objects.len()
            })
        {
            return Err(invalid(
                "transient execution checkpoint has invalid schema or block responses",
            ));
        }
        if self.identity
            != checkpoint_identity(
                self.schema_version,
                self.preparation_digest,
                &self.completed_block_responses,
            )?
        {
            return Err(AnalysisBridgeError::Identity(
                "transient execution checkpoint identity does not converge",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct PreparationIdentityInput<'a> {
    scope: runmat_analysis_fea::execution::StructuralTransientCheckpointScopeIdentity,
    workload_manifest: &'a runmat_analysis_fea::execution::AnalysisObjectManifest,
    workload_root: &'a runmat_execution_artifact::ObjectDescriptor,
    mesh_manifest: &'a runmat_analysis_fea::execution::AnalysisObjectManifest,
    mesh_root: &'a runmat_execution_artifact::ObjectDescriptor,
    terminal_schedule_ordinals: &'a [u64],
    initial_dependencies: &'a [runmat_analysis_fea::execution::AnalysisTaskIdentity],
    artifact_access: &'a runmat_execution_artifact::ArtifactAccess,
    program_revision: &'a runmat_execution::ProgramRevision,
    chunk_bytes: u64,
    maximum_objects: u64,
    maximum_object_bytes: u64,
    maximum_total_bytes: u64,
}

pub(crate) fn preparation_digest(
    preparation: &StructuralTransientSequencePreparation,
) -> AnalysisBridgeResult<Digest> {
    let input = PreparationIdentityInput {
        scope: preparation.scope.identity()?,
        workload_manifest: &preparation.workload.manifest,
        workload_root: &preparation.workload.root,
        mesh_manifest: &preparation.mesh.manifest,
        mesh_root: &preparation.mesh.root,
        terminal_schedule_ordinals: &preparation.terminal_schedule_ordinals,
        initial_dependencies: &preparation.initial_dependencies,
        artifact_access: &preparation.artifact_access,
        program_revision: &preparation.program_revision,
        chunk_bytes: u64::try_from(preparation.chunk_bytes)
            .map_err(|_| invalid("transient chunk size exceeds u64"))?,
        maximum_objects: u64::try_from(preparation.inventory_limits.max_objects)
            .map_err(|_| invalid("transient inventory count exceeds u64"))?,
        maximum_object_bytes: preparation.inventory_limits.max_object_bytes,
        maximum_total_bytes: preparation.inventory_limits.max_total_bytes,
    };
    let bytes = encode_contract(
        b"runmat-structural-transient-execution-preparation\0",
        "structural-transient-execution-preparation",
        &input,
        CHECKPOINT_LIMITS,
    )?;
    Ok(Digest::sha256(bytes))
}

fn checkpoint_identity(
    schema_version: u16,
    preparation_digest: Digest,
    responses: &[ProgramExecutionResponse],
) -> AnalysisBridgeResult<StructuralTransientExecutionCheckpointIdentity> {
    let bytes = encode_contract(
        b"runmat-structural-transient-execution-checkpoint-identity\0",
        "structural-transient-execution-checkpoint-identity",
        &(schema_version, preparation_digest, responses),
        CHECKPOINT_LIMITS,
    )?;
    Ok(StructuralTransientExecutionCheckpointIdentity(
        Digest::sha256(bytes),
    ))
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
