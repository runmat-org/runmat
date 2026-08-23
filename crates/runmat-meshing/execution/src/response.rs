//! Bounded host response containing references only; stage bytes remain in the object store.

use std::collections::BTreeSet;

use runmat_execution::value::{ValueLimits, ValuePayload, ValueRef, ValueRefKind};
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingCanonicalLimits, MeshingFailure, MeshingStageEvidence,
    MeshingWorkloadResult, StableDigest,
};
use serde::{Deserialize, Serialize};

use crate::{
    CompletedMeshingStage, MeshingArtifactAccess, MeshingExecutionError, MeshingExecutionResult,
    MeshingHostWorkload, MeshingSerialExecutionError, MESHING_STAGE_MANIFEST_MEDIA_TYPE,
};

pub const MESHING_HOST_RESPONSE_SCHEMA_VERSION: u16 = 3;
const MAX_RESULT_OBJECTS: usize = 65_538;
const STAGE_MANIFEST_SCHEMA: &str = "runmat.meshing.stage-manifest.v2";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeshingHostResponse {
    Validated {
        schema_version: u16,
        stage_manifest_digest: StableDigest,
        stage_evidence: Box<MeshingStageEvidence>,
        root: ValueRef,
        result_objects: Vec<ValueRef>,
    },
    Failed {
        schema_version: u16,
        failure: MeshingFailure,
    },
}

impl MeshingHostResponse {
    pub fn completed(
        host: &MeshingHostWorkload,
        completed: &CompletedMeshingStage,
    ) -> MeshingExecutionResult<Self> {
        let MeshingWorkloadResult::Validated {
            stage_manifest_digest,
        } = completed.workload_result()
        else {
            return Err(MeshingExecutionError::Invalid(
                "completed meshing stage does not contain a validated result".into(),
            ));
        };
        let attempt = completed.attempt_success();
        let [ValuePayload::Object(root)] = attempt.outputs.as_slice() else {
            return Err(MeshingExecutionError::Invalid(
                "completed meshing stage must have one externalized root output".into(),
            ));
        };
        let response = Self::Validated {
            schema_version: MESHING_HOST_RESPONSE_SCHEMA_VERSION,
            stage_manifest_digest: *stage_manifest_digest,
            stage_evidence: Box::new(completed.stage_evidence().clone()),
            root: (**root).clone(),
            result_objects: attempt.result_objects,
        };
        response.validate_against(host)?;
        Ok(response)
    }

    pub fn failed(
        host: &MeshingHostWorkload,
        error: &MeshingSerialExecutionError,
    ) -> MeshingExecutionResult<Option<Self>> {
        let Some(MeshingWorkloadResult::Failed { failure }) = error.workload_result() else {
            return Ok(None);
        };
        let response = Self::Failed {
            schema_version: MESHING_HOST_RESPONSE_SCHEMA_VERSION,
            failure,
        };
        response.validate_against(host)?;
        Ok(Some(response))
    }

    pub fn validate_against(&self, host: &MeshingHostWorkload) -> MeshingExecutionResult<()> {
        host.validate()?;
        self.validate_standalone()?;
        match self {
            Self::Validated {
                stage_manifest_digest,
                stage_evidence,
                root,
                result_objects: _,
                ..
            } => {
                MeshingWorkloadResult::Validated {
                    stage_manifest_digest: *stage_manifest_digest,
                }
                .validate_against(&host.workload)?;
                stage_evidence.validate()?;
                if stage_evidence.stage != host.workload.stage
                    || stage_evidence.partition != host.workload.partition
                    || stage_evidence.stage_result_digest != *stage_manifest_digest
                {
                    return Err(MeshingExecutionError::Invalid(
                        "meshing host stage evidence does not bind its workload result".into(),
                    ));
                }
                validate_access(root, &host.artifact_access)?;
            }
            Self::Failed { failure, .. } => {
                MeshingWorkloadResult::Failed {
                    failure: failure.clone(),
                }
                .validate_against(&host.workload)?;
            }
        }
        Ok(())
    }

    pub fn attempt_success(&self) -> Option<runmat_execution_runner::AttemptSuccess> {
        match self {
            Self::Validated {
                root,
                result_objects,
                ..
            } => Some(runmat_execution_runner::AttemptSuccess {
                outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
                result_objects: result_objects.clone(),
            }),
            Self::Failed { .. } => None,
        }
    }

    pub fn program_response(&self) -> runmat_execution_artifact::ProgramExecutionResponse {
        match self {
            Self::Validated {
                root,
                result_objects,
                ..
            } => runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
                outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
                result_objects: result_objects.clone(),
            },
            Self::Failed { failure, .. } => {
                runmat_execution_artifact::ProgramExecutionResponse::Failure {
                    message: failure.to_string(),
                }
            }
        }
    }

    fn validate_standalone(&self) -> MeshingExecutionResult<()> {
        match self {
            Self::Validated {
                schema_version,
                stage_manifest_digest,
                stage_evidence,
                root,
                result_objects,
            } => {
                validate_version(*schema_version)?;
                stage_manifest_digest.validate_nonzero("host response manifest digest")?;
                stage_evidence.validate()?;
                if stage_evidence.stage_result_digest != *stage_manifest_digest {
                    return Err(MeshingExecutionError::Invalid(
                        "host response evidence digest differs from its result manifest".into(),
                    ));
                }
                if root.logical_digest.bytes() != stage_manifest_digest.bytes()
                    || root.kind != ValueRefKind::ResultObject
                    || root.media_type != MESHING_STAGE_MANIFEST_MEDIA_TYPE
                    || root.value_schema != STAGE_MANIFEST_SCHEMA
                    || result_objects.is_empty()
                    || result_objects.len() > MAX_RESULT_OBJECTS
                {
                    return Err(MeshingExecutionError::Invalid(
                        "validated meshing host response has an invalid root or inventory".into(),
                    ));
                }
                let access = MeshingArtifactAccess {
                    authorization_scope: root.authorization_scope.clone(),
                    encryption_context: root.encryption_context,
                };
                access.validate()?;
                let mut value_ids = BTreeSet::new();
                let mut logical_digests = BTreeSet::new();
                let mut contains_root = false;
                for object in result_objects {
                    ValuePayload::Object(Box::new(object.clone()))
                        .validate(ValueLimits::default())?;
                    validate_access(object, &access)?;
                    if object.kind != ValueRefKind::ResultObject
                        || !value_ids.insert(object.id)
                        || !logical_digests.insert(object.logical_digest)
                    {
                        return Err(MeshingExecutionError::Invalid(
                            "meshing host response object inventory is not unique and canonical"
                                .into(),
                        ));
                    }
                    contains_root |= object == root;
                }
                if !contains_root {
                    return Err(MeshingExecutionError::Invalid(
                        "meshing host response inventory omits its root".into(),
                    ));
                }
            }
            Self::Failed {
                schema_version,
                failure,
            } => {
                validate_version(*schema_version)?;
                failure.validate()?;
            }
        }
        Ok(())
    }
}

impl CanonicalMeshingContract for MeshingHostResponse {
    const DOMAIN: &'static str = "analysis.mesh.host-response/v2";
    const LIMITS: MeshingCanonicalLimits = MeshingCanonicalLimits::MANIFEST;

    fn validate_canonical(&self) -> Result<(), runmat_meshing_core::MeshingContractError> {
        self.validate_standalone().map_err(|error| {
            runmat_meshing_core::MeshingContractError::invalid(
                "meshing host response",
                error.to_string(),
            )
        })
    }
}

fn validate_version(schema_version: u16) -> MeshingExecutionResult<()> {
    if schema_version != MESHING_HOST_RESPONSE_SCHEMA_VERSION {
        return Err(MeshingExecutionError::Invalid(
            "unsupported meshing host response schema".into(),
        ));
    }
    Ok(())
}

fn validate_access(
    reference: &ValueRef,
    access: &MeshingArtifactAccess,
) -> MeshingExecutionResult<()> {
    if reference.authorization_scope != access.authorization_scope
        || reference.encryption_context != access.encryption_context
        || reference.id != access.value_id(reference.logical_digest)
        || reference.resident_fence.is_some()
    {
        return Err(MeshingExecutionError::Identity(
            "meshing host response object is outside artifact authority",
        ));
    }
    Ok(())
}
