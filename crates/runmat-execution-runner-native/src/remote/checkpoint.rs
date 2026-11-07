use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionObjectLocation, ProgramExecutionRequest,
};
use runmat_execution_transport_native::control::{CheckpointDisposition, DriverAuthority};
use serde::{Deserialize, Serialize};

use crate::{NativeExecutionError, NativeExecutionResult};

const REMOTE_DRIVER_CHECKPOINT_PREFIX: &[u8] = b"runmat-remote-driver-checkpoint\0";
const REMOTE_DRIVER_CHECKPOINT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(256 * 1024, 4096, 1024, 16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum RemoteDriverCheckpointPhase {
    ReadyToExecute,
    ReplayableExecutionStarted,
    EffectsMayHaveStarted,
}

impl RemoteDriverCheckpointPhase {
    pub(super) const fn disposition(self) -> CheckpointDisposition {
        match self {
            Self::ReadyToExecute | Self::ReplayableExecutionStarted => {
                CheckpointDisposition::ResumeSafe
            }
            Self::EffectsMayHaveStarted => CheckpointDisposition::Indeterminate,
        }
    }

    const fn is_resume_safe(self) -> bool {
        matches!(
            self,
            Self::ReadyToExecute | Self::ReplayableExecutionStarted
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RemoteDriverCheckpoint {
    schema_version: u16,
    pub(super) sequence: u64,
    source_driver_fence: u64,
    run_id: String,
    project_revision: String,
    request_digest: Digest,
    pub(super) phase: RemoteDriverCheckpointPhase,
    pub(super) closure_catalog: Option<ProgramExecutionObjectLocation>,
    pub(super) pool_catalog: Option<ProgramExecutionObjectLocation>,
}

impl RemoteDriverCheckpoint {
    pub(super) fn source_driver_fence(&self) -> u64 {
        self.source_driver_fence
    }
    pub(super) fn new(
        sequence: u64,
        authority: &DriverAuthority,
        project_revision: &str,
        request: &ProgramExecutionRequest,
        phase: RemoteDriverCheckpointPhase,
        closure_catalog: Option<ProgramExecutionObjectLocation>,
    ) -> NativeExecutionResult<Self> {
        Self::for_program(
            sequence,
            authority,
            project_revision,
            request_digest(request)?,
            phase,
            closure_catalog,
        )
    }

    /// Bind a checkpoint to an exact admitted program identity: a Direct
    /// request's canonical digest or a Graph Program object's digest.
    pub(super) fn for_program(
        sequence: u64,
        authority: &DriverAuthority,
        project_revision: &str,
        program_digest: Digest,
        phase: RemoteDriverCheckpointPhase,
        closure_catalog: Option<ProgramExecutionObjectLocation>,
    ) -> NativeExecutionResult<Self> {
        let checkpoint = Self {
            schema_version: 2,
            sequence,
            source_driver_fence: authority.fencing_token,
            run_id: authority.run_id.clone(),
            project_revision: project_revision.into(),
            request_digest: program_digest,
            phase,
            closure_catalog,
            pool_catalog: None,
        };
        checkpoint.validate_content()?;
        Ok(checkpoint)
    }

    pub(super) fn with_pool_catalog(
        mut self,
        catalog: Option<ProgramExecutionObjectLocation>,
    ) -> NativeExecutionResult<Self> {
        self.pool_catalog = catalog;
        self.validate_content()?;
        Ok(self)
    }

    #[cfg(test)]
    pub(super) fn validate_for_replacement(
        &self,
        authority: &DriverAuthority,
        project_revision: &str,
        request: &ProgramExecutionRequest,
    ) -> NativeExecutionResult<()> {
        self.validate_for_program(authority, project_revision, request_digest(request)?)
    }

    pub(super) fn validate_for_program(
        &self,
        authority: &DriverAuthority,
        project_revision: &str,
        program_digest: Digest,
    ) -> NativeExecutionResult<()> {
        self.validate_content()?;
        if !self.phase.is_resume_safe()
            || self.source_driver_fence >= authority.fencing_token
            || self.run_id != authority.run_id
            || self.project_revision != project_revision
            || self.request_digest != program_digest
        {
            return Err(protocol(
                "remote driver checkpoint is unsafe, stale, or belongs to another execution",
            ));
        }
        Ok(())
    }

    pub(super) fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate_content()?;
        encode_contract(
            REMOTE_DRIVER_CHECKPOINT_PREFIX,
            "remote-driver-checkpoint",
            self,
            REMOTE_DRIVER_CHECKPOINT_LIMITS,
        )
        .map_err(protocol)
    }

    pub(super) fn from_canonical_bytes(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let checkpoint: Self = decode_contract(
            REMOTE_DRIVER_CHECKPOINT_PREFIX,
            "remote-driver-checkpoint",
            bytes,
            REMOTE_DRIVER_CHECKPOINT_LIMITS,
        )
        .map_err(protocol)?;
        checkpoint.validate_content()?;
        Ok(checkpoint)
    }

    fn validate_content(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 2
            || self.sequence == 0
            || self.source_driver_fence == 0
            || self.run_id.is_empty()
            || self.project_revision.is_empty()
        {
            return Err(protocol("remote driver checkpoint is malformed"));
        }
        for catalog in self.closure_catalog.iter().chain(self.pool_catalog.iter()) {
            catalog.validate().map_err(protocol)?;
            if catalog.reference.kind != ValueRefKind::ResultObject
                || catalog.reference.authorization_scope != self.run_id
                || !self.phase.is_resume_safe()
            {
                return Err(protocol(
                    "remote driver checkpoint closure catalog is outside its safe run authority",
                ));
            }
        }
        Ok(())
    }
}

pub(super) const fn execution_start_phase(
    request: &ProgramExecutionRequest,
) -> RemoteDriverCheckpointPhase {
    // These closed host forms orchestrate deterministic worker tasks. Arbitrary
    // bytecode/native host programs may have effects and cannot claim replay.
    if matches!(
        request.artifact.form,
        ExecutableForm::AnalysisWorkload
            | ExecutableForm::GeometryWorkload
            | ExecutableForm::MeshingWorkload
    ) {
        RemoteDriverCheckpointPhase::ReplayableExecutionStarted
    } else {
        RemoteDriverCheckpointPhase::EffectsMayHaveStarted
    }
}

fn request_digest(request: &ProgramExecutionRequest) -> NativeExecutionResult<Digest> {
    request.canonical_digest().map_err(protocol)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::value::ValueRef;
    use runmat_execution::{OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{
        ProgramArtifact, ProgramBuildRecipe, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };

    use super::*;

    fn digest(label: &str) -> Digest {
        Digest::sha256(label)
    }

    fn authority(fence: u64) -> DriverAuthority {
        DriverAuthority {
            server_url: "https://control.invalid".into(),
            run_id: "run".into(),
            org_id: crate::remote::test_ids::test_org_id(),
            project_id: "project".into(),
            allocation_lease_id: "allocation".into(),
            driver_lease_id: crate::remote::test_ids::test_driver_lease_id(fence),
            fencing_token: fence,
            credential: "credential".into(),
        }
    }

    fn request(form: ExecutableForm) -> ProgramExecutionRequest {
        let revision = ProgramRevision::new(
            digest("graph"),
            digest("source"),
            ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
        )
        .unwrap();
        let entrypoint = match form {
            ExecutableForm::AnalysisWorkload => "analysis_workload",
            ExecutableForm::GeometryWorkload => "geometry_workload",
            _ => "0",
        };
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: entrypoint.into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "checkpoint-test".into(),
            target: ProgramTarget::portable("checkpoint-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(&recipe, form, vec![1]).unwrap();
        ProgramExecutionRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                None,
            ),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    fn closure_catalog(run_id: &str) -> ProgramExecutionObjectLocation {
        ProgramExecutionObjectLocation {
            reference: ValueRef {
                schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
                id: ValueId::derive(&[b"closure-catalog"]),
                logical_digest: digest("closure-catalog"),
                encoded_length: 128,
                media_type: "application/vnd.runmat.execution-object-closure+json".into(),
                value_schema:
                    runmat_execution_artifact::PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA.into(),
                encryption_context: digest("closure-encryption"),
                kind: ValueRefKind::ResultObject,
                authorization_scope: run_id.into(),
                resident_fence: None,
            },
            artifact_id: "closure-artifact".into(),
        }
    }

    #[test]
    fn replacement_accepts_only_an_older_safe_exact_checkpoint() {
        let analysis_request = request(ExecutableForm::AnalysisWorkload);
        let checkpoint = RemoteDriverCheckpoint::new(
            7,
            &authority(2),
            "revision",
            &analysis_request,
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted,
            None,
        )
        .unwrap();
        let decoded =
            RemoteDriverCheckpoint::from_canonical_bytes(&checkpoint.canonical_bytes().unwrap())
                .unwrap();
        decoded
            .validate_for_replacement(&authority(3), "revision", &analysis_request)
            .unwrap();
        assert!(decoded
            .validate_for_replacement(&authority(2), "revision", &analysis_request)
            .is_err());
        assert!(decoded
            .validate_for_replacement(&authority(3), "other", &analysis_request)
            .is_err());
        let other_request = request(ExecutableForm::InterpreterBytecodeV1);
        assert!(decoded
            .validate_for_replacement(&authority(3), "revision", &other_request)
            .is_err());
    }

    #[test]
    fn execution_start_is_resume_safe_for_closed_analysis_geometry_and_meshing_hosts() {
        assert_eq!(
            execution_start_phase(&request(ExecutableForm::AnalysisWorkload)).disposition(),
            CheckpointDisposition::ResumeSafe
        );
        assert_eq!(
            execution_start_phase(&request(ExecutableForm::GeometryWorkload)).disposition(),
            CheckpointDisposition::ResumeSafe
        );
        assert_eq!(
            execution_start_phase(&request(ExecutableForm::MeshingWorkload)).disposition(),
            CheckpointDisposition::ResumeSafe
        );
        assert_eq!(
            execution_start_phase(&request(ExecutableForm::InterpreterBytecodeV1)).disposition(),
            CheckpointDisposition::Indeterminate
        );
    }

    #[test]
    fn recovery_catalog_is_bound_to_a_safe_phase_and_exact_run() {
        let request = request(ExecutableForm::AnalysisWorkload);
        RemoteDriverCheckpoint::new(
            8,
            &authority(2),
            "revision",
            &request,
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted,
            Some(closure_catalog("run")),
        )
        .unwrap();

        assert!(RemoteDriverCheckpoint::new(
            8,
            &authority(2),
            "revision",
            &request,
            RemoteDriverCheckpointPhase::EffectsMayHaveStarted,
            Some(closure_catalog("run")),
        )
        .is_err());
        assert!(RemoteDriverCheckpoint::new(
            8,
            &authority(2),
            "revision",
            &request,
            RemoteDriverCheckpointPhase::ReplayableExecutionStarted,
            Some(closure_catalog("another-run")),
        )
        .is_err());
    }
}
