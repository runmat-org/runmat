//! Canonical observation artifact, manifest and exact identity codecs.
use super::*;
/// Canonical observation outputs produced from one recovered-field artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationArtifactWire")]
pub struct ObservationArtifact {
    pub schema_version: u16,
    identity: ObservationArtifactIdentity,
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationArtifactEntry {
    pub result: ObservationResultIdentity,
    pub partition_evidence: ObservationPartitionEvidenceIdentity,
    pub acceleration: Option<ObservationAccelerationSummary>,
    pub observation_id: ObservationId,
    pub sample: RecoveredFieldSample,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub partial_access: PartialAccessPolicy,
}

impl ObservationArtifactEntry {
    /// Admit one retained result against its semantic catalog entry.
    pub fn validate_result(&self, result: &ObservationResult) -> Result<(), ObservationError> {
        result.validate()?;
        if self != &observation_artifact_entry(result)? {
            return Err(ObservationError::Identity(
                "observation result differs from its manifest entry",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationArtifactManifestWire")]
pub struct ObservationArtifactManifest {
    pub schema_version: u16,
    pub identity: ObservationArtifactIdentity,
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub results: Vec<ObservationArtifactEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationArtifactManifestWire {
    schema_version: u16,
    identity: ObservationArtifactIdentity,
    evaluation_request: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    source_selection: ObservationFieldSelectionIdentity,
    source: ObservationFieldSource,
    topology_digest: Digest,
    plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationArtifactEntry>,
}

impl TryFrom<ObservationArtifactManifestWire> for ObservationArtifactManifest {
    type Error = ObservationError;

    fn try_from(wire: ObservationArtifactManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            evaluation_request: wire.evaluation_request,
            producer: wire.producer,
            source_selection: wire.source_selection,
            source: wire.source,
            topology_digest: wire.topology_digest,
            plan_identity: wire.plan_identity,
            results: wire.results,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl ObservationArtifactManifest {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 2
            || self.results.is_empty()
            || self.topology_digest != self.source.observation_topology_digest()
        {
            return Err(ObservationError::Invalid(
                "observation manifest requires a supported schema and results".into(),
            ));
        }
        let mut result_ids = std::collections::BTreeSet::new();
        let mut observation_ids = std::collections::BTreeSet::new();
        let mut acceleration_evidence = std::collections::BTreeSet::new();
        for entry in &self.results {
            if !result_ids.insert(entry.result) || !observation_ids.insert(&entry.observation_id) {
                return Err(ObservationError::Identity(
                    "observation manifest contains duplicate results",
                ));
            }
            if let Some(summary) = &entry.acceleration {
                summary.validate()?;
                if summary
                    .operations
                    .iter()
                    .any(|operation| !acceleration_evidence.insert(operation.evidence))
                {
                    return Err(ObservationError::Identity(
                        "observation manifest reuses acceleration evidence",
                    ));
                }
            }
        }
        if self.identity != derive_artifact_identity(self)? {
            return Err(ObservationError::Identity(
                "observation manifest identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact-manifest",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let manifest: Self = decode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact-manifest",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationArtifactWire {
    schema_version: u16,
    identity: ObservationArtifactIdentity,
    evaluation_request: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    source_selection: ObservationFieldSelectionIdentity,
    source: ObservationFieldSource,
    topology_digest: Digest,
    plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationResult>,
}

impl TryFrom<ObservationArtifactWire> for ObservationArtifact {
    type Error = ObservationError;

    fn try_from(wire: ObservationArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            evaluation_request: wire.evaluation_request,
            producer: wire.producer,
            source_selection: wire.source_selection,
            source: wire.source,
            topology_digest: wire.topology_digest,
            plan_identity: wire.plan_identity,
            results: wire.results,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

pub struct ObservationArtifactInput {
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub results: Vec<ObservationResult>,
}

impl ObservationArtifact {
    pub fn new(input: ObservationArtifactInput) -> Result<Self, ObservationError> {
        let mut artifact = Self {
            schema_version: 2,
            identity: ObservationArtifactIdentity(Digest::sha256([])),
            evaluation_request: input.evaluation_request,
            producer: input.producer,
            source_selection: input.source_selection,
            source: input.source,
            topology_digest: input.topology_digest,
            plan_identity: input.plan_identity,
            results: input.results,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> ObservationArtifactIdentity {
        self.identity
    }

    pub fn results(&self) -> &[ObservationResult] {
        &self.results
    }

    pub fn manifest(&self) -> Result<ObservationArtifactManifest, ObservationError> {
        let manifest = ObservationArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            evaluation_request: self.evaluation_request,
            producer: self.producer,
            source_selection: self.source_selection,
            source: self.source,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(observation_artifact_entry)
                .collect::<Result<Vec<_>, _>>()?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_manifest_and_results(
        manifest: ObservationArtifactManifest,
        results: Vec<ObservationResult>,
    ) -> Result<Self, ObservationError> {
        manifest.validate()?;
        if manifest.results.len() != results.len() {
            return Err(ObservationError::Identity(
                "observation results differ from their manifest",
            ));
        }
        for (entry, result) in manifest.results.iter().zip(&results) {
            let acceleration = ObservationAccelerationSummary::from_evidence(
                &result.partition_evidence.accelerations,
            )?;
            if entry.result != result.logical_identity
                || entry.partition_evidence != result.partition_evidence.identity
                || entry.acceleration != acceleration
                || entry.observation_id != result.observation_id
                || entry.sample != result.sample
                || entry.quantity != result.quantity
                || entry.location != result.location
                || entry.value_shape != result.value_shape
                || entry.partial_access != result.partial_access
            {
                return Err(ObservationError::Identity(
                    "observation results differ from their manifest",
                ));
            }
        }
        let artifact = Self {
            schema_version: manifest.schema_version,
            identity: manifest.identity,
            evaluation_request: manifest.evaluation_request,
            producer: manifest.producer,
            source_selection: manifest.source_selection,
            source: manifest.source,
            topology_digest: manifest.topology_digest,
            plan_identity: manifest.plan_identity,
            results,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 2
            || self.results.is_empty()
            || self.topology_digest != self.source.observation_topology_digest()
        {
            return Err(ObservationError::Invalid(
                "observation artifact requires a supported schema and results".into(),
            ));
        }
        let mut result_ids = std::collections::BTreeSet::new();
        let mut observation_ids = std::collections::BTreeSet::new();
        for result in &self.results {
            result.validate()?;
            if result.plan_identity != self.plan_identity
                || result.topology_digest != self.topology_digest
                || !result_ids.insert(result.logical_identity)
                || !observation_ids.insert(&result.observation_id)
            {
                return Err(ObservationError::Identity(
                    "observation artifact contains an unrelated or duplicate result",
                ));
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "observation artifact identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let artifact = decode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Ok(artifact)
    }

    fn derive_identity(&self) -> Result<ObservationArtifactIdentity, ObservationError> {
        derive_artifact_identity(&ObservationArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            evaluation_request: self.evaluation_request,
            producer: self.producer,
            source_selection: self.source_selection,
            source: self.source,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(observation_artifact_entry)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

fn observation_artifact_entry(
    result: &ObservationResult,
) -> Result<ObservationArtifactEntry, ObservationError> {
    Ok(ObservationArtifactEntry {
        result: result.logical_identity,
        partition_evidence: result.partition_evidence.identity,
        acceleration: ObservationAccelerationSummary::from_evidence(
            &result.partition_evidence.accelerations,
        )?,
        observation_id: result.observation_id.clone(),
        sample: result.sample,
        quantity: result.quantity,
        location: result.location,
        value_shape: result.value_shape.clone(),
        partial_access: result.partial_access,
    })
}

fn derive_artifact_identity(
    manifest: &ObservationArtifactManifest,
) -> Result<ObservationArtifactIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_ARTIFACT_PREFIX,
        "observation-artifact-identity",
        &(
            manifest.schema_version,
            manifest.evaluation_request,
            manifest.producer,
            manifest.source_selection,
            manifest.source,
            manifest.topology_digest,
            manifest.plan_identity,
            &manifest.results,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationArtifactIdentity(Digest::sha256(bytes)))
}
