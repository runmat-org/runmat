use std::collections::BTreeSet;

use runmat_analysis_core::{ObservationId, ObservationRequest};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::planning::resource::ResourceUsageEvidence;
use crate::results::field::{RecoveredFieldArtifactManifest, RecoveredFieldSample};
use crate::results::observation::{
    ObservationArtifactManifest, ObservationHistoryCheckpointManifest,
    ObservationHistoryPublication,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoverySource,
    StructuralFrequencyEntryResultManifest, StructuralFrequencyNumericalReceipt,
    StructuralFrequencyPublicationIdentity, StructuralFrequencyResultEntry,
    StructuralFrequencySolverResult, StructuralFrequencySolverResultManifest,
    StructuralFrequencyWorkload, StructuralFrequencyWorkloadIdentity,
};

const FREQUENCY_PUBLICATION_PREFIX: &[u8] = b"runmat-structural-frequency-publication/v1\0";
pub(super) const FREQUENCY_PUBLICATION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Semantic publication for one frequency sample. Physical object locations are deliberately
/// absent: the generic execution-artifact catalog owns transport and retention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyPublicationSample {
    pub result: StructuralFrequencyResultEntry,
    pub source_entry: StructuralFrequencyEntryResultManifest,
    pub coordinate: RecoveredFieldSample,
    pub fields: RecoveredFieldArtifactManifest,
    pub observations: Option<ObservationArtifactManifest>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyPublicationHistory {
    pub observation_id: ObservationId,
    pub checkpoint: ObservationHistoryCheckpointManifest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalization: Option<ObservationHistoryPublication>,
}

pub struct StructuralFrequencyPublicationInput<'a> {
    pub workload: &'a StructuralFrequencyWorkload,
    pub result: &'a StructuralFrequencySolverResult,
    pub receipt: StructuralFrequencyNumericalReceipt,
    pub samples: Vec<StructuralFrequencyPublicationSample>,
    pub histories: Vec<StructuralFrequencyPublicationHistory>,
    pub resource_usage: ResourceUsageEvidence,
}

/// Path-free publication closure for a structural harmonic-response result.
///
/// This manifest is small enough to admit and query without loading displacement vectors or
/// field partitions. It binds the physical result, numerical receipt, recovered-field catalogs,
/// observation catalogs, bounded histories, and measured resource evidence. The numerical
/// identities are placement-neutral; this publication identity also binds truthful attempt and
/// worker evidence. `ValueRef`s and storage paths remain in the generic published-object catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralFrequencyPublicationManifestWire"
)]
pub struct StructuralFrequencyPublicationManifest {
    schema_version: u16,
    identity: StructuralFrequencyPublicationIdentity,
    workload: StructuralFrequencyWorkloadIdentity,
    result: StructuralFrequencySolverResultManifest,
    receipt: StructuralFrequencyNumericalReceipt,
    samples: Vec<StructuralFrequencyPublicationSample>,
    histories: Vec<StructuralFrequencyPublicationHistory>,
    resource_usage: ResourceUsageEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyPublicationManifestWire {
    schema_version: u16,
    identity: StructuralFrequencyPublicationIdentity,
    workload: StructuralFrequencyWorkloadIdentity,
    result: StructuralFrequencySolverResultManifest,
    receipt: StructuralFrequencyNumericalReceipt,
    samples: Vec<StructuralFrequencyPublicationSample>,
    histories: Vec<StructuralFrequencyPublicationHistory>,
    resource_usage: ResourceUsageEvidence,
}

impl TryFrom<StructuralFrequencyPublicationManifestWire>
    for StructuralFrequencyPublicationManifest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyPublicationManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            workload: wire.workload,
            result: wire.result,
            receipt: wire.receipt,
            samples: wire.samples,
            histories: wire.histories,
            resource_usage: wire.resource_usage,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl StructuralFrequencyPublicationManifest {
    pub fn new(input: StructuralFrequencyPublicationInput<'_>) -> AnalysisExecutionResult<Self> {
        input.workload.validate()?;
        input.result.validate()?;
        input.receipt.validate_for_result(input.result)?;
        let mut manifest = Self {
            schema_version: 1,
            identity: StructuralFrequencyPublicationIdentity::derive(
                "structural-frequency-publication-placeholder",
                &1_u8,
            )?,
            workload: input.workload.identity(),
            result: input.result.manifest().clone(),
            receipt: input.receipt,
            samples: input.samples,
            histories: input.histories,
            resource_usage: input.resource_usage,
        };
        manifest.identity = manifest.derive_identity()?;
        manifest.validate_for_workload(input.workload)?;
        Ok(manifest)
    }

    pub const fn identity(&self) -> StructuralFrequencyPublicationIdentity {
        self.identity
    }

    pub const fn workload(&self) -> StructuralFrequencyWorkloadIdentity {
        self.workload
    }

    pub fn result(&self) -> &StructuralFrequencySolverResultManifest {
        &self.result
    }

    pub fn receipt(&self) -> &StructuralFrequencyNumericalReceipt {
        &self.receipt
    }

    pub fn samples(&self) -> &[StructuralFrequencyPublicationSample] {
        &self.samples
    }

    pub fn histories(&self) -> &[StructuralFrequencyPublicationHistory] {
        &self.histories
    }

    pub fn resource_usage(&self) -> &ResourceUsageEvidence {
        &self.resource_usage
    }

    pub fn validate_for_workload(
        &self,
        workload: &StructuralFrequencyWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        if self.workload != workload.identity() || self.result.scope().plan() != workload.plan() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency publication names another workload or frequency plan",
            ));
        }
        let step = workload.procedure_step()?;
        let authored_histories = step
            .observations
            .iter()
            .filter_map(|request| match request {
                ObservationRequest::History(spec) => Some(spec),
                _ => None,
            })
            .collect::<Vec<_>>();
        if authored_histories.len() != self.histories.len() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency publication does not cover every authored history",
            ));
        }
        for (authored, published) in authored_histories.into_iter().zip(&self.histories) {
            let stream = &published.checkpoint.stream;
            if published.observation_id != authored.observation_id
                || stream.observation_id != authored.observation_id
                || stream.independent_domain != step.independent_domain
                || stream.quantity != authored.quantity
                || stream.location != authored.output_location
                || stream.value_shape != authored.result_value_shape()
                || stream.complex_representation != authored.complex_representation
                || stream.sampling != authored.sampling
                || stream.retention != authored.storage.retention
                || stream.partial_access != authored.storage.partial_access
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency history differs from its authored observation",
                ));
            }
            let observed = self
                .samples
                .iter()
                .filter_map(|sample| {
                    let observations = sample.observations.as_ref()?;
                    observations
                        .results
                        .iter()
                        .find(|result| result.observation_id == authored.observation_id)
                        .map(|result| (observations, result))
                })
                .collect::<Vec<_>>();
            if published.checkpoint.observed_sample_count
                != u64::try_from(observed.len()).map_err(|_| {
                    AnalysisExecutionError::Invalid(
                        "structural frequency history sample count overflows".into(),
                    )
                })?
                || observed.iter().any(|(observations, _)| {
                    observations.plan_identity != stream.plan_identity
                        || observations.topology_digest != stream.topology_digest
                })
                || published.checkpoint.samples.iter().any(|retained| {
                    !observed.iter().any(|(_, observation)| {
                        observation.sample == retained.coordinate
                            && observation.result == retained.result
                    })
                })
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency history is not the exact retained chain of its published observations",
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FREQUENCY_PUBLICATION_PREFIX,
            "structural-frequency-publication-manifest",
            self,
            FREQUENCY_PUBLICATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_PUBLICATION_PREFIX,
            "structural-frequency-publication-manifest",
            bytes,
            FREQUENCY_PUBLICATION_LIMITS,
        )?)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.samples.len() != self.result.samples().len() {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency publication has an invalid schema or sample inventory".into(),
            ));
        }
        self.result.validate()?;
        self.receipt.validate_for_manifest(&self.result)?;
        self.resource_usage
            .validate_identity()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.resource_usage.workload_digest != self.workload.digest()
            || self.resource_usage.result_digest != self.result.identity().digest()
            || !self.resource_usage.overruns.is_empty()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency publication resource evidence names another workload or result",
            ));
        }
        let mut source_entries = BTreeSet::new();
        for ((published, expected), receipt) in self
            .samples
            .iter()
            .zip(self.result.samples())
            .zip(self.receipt.samples())
        {
            let coordinate =
                RecoveredFieldSample::FrequencyHertz(expected.plan_entry().frequency_hz());
            published.source_entry.validate()?;
            published
                .fields
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if published.result != *expected
                || published.coordinate != coordinate
                || !source_entries.insert(published.source_entry.identity())
                || published.source_entry.workload() != self.workload
                || published.source_entry.scope() != self.result.scope().identity()
                || published.source_entry.entry() != expected.plan_entry()
                || published.source_entry.sample() != expected.sample()
                || published.source_entry.solution_space() != self.receipt.solution_space()
                || published.source_entry.evidence() != receipt
                || published.fields.source
                    != FieldRecoverySource::StructuralFrequencyEntry(
                        published.source_entry.identity(),
                    )
                || published
                    .fields
                    .fields
                    .iter()
                    .any(|field| field.sample != coordinate)
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency published field sample differs from its physical result",
                ));
            }
            if let Some(observations) = &published.observations {
                observations
                    .validate()
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if !matches!(observations.source,
                    crate::results::observation::ObservationFieldSource::StructuralRecovery {
                        recovery_request, artifact, numerical_source, ..
                    } if artifact == published.fields.identity
                        && recovery_request == published.fields.recovery_request
                        && numerical_source == FieldRecoverySource::StructuralFrequencyEntry(published.source_entry.identity()))
                    || observations
                        .results
                        .iter()
                        .any(|result| result.sample != coordinate)
                {
                    return Err(AnalysisExecutionError::Identity(
                        "structural frequency observations differ from their field sample",
                    ));
                }
            }
        }
        let mut history_ids = BTreeSet::new();
        for history in &self.histories {
            history
                .checkpoint
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if let Some(finalization) = &history.finalization {
                finalization.validate()?;
                if finalization.checkpoint != history.checkpoint
                    || finalization.request.observation_id != history.observation_id
                {
                    return Err(AnalysisExecutionError::Identity(
                        "frequency history finalization differs from its terminal checkpoint",
                    ));
                }
            }
            if history.observation_id != history.checkpoint.stream.observation_id
                || !history_ids.insert(&history.observation_id)
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural frequency publication contains invalid or duplicate histories",
                ));
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency publication identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralFrequencyPublicationIdentity> {
        StructuralFrequencyPublicationIdentity::derive_publication(
            &(
                self.schema_version,
                self.workload,
                &self.result,
                &self.receipt,
                &self.samples,
                &self.histories,
                &self.resource_usage,
            ),
            FREQUENCY_PUBLICATION_LIMITS,
        )
    }
}
