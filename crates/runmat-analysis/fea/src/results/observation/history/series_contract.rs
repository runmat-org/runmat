use std::num::NonZeroU32;

use runmat_analysis_core::{AnalysisField, ExtremaTiePolicy, ObservationId, SeriesReduction};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{OBSERVATION_LIMITS, OBSERVATION_PREFIX};
use super::{
    super::{ObservationEntity, ObservationError, ResolvedObservationPlan},
    series_reduce::{reduce_values, validate_evidence_against_history},
    ObservationHistory, ObservationHistoryIdentity, ObservationHistorySample,
    ObservationHistorySampleIdentity,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SeriesReductionResultIdentity(Digest);

impl SeriesReductionResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }

    pub(super) fn placeholder() -> Self {
        Self(Digest::sha256([]))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SeriesReductionContributionIdentity(Digest);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesReductionSource {
    pub sample: ObservationHistorySampleIdentity,
    pub coordinate: RecoveredFieldSample,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "SeriesReductionContributionWire")]
pub struct SeriesReductionContribution {
    pub schema_version: u16,
    pub identity: SeriesReductionContributionIdentity,
    pub ordinal: u32,
    pub sources: Vec<SeriesReductionSource>,
    pub values: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeriesReductionContributionWire {
    schema_version: u16,
    identity: SeriesReductionContributionIdentity,
    ordinal: u32,
    sources: Vec<SeriesReductionSource>,
    values: Vec<f64>,
}

impl TryFrom<SeriesReductionContributionWire> for SeriesReductionContribution {
    type Error = ObservationError;

    fn try_from(wire: SeriesReductionContributionWire) -> Result<Self, Self::Error> {
        let contribution = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            ordinal: wire.ordinal,
            sources: wire.sources,
            values: wire.values,
        };
        contribution.validate()?;
        Ok(contribution)
    }
}

impl SeriesReductionContribution {
    pub(super) fn new(
        ordinal: u32,
        samples: &[ObservationHistorySample],
    ) -> Result<Self, ObservationError> {
        let sources = samples
            .iter()
            .map(|sample| SeriesReductionSource {
                sample: sample.identity,
                coordinate: sample.coordinate,
            })
            .collect::<Vec<_>>();
        let values = samples
            .iter()
            .flat_map(|sample| sample.result.field.as_host_f64().unwrap().iter().copied())
            .collect::<Vec<_>>();
        let mut contribution = Self {
            schema_version: 1,
            identity: SeriesReductionContributionIdentity(Digest::sha256([])),
            ordinal,
            sources,
            values,
        };
        contribution.identity = contribution.derive_identity()?;
        contribution.validate()?;
        Ok(contribution)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1
            || self.sources.is_empty()
            || self.values.is_empty()
            || self.values.iter().any(|value| !value.is_finite())
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "series-reduction contribution is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<SeriesReductionContributionIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "series-reduction-contribution-identity",
            &(
                self.schema_version,
                self.ordinal,
                &self.sources,
                &self.values,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(SeriesReductionContributionIdentity(Digest::sha256(bytes)))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesReductionEvidence {
    pub source_history: ObservationHistoryIdentity,
    pub requested_partition_limit: NonZeroU32,
    pub contributions: Vec<SeriesReductionContribution>,
}

impl SeriesReductionEvidence {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.contributions.is_empty()
            || self.contributions.len() > self.requested_partition_limit.get() as usize
        {
            return Err(ObservationError::Invalid(
                "series-reduction evidence has an invalid partition inventory".into(),
            ));
        }
        let mut samples = std::collections::BTreeSet::new();
        let mut lane_count = None;
        for (ordinal, contribution) in self.contributions.iter().enumerate() {
            contribution.validate()?;
            if contribution.ordinal as usize != ordinal
                || contribution.values.len() % contribution.sources.len() != 0
            {
                return Err(ObservationError::Identity(
                    "series-reduction contributions are not canonical",
                ));
            }
            let local_lanes = contribution.values.len() / contribution.sources.len();
            if local_lanes == 0 || lane_count.is_some_and(|expected| expected != local_lanes) {
                return Err(ObservationError::Identity(
                    "series-reduction contributions do not share one value layout",
                ));
            }
            lane_count = Some(local_lanes);
            for source in &contribution.sources {
                if !valid_coordinate(source.coordinate) || !samples.insert(source.sample) {
                    return Err(ObservationError::Identity(
                        "series-reduction source samples are invalid or duplicated",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesExtremumLocation {
    pub lane_index: usize,
    pub value: f64,
    pub source: SeriesReductionSource,
    pub entity: Option<ObservationEntity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SeriesReductionPayload {
    Values {
        field: AnalysisField,
        extrema: Vec<SeriesExtremumLocation>,
    },
    Envelope {
        minimum: AnalysisField,
        maximum: AnalysisField,
        minima: Vec<SeriesExtremumLocation>,
        maxima: Vec<SeriesExtremumLocation>,
    },
}

impl SeriesReductionPayload {
    pub fn fields(&self) -> [Option<&AnalysisField>; 2] {
        match self {
            Self::Values { field, .. } => [Some(field), None],
            Self::Envelope {
                minimum, maximum, ..
            } => [Some(minimum), Some(maximum)],
        }
    }

    pub fn field(&self, field_id: &str) -> Option<&AnalysisField> {
        self.fields()
            .into_iter()
            .flatten()
            .find(|field| field.field_id == field_id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "SeriesReductionResultWire")]
pub struct SeriesReductionResult {
    pub schema_version: u16,
    pub identity: SeriesReductionResultIdentity,
    pub source_history: ObservationHistoryIdentity,
    pub observation_id: ObservationId,
    pub operation: SeriesReduction,
    pub extrema_ties: ExtremaTiePolicy,
    pub payload: SeriesReductionPayload,
    pub evidence: SeriesReductionEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeriesReductionResultWire {
    schema_version: u16,
    identity: SeriesReductionResultIdentity,
    source_history: ObservationHistoryIdentity,
    observation_id: ObservationId,
    operation: SeriesReduction,
    extrema_ties: ExtremaTiePolicy,
    payload: SeriesReductionPayload,
    evidence: SeriesReductionEvidence,
}

impl TryFrom<SeriesReductionResultWire> for SeriesReductionResult {
    type Error = ObservationError;

    fn try_from(wire: SeriesReductionResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_history: wire.source_history,
            observation_id: wire.observation_id,
            operation: wire.operation,
            extrema_ties: wire.extrema_ties,
            payload: wire.payload,
            evidence: wire.evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

impl SeriesReductionResult {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1
            || self.observation_id.0.trim().is_empty()
            || self.operation == SeriesReduction::None
            || self.source_history != self.evidence.source_history
            || self.evidence.contributions.is_empty()
        {
            return Err(ObservationError::Invalid(
                "series-reduction result metadata is inconsistent".into(),
            ));
        }
        self.evidence.validate()?;
        validate_payload(&self.payload, self.operation, self.extrema_ties)?;
        if self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "series-reduction result identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn validate_for(
        &self,
        plan: &ResolvedObservationPlan,
        history: &ObservationHistory,
    ) -> Result<(), ObservationError> {
        self.validate_source(plan, history)?;
        if self.payload != reduce_values(history, self.operation, self.extrema_ties)? {
            return Err(ObservationError::Identity(
                "series-reduction payload differs from its source history",
            ));
        }
        Ok(())
    }

    /// Validate metadata and exact source-sample evidence without repeating the reduction.
    /// Numerical workers use `validate_for` when a recomputed payload comparison is required.
    pub fn validate_source(
        &self,
        plan: &ResolvedObservationPlan,
        history: &ObservationHistory,
    ) -> Result<(), ObservationError> {
        self.validate()?;
        history.validate()?;
        let observation = plan
            .observations
            .iter()
            .find(|candidate| candidate.spec.observation_id == self.observation_id)
            .ok_or_else(|| {
                ObservationError::Invalid(
                    "series-reduction observation is absent from its plan".into(),
                )
            })?;
        let reduction = observation.spec.reduction.as_ref().ok_or_else(|| {
            ObservationError::Invalid("series-reduction policy is absent from its plan".into())
        })?;
        if plan.identity != history.manifest().plan_identity
            || self.source_history != history.manifest().identity
            || self.observation_id != history.manifest().observation_id
            || self.operation != reduction.series
            || self.extrema_ties != reduction.extrema_ties
        {
            return Err(ObservationError::Identity(
                "series-reduction result differs from its plan or source history",
            ));
        }
        validate_evidence_against_history(&self.evidence, history)?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "series-reduction-result",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        Ok(decode_contract(
            OBSERVATION_PREFIX,
            "series-reduction-result",
            bytes,
            OBSERVATION_LIMITS,
        )?)
    }

    pub(super) fn derive_identity(
        &self,
    ) -> Result<SeriesReductionResultIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "series-reduction-result-identity",
            &(
                self.schema_version,
                self.source_history,
                &self.observation_id,
                self.operation,
                self.extrema_ties,
                &self.payload,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(SeriesReductionResultIdentity(Digest::sha256(bytes)))
    }
}

fn validate_payload(
    payload: &SeriesReductionPayload,
    operation: SeriesReduction,
    ties: ExtremaTiePolicy,
) -> Result<(), ObservationError> {
    let validate_field = |field: &AnalysisField| {
        field
            .as_host_f64()
            .filter(|values| {
                !values.is_empty()
                    && values.len() == field.element_count()
                    && values.iter().all(|value| value.is_finite())
            })
            .ok_or_else(|| ObservationError::Invalid("series result field is invalid".into()))
            .map(|_| ())
    };
    match (payload, operation) {
        (
            SeriesReductionPayload::Values { field, extrema },
            SeriesReduction::Minimum | SeriesReduction::Maximum | SeriesReduction::AbsoluteMaximum,
        ) => {
            validate_field(field)?;
            validate_extrema(extrema, field.element_count(), ties)
        }
        (
            SeriesReductionPayload::Values { field, extrema },
            SeriesReduction::RootMeanSquare { .. }
            | SeriesReduction::Mean { .. }
            | SeriesReduction::Variance { .. },
        ) if extrema.is_empty() => validate_field(field),
        (
            SeriesReductionPayload::Envelope {
                minimum,
                maximum,
                minima,
                maxima,
            },
            SeriesReduction::Envelope,
        ) => {
            validate_field(minimum)?;
            validate_field(maximum)?;
            if minimum.shape != maximum.shape {
                return Err(ObservationError::Identity(
                    "series envelope fields do not share a shape",
                ));
            }
            validate_extrema(minima, minimum.element_count(), ties)?;
            validate_extrema(maxima, maximum.element_count(), ties)
        }
        _ => Err(ObservationError::Invalid(
            "series-reduction payload differs from its operation".into(),
        )),
    }
}

fn validate_extrema(
    extrema: &[SeriesExtremumLocation],
    lane_count: usize,
    ties: ExtremaTiePolicy,
) -> Result<(), ObservationError> {
    if extrema.is_empty()
        || extrema.iter().any(|item| {
            item.lane_index >= lane_count
                || !item.value.is_finite()
                || !valid_coordinate(item.source.coordinate)
        })
    {
        return Err(ObservationError::Invalid(
            "series extrema attribution is invalid".into(),
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut counts = vec![0usize; lane_count];
    for item in extrema {
        if !seen.insert((item.lane_index, item.source.sample)) {
            return Err(ObservationError::Identity(
                "series extrema attribution contains duplicate source samples",
            ));
        }
        counts[item.lane_index] += 1;
    }
    if counts
        .iter()
        .any(|count| *count == 0 || (ties == ExtremaTiePolicy::FirstCanonical && *count != 1))
    {
        return Err(ObservationError::Identity(
            "series extrema attribution does not cover every output lane",
        ));
    }
    Ok(())
}

fn valid_coordinate(sample: RecoveredFieldSample) -> bool {
    match sample {
        RecoveredFieldSample::Final | RecoveredFieldSample::ModeIndex(_) => true,
        RecoveredFieldSample::TimeSeconds(value)
        | RecoveredFieldSample::FrequencyHertz(value)
        | RecoveredFieldSample::LoadFactor(value)
        | RecoveredFieldSample::StudyParameterValue(value) => {
            value.is_finite() && value.to_bits() != (-0.0_f64).to_bits()
        }
    }
}
