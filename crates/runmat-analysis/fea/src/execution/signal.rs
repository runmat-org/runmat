use runmat_analysis_core::{
    GeneratorDomain, GeneratorProvenance, PhysicalQuantityKind, SignalApproximationEvidence,
    SignalExtrapolation, SignalId, SignalInterpolation, SignalPrecision, SignalShape,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValueAdmissionReceipt, EXTERNALIZED_VALUE_SCHEMA};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, SignalMaterializationIdentity,
    SignalRealizationIdentity,
};

const REALIZATION_PREFIX: &[u8] = b"runmat-analysis-signal-realization/v1\0";
const REALIZATION_DOMAIN: &str = "signal-realization";
const CONTENT_DOMAIN: &str = "signal-content";
const REALIZATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalInputAdmission {
    pub artifact_id: String,
    pub logical_digest: Digest,
    pub value_schema: String,
    pub receipt: ValueAdmissionReceipt,
}

impl SignalInputAdmission {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.receipt
            .validate_portable_immutable_real64()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.artifact_id.trim().is_empty()
            || self.logical_digest.bytes() == &[0; 32]
            || self.value_schema != EXTERNALIZED_VALUE_SCHEMA
        {
            return Err(AnalysisExecutionError::Invalid(
                "signal input admission is incomplete or non-portable".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSignalValues {
    Real { values: Vec<u64> },
    Complex { real: Vec<u64>, imaginary: Vec<u64> },
}

impl CanonicalSignalValues {
    pub fn real(values: impl IntoIterator<Item = f64>) -> Self {
        Self::Real {
            values: values.into_iter().map(f64::to_bits).collect(),
        }
    }

    pub fn complex(values: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let (real, imaginary) = values
            .into_iter()
            .map(|(real, imaginary)| (real.to_bits(), imaginary.to_bits()))
            .unzip();
        Self::Complex { real, imaginary }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Real { values } => values.len(),
            Self::Complex { real, .. } => real.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn validate(&self, precision: SignalPrecision) -> AnalysisExecutionResult<()> {
        let valid = match self {
            Self::Real { values } => {
                matches!(precision, SignalPrecision::Real32 | SignalPrecision::Real64)
                    && values
                        .iter()
                        .all(|value| f64::from_bits(*value).is_finite())
            }
            Self::Complex { real, imaginary } => {
                matches!(
                    precision,
                    SignalPrecision::Complex64 | SignalPrecision::Complex128
                ) && !real.is_empty()
                    && real.len() == imaginary.len()
                    && real
                        .iter()
                        .chain(imaginary)
                        .all(|value| f64::from_bits(*value).is_finite())
            }
        };
        if !valid {
            return Err(AnalysisExecutionError::Invalid(
                "signal values are non-finite, inconsistent, or differ from their precision".into(),
            ));
        }
        Ok(())
    }
}

/// Canonical, provider-neutral output of a bounded signal-generator task.
///
/// Values are sample-major. Each sample contains the complete declared component shape. A
/// topology field additionally names its realized entity count; no solver callback survives this
/// boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSignalRealization {
    pub schema_version: u16,
    pub materialization: SignalMaterializationIdentity,
    pub identity: SignalRealizationIdentity,
    pub content_digest: Digest,
    pub signal_id: SignalId,
    pub quantity: PhysicalQuantityKind,
    pub shape: SignalShape,
    pub topology_entity_count: Option<u64>,
    pub domain: GeneratorDomain,
    pub sample_points: Vec<u64>,
    pub precision: SignalPrecision,
    pub values: CanonicalSignalValues,
    pub interpolation: SignalInterpolation,
    pub extrapolation: SignalExtrapolation,
    pub evidence: SignalApproximationEvidence,
    pub provenance: GeneratorProvenance,
    pub input_admissions: Vec<SignalInputAdmission>,
}

pub struct CanonicalSignalRealizationInput {
    pub materialization: SignalMaterializationIdentity,
    pub signal_id: SignalId,
    pub quantity: PhysicalQuantityKind,
    pub shape: SignalShape,
    pub topology_entity_count: Option<u64>,
    pub domain: GeneratorDomain,
    pub sample_points: Vec<f64>,
    pub precision: SignalPrecision,
    pub values: CanonicalSignalValues,
    pub interpolation: SignalInterpolation,
    pub extrapolation: SignalExtrapolation,
    pub evidence: SignalApproximationEvidence,
    pub provenance: GeneratorProvenance,
    pub input_admissions: Vec<SignalInputAdmission>,
}

impl CanonicalSignalRealization {
    pub fn new(input: CanonicalSignalRealizationInput) -> AnalysisExecutionResult<Self> {
        let mut realization = Self {
            schema_version: 1,
            materialization: input.materialization,
            identity: SignalRealizationIdentity::new(input.materialization, Digest::sha256([]))?,
            content_digest: Digest::sha256([]),
            signal_id: input.signal_id,
            quantity: input.quantity,
            shape: input.shape,
            topology_entity_count: input.topology_entity_count,
            domain: input.domain,
            sample_points: input.sample_points.into_iter().map(f64::to_bits).collect(),
            precision: input.precision,
            values: input.values,
            interpolation: input.interpolation,
            extrapolation: input.extrapolation,
            evidence: input.evidence,
            provenance: input.provenance,
            input_admissions: input.input_admissions,
        };
        realization.content_digest = realization.derive_content_digest()?;
        realization.identity = SignalRealizationIdentity::new(
            realization.materialization,
            realization.content_digest,
        )?;
        realization.validate()?;
        Ok(realization)
    }

    pub fn sample_points(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.sample_points
            .iter()
            .map(|point| f64::from_bits(*point))
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.signal_id.0.trim().is_empty()
            || self.sample_points.len() < 2
            || self.evidence.sample_count != self.sample_points.len()
            || self.provenance.project_revision.trim().is_empty()
            || self.provenance.callable_id.trim().is_empty()
            || self.provenance.frozen_input_digest.trim().is_empty()
        {
            return Err(AnalysisExecutionError::Invalid(
                "signal realization metadata is incomplete or inconsistent".into(),
            ));
        }
        let points = self.sample_points().collect::<Vec<_>>();
        if points.iter().any(|point| !point.is_finite())
            || points.windows(2).any(|pair| pair[0] >= pair[1])
            || (self.domain == GeneratorDomain::FrequencyHertz
                && points.first().is_some_and(|point| *point < 0.0))
        {
            return Err(AnalysisExecutionError::Invalid(
                "signal realization points must be finite, strictly increasing, and valid for their domain"
                    .into(),
            ));
        }
        self.values.validate(self.precision)?;
        let mut admission_keys = std::collections::BTreeSet::new();
        for admission in &self.input_admissions {
            admission.validate()?;
            if !admission_keys.insert((admission.artifact_id.as_str(), admission.logical_digest)) {
                return Err(AnalysisExecutionError::Invalid(
                    "signal input admission inventory is duplicated".into(),
                ));
            }
        }
        let component_count = component_count(&self.shape, self.topology_entity_count)?;
        let expected_values = self
            .sample_points
            .len()
            .checked_mul(component_count)
            .ok_or(AnalysisExecutionError::Limit(
                "signal realization value count",
            ))?;
        if self.values.len() != expected_values {
            return Err(AnalysisExecutionError::Invalid(
                "signal realization values do not match its sample and component shape".into(),
            ));
        }
        let content_digest = self.derive_content_digest()?;
        if self.content_digest != content_digest
            || self.identity
                != SignalRealizationIdentity::new(self.materialization, content_digest)?
        {
            return Err(AnalysisExecutionError::Identity(
                "signal realization identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            REALIZATION_PREFIX,
            REALIZATION_DOMAIN,
            self,
            REALIZATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(
            REALIZATION_PREFIX,
            REALIZATION_DOMAIN,
            bytes,
            REALIZATION_LIMITS,
        )?;
        value.validate()?;
        Ok(value)
    }

    fn derive_content_digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            REALIZATION_PREFIX,
            CONTENT_DOMAIN,
            &(
                self.quantity,
                &self.shape,
                self.topology_entity_count,
                &self.domain,
                &self.sample_points,
                self.precision,
                &self.values,
                self.interpolation,
                self.extrapolation,
                &self.evidence,
            ),
            REALIZATION_LIMITS,
        )?))
    }
}

fn component_count(
    shape: &SignalShape,
    topology_entity_count: Option<u64>,
) -> AnalysisExecutionResult<usize> {
    let entities = topology_entity_count
        .map(usize::try_from)
        .transpose()
        .map_err(|_| AnalysisExecutionError::Limit("topology entity count"))?;
    shape.realized_component_count(entities).ok_or_else(|| {
        AnalysisExecutionError::Invalid(
            "signal realization shape and topology entity count are inconsistent or overflowed"
                .into(),
        )
    })
}
