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

    fn normalize_signed_zero(&mut self) {
        let normalize = |value: &mut u64| {
            if f64::from_bits(*value) == 0.0 {
                *value = 0.0_f64.to_bits();
            }
        };
        match self {
            Self::Real { values } => values.iter_mut().for_each(normalize),
            Self::Complex { real, imaginary } => {
                real.iter_mut().for_each(normalize);
                imaginary.iter_mut().for_each(normalize);
            }
        }
    }

    fn validate(&self, precision: SignalPrecision) -> AnalysisExecutionResult<()> {
        let canonical = |value| canonical_finite_bits(value, precision);
        let valid = match self {
            Self::Real { values } => {
                matches!(precision, SignalPrecision::Real32 | SignalPrecision::Real64)
                    && values.iter().copied().all(canonical)
            }
            Self::Complex { real, imaginary } => {
                matches!(
                    precision,
                    SignalPrecision::Complex64 | SignalPrecision::Complex128
                ) && !real.is_empty()
                    && real.len() == imaginary.len()
                    && real.iter().chain(imaginary).copied().all(canonical)
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanonicalSignalContent<'a> {
    pub quantity: PhysicalQuantityKind,
    pub shape: &'a SignalShape,
    pub topology_entity_count: Option<u64>,
    pub domain: &'a GeneratorDomain,
    pub sample_points: &'a [u64],
    pub precision: SignalPrecision,
    pub values: &'a CanonicalSignalValues,
    pub interpolation: SignalInterpolation,
    pub extrapolation: SignalExtrapolation,
    pub evidence: &'a SignalApproximationEvidence,
}

impl CanonicalSignalContent<'_> {
    pub fn digest(&self) -> AnalysisExecutionResult<Digest> {
        self.validate()?;
        Ok(Digest::sha256(encode_contract(
            REALIZATION_PREFIX,
            CONTENT_DOMAIN,
            self,
            REALIZATION_LIMITS,
        )?))
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.sample_points.len() < 2 || self.evidence.sample_count != self.sample_points.len() {
            return Err(AnalysisExecutionError::Invalid(
                "signal content sample metadata is incomplete or inconsistent".into(),
            ));
        }
        if matches!(
            self.domain,
            GeneratorDomain::StudyParameter { parameter_id } if parameter_id.trim().is_empty()
        ) {
            return Err(AnalysisExecutionError::Invalid(
                "study-parameter signal content requires a parameter id".into(),
            ));
        }
        for value in [
            self.evidence.max_absolute_error,
            self.evidence.max_relative_error,
            self.evidence.derivative_max_absolute_error,
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(AnalysisExecutionError::Invalid(
                    "signal approximation error evidence must be finite and non-negative".into(),
                ));
            }
        }
        let points = self
            .sample_points
            .iter()
            .map(|point| f64::from_bits(*point))
            .collect::<Vec<_>>();
        if points.iter().any(|point| !point.is_finite())
            || points.windows(2).any(|pair| pair[0] >= pair[1])
            || (self.domain == &GeneratorDomain::FrequencyHertz
                && points.first().is_some_and(|point| *point < 0.0))
        {
            return Err(AnalysisExecutionError::Invalid(
                "signal content points must be finite, strictly increasing, and valid for their domain"
                    .into(),
            ));
        }
        self.values.validate(self.precision)?;
        let expected_values = self
            .sample_points
            .len()
            .checked_mul(component_count(self.shape, self.topology_entity_count)?)
            .ok_or(AnalysisExecutionError::Limit("signal content value count"))?;
        if self.values.len() != expected_values {
            return Err(AnalysisExecutionError::Invalid(
                "signal content values do not match its sample and component shape".into(),
            ));
        }
        Ok(())
    }
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
        realization.values.normalize_signed_zero();
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
            || self.provenance.project_revision.trim().is_empty()
            || self.provenance.callable_id.trim().is_empty()
            || self.provenance.frozen_input_digest.trim().is_empty()
        {
            return Err(AnalysisExecutionError::Invalid(
                "signal realization metadata is incomplete or inconsistent".into(),
            ));
        }
        let content = self.content();
        content.validate()?;
        let mut admission_keys = std::collections::BTreeSet::new();
        for admission in &self.input_admissions {
            admission.validate()?;
            if !admission_keys.insert((admission.artifact_id.as_str(), admission.logical_digest)) {
                return Err(AnalysisExecutionError::Invalid(
                    "signal input admission inventory is duplicated".into(),
                ));
            }
        }
        let content_digest = content.digest()?;
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
        self.content().digest()
    }

    fn content(&self) -> CanonicalSignalContent<'_> {
        CanonicalSignalContent {
            quantity: self.quantity,
            shape: &self.shape,
            topology_entity_count: self.topology_entity_count,
            domain: &self.domain,
            sample_points: &self.sample_points,
            precision: self.precision,
            values: &self.values,
            interpolation: self.interpolation,
            extrapolation: self.extrapolation,
            evidence: &self.evidence,
        }
    }
}

fn canonical_finite_bits(bits: u64, precision: SignalPrecision) -> bool {
    let value = f64::from_bits(bits);
    value.is_finite()
        && (value != 0.0 || bits == 0.0_f64.to_bits())
        && match precision {
            SignalPrecision::Real32 | SignalPrecision::Complex64 => {
                f64::from(value as f32).to_bits() == bits
            }
            SignalPrecision::Real64 | SignalPrecision::Complex128 => true,
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
