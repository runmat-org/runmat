use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const FREQUENCY_PLAN_PREFIX: &[u8] = b"runmat-fea-frequency-response-plan/v1\0";
const FREQUENCY_PLAN_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 100_000, 4096, 8);

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("FEA_FREQUENCY_PLAN_INVALID: {0}")]
pub struct FrequencyPlanError(String);

impl FrequencyPlanError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrequencyResponsePlanIdentity(Digest);

impl FrequencyResponsePlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrequencyResponsePlanEntryIdentity(Digest);

impl FrequencyResponsePlanEntryIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// One independently addressable frequency in a canonical response plan.
///
/// A generic execution request binds this entry alongside the immutable model and solver inputs;
/// the entry identifies schedule membership and order, not authority to execute those inputs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "FrequencyResponsePlanEntryData")]
pub struct FrequencyResponsePlanEntry {
    identity: FrequencyResponsePlanEntryIdentity,
    plan_identity: FrequencyResponsePlanIdentity,
    ordinal: u32,
    frequency_hz: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FrequencyResponsePlanEntryData {
    identity: FrequencyResponsePlanEntryIdentity,
    plan_identity: FrequencyResponsePlanIdentity,
    ordinal: u32,
    frequency_hz: f64,
}

impl TryFrom<FrequencyResponsePlanEntryData> for FrequencyResponsePlanEntry {
    type Error = FrequencyPlanError;

    fn try_from(data: FrequencyResponsePlanEntryData) -> Result<Self, Self::Error> {
        let entry = Self {
            identity: data.identity,
            plan_identity: data.plan_identity,
            ordinal: data.ordinal,
            frequency_hz: data.frequency_hz,
        };
        entry.validate()?;
        Ok(entry)
    }
}

impl FrequencyResponsePlanEntry {
    fn new(
        plan_identity: FrequencyResponsePlanIdentity,
        ordinal: u32,
        frequency_hz: f64,
    ) -> Result<Self, FrequencyPlanError> {
        let mut entry = Self {
            identity: FrequencyResponsePlanEntryIdentity(Digest::sha256([])),
            plan_identity,
            ordinal,
            frequency_hz,
        };
        entry.identity = entry.derive_identity()?;
        entry.validate()?;
        Ok(entry)
    }

    pub const fn identity(self) -> FrequencyResponsePlanEntryIdentity {
        self.identity
    }

    pub const fn plan_identity(self) -> FrequencyResponsePlanIdentity {
        self.plan_identity
    }

    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }

    pub const fn frequency_hz(self) -> f64 {
        self.frequency_hz
    }

    fn validate(&self) -> Result<(), FrequencyPlanError> {
        if !valid_frequency(self.frequency_hz) || self.identity != self.derive_identity()? {
            return Err(FrequencyPlanError::invalid(
                "frequency response plan entry is not canonical",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<FrequencyResponsePlanEntryIdentity, FrequencyPlanError> {
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "frequency-response-plan-entry",
            &(self.plan_identity, self.ordinal, self.frequency_hz),
            FREQUENCY_PLAN_LIMITS,
        )
        .map(Digest::sha256)
        .map(FrequencyResponsePlanEntryIdentity)
        .map_err(|error| FrequencyPlanError::invalid(error.to_string()))
    }
}

/// Canonical independent-frequency work for one harmonic response step.
///
/// Each entry is independently executable. Ordering and the reference entry are fixed here so
/// direct, parallel, and distributed execution can assemble the same response artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "FrequencyResponsePlanData")]
pub struct FrequencyResponsePlan {
    identity: FrequencyResponsePlanIdentity,
    reference_frequency_hz: f64,
    frequencies_hz: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FrequencyResponsePlanData {
    identity: FrequencyResponsePlanIdentity,
    reference_frequency_hz: f64,
    frequencies_hz: Vec<f64>,
}

impl TryFrom<FrequencyResponsePlanData> for FrequencyResponsePlan {
    type Error = FrequencyPlanError;

    fn try_from(data: FrequencyResponsePlanData) -> Result<Self, Self::Error> {
        let plan = Self {
            identity: data.identity,
            reference_frequency_hz: data.reference_frequency_hz,
            frequencies_hz: data.frequencies_hz,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl FrequencyResponsePlan {
    pub fn new(
        reference_frequency_hz: f64,
        requested_frequencies_hz: &[f64],
        observation_frequencies_hz: &[f64],
    ) -> Result<Self, FrequencyPlanError> {
        let mut frequencies_hz = requested_frequencies_hz.to_vec();
        frequencies_hz.extend_from_slice(observation_frequencies_hz);
        frequencies_hz.push(reference_frequency_hz);
        if frequencies_hz.iter().any(|value| !valid_frequency(*value)) {
            return Err(FrequencyPlanError::invalid(
                "response frequencies must be finite, positive, and not negative zero",
            ));
        }
        frequencies_hz.sort_by(f64::total_cmp);
        frequencies_hz.dedup_by(|left, right| left.to_bits() == right.to_bits());
        let mut plan = Self {
            identity: FrequencyResponsePlanIdentity(Digest::sha256([])),
            reference_frequency_hz,
            frequencies_hz,
        };
        plan.identity = plan.derive_identity()?;
        plan.validate()?;
        Ok(plan)
    }

    pub const fn identity(&self) -> FrequencyResponsePlanIdentity {
        self.identity
    }

    pub const fn reference_frequency_hz(&self) -> f64 {
        self.reference_frequency_hz
    }

    pub fn frequencies_hz(&self) -> &[f64] {
        &self.frequencies_hz
    }

    pub fn entries(&self) -> Result<Vec<FrequencyResponsePlanEntry>, FrequencyPlanError> {
        self.validate()?;
        self.frequencies_hz
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, frequency_hz)| {
                let ordinal = u32::try_from(ordinal).map_err(|_| {
                    FrequencyPlanError::invalid(
                        "frequency response plan exceeds the entry ordinal domain",
                    )
                })?;
                FrequencyResponsePlanEntry::new(self.identity, ordinal, frequency_hz)
            })
            .collect()
    }

    pub fn validate_entry(
        &self,
        entry: FrequencyResponsePlanEntry,
    ) -> Result<(), FrequencyPlanError> {
        self.validate()?;
        entry.validate()?;
        let frequency = usize::try_from(entry.ordinal)
            .ok()
            .and_then(|ordinal| self.frequencies_hz.get(ordinal));
        if entry.plan_identity != self.identity
            || frequency.is_none_or(|frequency| frequency.to_bits() != entry.frequency_hz.to_bits())
        {
            return Err(FrequencyPlanError::invalid(
                "frequency response plan entry is absent from its claimed plan",
            ));
        }
        Ok(())
    }

    pub fn reference_index(&self) -> usize {
        self.frequencies_hz
            .binary_search_by(|value| value.total_cmp(&self.reference_frequency_hz))
            .expect("validated frequency plan contains its reference frequency")
    }

    pub fn validate(&self) -> Result<(), FrequencyPlanError> {
        if !valid_frequency(self.reference_frequency_hz)
            || self.frequencies_hz.is_empty()
            || self
                .frequencies_hz
                .iter()
                .any(|value| !valid_frequency(*value))
            || !self
                .frequencies_hz
                .windows(2)
                .all(|pair| pair[0].total_cmp(&pair[1]).is_lt())
            || self
                .frequencies_hz
                .binary_search_by(|value| value.total_cmp(&self.reference_frequency_hz))
                .is_err()
            || self.identity != self.derive_identity()?
        {
            return Err(FrequencyPlanError::invalid(
                "frequency response plan is not canonical",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FrequencyPlanError> {
        self.validate()?;
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "frequency-response-plan-artifact",
            self,
            FREQUENCY_PLAN_LIMITS,
        )
        .map_err(|error| FrequencyPlanError::invalid(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FrequencyPlanError> {
        decode_contract(
            FREQUENCY_PLAN_PREFIX,
            "frequency-response-plan-artifact",
            bytes,
            FREQUENCY_PLAN_LIMITS,
        )
        .map_err(|error| FrequencyPlanError::invalid(error.to_string()))
    }

    fn derive_identity(&self) -> Result<FrequencyResponsePlanIdentity, FrequencyPlanError> {
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "frequency-response-plan",
            &(self.reference_frequency_hz, &self.frequencies_hz),
            FREQUENCY_PLAN_LIMITS,
        )
        .map(Digest::sha256)
        .map(FrequencyResponsePlanIdentity)
        .map_err(|error| FrequencyPlanError::invalid(error.to_string()))
    }
}

fn valid_frequency(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value.to_bits() != (-0.0_f64).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_plan_is_canonical_and_rejects_identity_substitution() {
        let plan = FrequencyResponsePlan::new(60.0, &[80.0, 40.0, 80.0], &[60.0, 40.0]).unwrap();
        assert_eq!(plan.frequencies_hz(), &[40.0, 60.0, 80.0]);
        assert_eq!(plan.reference_index(), 1);
        let entries = plan.entries().unwrap();
        assert_eq!(
            entries
                .iter()
                .copied()
                .map(FrequencyResponsePlanEntry::frequency_hz)
                .collect::<Vec<_>>(),
            plan.frequencies_hz()
        );
        assert!(entries
            .iter()
            .all(|entry| entry.plan_identity() == plan.identity()
                && plan.validate_entry(*entry).is_ok()));
        assert!(entries
            .windows(2)
            .all(|pair| pair[0].identity() != pair[1].identity()));
        assert_eq!(
            FrequencyResponsePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
            plan
        );

        let mut substituted = plan.clone();
        substituted.identity = FrequencyResponsePlanIdentity(Digest::sha256(b"other"));
        assert!(serde_json::from_value::<FrequencyResponsePlan>(
            serde_json::to_value(substituted).unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("not canonical"));
        assert!(FrequencyResponsePlan::new(60.0, &[f64::NAN], &[]).is_err());
        assert!(FrequencyResponsePlan::new(-0.0, &[], &[]).is_err());

        let mut substituted_entry = entries[0];
        substituted_entry.plan_identity =
            FrequencyResponsePlanIdentity(Digest::sha256(b"another-plan"));
        assert!(serde_json::from_value::<FrequencyResponsePlanEntry>(
            serde_json::to_value(substituted_entry).unwrap()
        )
        .is_err());
    }
}
