use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::contracts::{FeaElectromagneticRunResult, FeaRunError};

const FREQUENCY_PLAN_PREFIX: &[u8] = b"runmat-fea-electromagnetic-frequency-plan/v1\0";
const FREQUENCY_PLAN_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 100_000, 4096, 8);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ElectromagneticFrequencyPlanIdentity(Digest);

impl ElectromagneticFrequencyPlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ElectromagneticFrequencyPlanEntryIdentity(Digest);

impl ElectromagneticFrequencyPlanEntryIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// One independently addressable frequency in a canonical response plan.
///
/// A generic execution request binds this entry alongside the immutable model and solver inputs;
/// the entry identifies schedule membership and order, not authority to execute those inputs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "ElectromagneticFrequencyPlanEntryData"
)]
pub struct ElectromagneticFrequencyPlanEntry {
    identity: ElectromagneticFrequencyPlanEntryIdentity,
    plan_identity: ElectromagneticFrequencyPlanIdentity,
    ordinal: u32,
    frequency_hz: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ElectromagneticFrequencyPlanEntryData {
    identity: ElectromagneticFrequencyPlanEntryIdentity,
    plan_identity: ElectromagneticFrequencyPlanIdentity,
    ordinal: u32,
    frequency_hz: f64,
}

impl TryFrom<ElectromagneticFrequencyPlanEntryData> for ElectromagneticFrequencyPlanEntry {
    type Error = FeaRunError;

    fn try_from(data: ElectromagneticFrequencyPlanEntryData) -> Result<Self, Self::Error> {
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

impl ElectromagneticFrequencyPlanEntry {
    fn new(
        plan_identity: ElectromagneticFrequencyPlanIdentity,
        ordinal: u32,
        frequency_hz: f64,
    ) -> Result<Self, FeaRunError> {
        let mut entry = Self {
            identity: ElectromagneticFrequencyPlanEntryIdentity(Digest::sha256([])),
            plan_identity,
            ordinal,
            frequency_hz,
        };
        entry.identity = entry.derive_identity()?;
        entry.validate()?;
        Ok(entry)
    }

    pub const fn identity(self) -> ElectromagneticFrequencyPlanEntryIdentity {
        self.identity
    }

    pub const fn plan_identity(self) -> ElectromagneticFrequencyPlanIdentity {
        self.plan_identity
    }

    pub const fn ordinal(self) -> u32 {
        self.ordinal
    }

    pub const fn frequency_hz(self) -> f64 {
        self.frequency_hz
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        if !valid_frequency(self.frequency_hz) || self.identity != self.derive_identity()? {
            return Err(FeaRunError::InvalidFrequencyPlan(
                "electromagnetic frequency plan entry is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ElectromagneticFrequencyPlanEntryIdentity, FeaRunError> {
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "electromagnetic-frequency-plan-entry",
            &(self.plan_identity, self.ordinal, self.frequency_hz),
            FREQUENCY_PLAN_LIMITS,
        )
        .map(Digest::sha256)
        .map(ElectromagneticFrequencyPlanEntryIdentity)
        .map_err(|error| FeaRunError::InvalidFrequencyPlan(error.to_string()))
    }
}

/// Canonical independent-frequency work for one harmonic electromagnetic step.
///
/// Each entry is independently executable. Ordering and the reference entry are fixed here so
/// direct, parallel, and distributed execution can assemble the same response artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ElectromagneticFrequencyPlanData")]
pub struct ElectromagneticFrequencyPlan {
    identity: ElectromagneticFrequencyPlanIdentity,
    reference_frequency_hz: f64,
    frequencies_hz: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ElectromagneticFrequencyPlanData {
    identity: ElectromagneticFrequencyPlanIdentity,
    reference_frequency_hz: f64,
    frequencies_hz: Vec<f64>,
}

impl TryFrom<ElectromagneticFrequencyPlanData> for ElectromagneticFrequencyPlan {
    type Error = FeaRunError;

    fn try_from(data: ElectromagneticFrequencyPlanData) -> Result<Self, Self::Error> {
        let plan = Self {
            identity: data.identity,
            reference_frequency_hz: data.reference_frequency_hz,
            frequencies_hz: data.frequencies_hz,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl ElectromagneticFrequencyPlan {
    pub fn new(
        reference_frequency_hz: f64,
        requested_frequencies_hz: &[f64],
        observation_frequencies_hz: &[f64],
    ) -> Result<Self, FeaRunError> {
        let mut frequencies_hz = requested_frequencies_hz.to_vec();
        frequencies_hz.extend_from_slice(observation_frequencies_hz);
        frequencies_hz.push(reference_frequency_hz);
        if frequencies_hz.iter().any(|value| !valid_frequency(*value)) {
            return Err(FeaRunError::InvalidFrequencyPlan(
                "electromagnetic frequencies must be finite, positive, and not negative zero"
                    .into(),
            ));
        }
        frequencies_hz.sort_by(f64::total_cmp);
        frequencies_hz.dedup_by(|left, right| left.to_bits() == right.to_bits());
        let mut plan = Self {
            identity: ElectromagneticFrequencyPlanIdentity(Digest::sha256([])),
            reference_frequency_hz,
            frequencies_hz,
        };
        plan.identity = plan.derive_identity()?;
        plan.validate()?;
        Ok(plan)
    }

    pub const fn identity(&self) -> ElectromagneticFrequencyPlanIdentity {
        self.identity
    }

    pub const fn reference_frequency_hz(&self) -> f64 {
        self.reference_frequency_hz
    }

    pub fn frequencies_hz(&self) -> &[f64] {
        &self.frequencies_hz
    }

    pub fn entries(&self) -> Result<Vec<ElectromagneticFrequencyPlanEntry>, FeaRunError> {
        self.validate()?;
        self.frequencies_hz
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, frequency_hz)| {
                let ordinal = u32::try_from(ordinal).map_err(|_| {
                    FeaRunError::InvalidFrequencyPlan(
                        "electromagnetic frequency plan exceeds the entry ordinal domain".into(),
                    )
                })?;
                ElectromagneticFrequencyPlanEntry::new(self.identity, ordinal, frequency_hz)
            })
            .collect()
    }

    pub fn validate_entry(
        &self,
        entry: ElectromagneticFrequencyPlanEntry,
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        entry.validate()?;
        let frequency = usize::try_from(entry.ordinal)
            .ok()
            .and_then(|ordinal| self.frequencies_hz.get(ordinal));
        if entry.plan_identity != self.identity
            || frequency.is_none_or(|frequency| frequency.to_bits() != entry.frequency_hz.to_bits())
        {
            return Err(FeaRunError::InvalidFrequencyPlan(
                "electromagnetic frequency plan entry is absent from its claimed plan".into(),
            ));
        }
        Ok(())
    }

    pub fn reference_index(&self) -> usize {
        self.frequencies_hz
            .binary_search_by(|value| value.total_cmp(&self.reference_frequency_hz))
            .expect("validated frequency plan contains its reference frequency")
    }

    pub fn validate(&self) -> Result<(), FeaRunError> {
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
            return Err(FeaRunError::InvalidFrequencyPlan(
                "electromagnetic frequency plan is not canonical".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "electromagnetic-frequency-plan-artifact",
            self,
            FREQUENCY_PLAN_LIMITS,
        )
        .map_err(|error| FeaRunError::InvalidFrequencyPlan(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        decode_contract(
            FREQUENCY_PLAN_PREFIX,
            "electromagnetic-frequency-plan-artifact",
            bytes,
            FREQUENCY_PLAN_LIMITS,
        )
        .map_err(|error| FeaRunError::InvalidFrequencyPlan(error.to_string()))
    }

    fn derive_identity(&self) -> Result<ElectromagneticFrequencyPlanIdentity, FeaRunError> {
        encode_contract(
            FREQUENCY_PLAN_PREFIX,
            "electromagnetic-frequency-plan",
            &(self.reference_frequency_hz, &self.frequencies_hz),
            FREQUENCY_PLAN_LIMITS,
        )
        .map(Digest::sha256)
        .map(ElectromagneticFrequencyPlanIdentity)
        .map_err(|error| FeaRunError::InvalidFrequencyPlan(error.to_string()))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElectromagneticFrequencySummary {
    pub peak_flux_density: Vec<f64>,
    pub solve_quality: Vec<f64>,
    pub resonance_peak_frequency_hz: Option<f64>,
    pub resonance_peak_flux_density: Option<f64>,
    pub resonance_bandwidth_hz: Option<f64>,
    pub resonance_quality_factor: Option<f64>,
    pub resonance_flux_gain: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeaElectromagneticFrequencyResponse {
    pub(super) plan: ElectromagneticFrequencyPlan,
    pub(super) runs: Vec<FeaElectromagneticRunResult>,
    pub(super) summary: ElectromagneticFrequencySummary,
}

impl FeaElectromagneticFrequencyResponse {
    pub const fn plan(&self) -> &ElectromagneticFrequencyPlan {
        &self.plan
    }

    pub fn runs(&self) -> &[FeaElectromagneticRunResult] {
        &self.runs
    }

    pub const fn summary(&self) -> &ElectromagneticFrequencySummary {
        &self.summary
    }

    pub fn primary(&self) -> &FeaElectromagneticRunResult {
        &self.runs[self.plan.reference_index()]
    }

    pub fn into_primary(self) -> FeaElectromagneticRunResult {
        self.runs
            .into_iter()
            .nth(self.plan.reference_index())
            .expect("validated frequency response contains its reference run")
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
        let plan =
            ElectromagneticFrequencyPlan::new(60.0, &[80.0, 40.0, 80.0], &[60.0, 40.0]).unwrap();
        assert_eq!(plan.frequencies_hz(), &[40.0, 60.0, 80.0]);
        assert_eq!(plan.reference_index(), 1);
        let entries = plan.entries().unwrap();
        assert_eq!(
            entries
                .iter()
                .copied()
                .map(ElectromagneticFrequencyPlanEntry::frequency_hz)
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
            ElectromagneticFrequencyPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap())
                .unwrap(),
            plan
        );

        let mut substituted = plan.clone();
        substituted.identity = ElectromagneticFrequencyPlanIdentity(Digest::sha256(b"other"));
        assert!(serde_json::from_value::<ElectromagneticFrequencyPlan>(
            serde_json::to_value(substituted).unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("not canonical"));
        assert!(ElectromagneticFrequencyPlan::new(60.0, &[f64::NAN], &[]).is_err());
        assert!(ElectromagneticFrequencyPlan::new(-0.0, &[], &[]).is_err());

        let mut substituted_entry = entries[0];
        substituted_entry.plan_identity =
            ElectromagneticFrequencyPlanIdentity(Digest::sha256(b"another-plan"));
        assert!(serde_json::from_value::<ElectromagneticFrequencyPlanEntry>(
            serde_json::to_value(substituted_entry).unwrap()
        )
        .is_err());
    }
}
