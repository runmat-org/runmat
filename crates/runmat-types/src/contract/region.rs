use crate::{
    AliasFact, CapabilityRequirement, CapabilitySet, DeoptimizationPointId, EffectSet,
    ProgramPointId, ProgramSourceId, ProgramSpan, RegionGuardId, RegionId, RegionValueId,
    ResidencyFact, SchemaValidationError, ShapeFact, ValueFact, ValueKindFact,
};
use serde::{Deserialize, Serialize};

pub const REGION_CONTRACT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionContract {
    pub schema_version: u16,
    pub id: RegionId,
    pub source: ProgramSourceId,
    pub span: ProgramSpan,
    pub entry: ProgramPointId,
    pub exits: Vec<ProgramPointId>,
    pub live_in: Vec<RegionValueId>,
    pub live_out: Vec<RegionValueId>,
    pub value_facts: Vec<RegionValueFact>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
    pub guards: Vec<RegionGuardContract>,
    pub provenance: RegionProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionValueFact {
    pub value: RegionValueId,
    pub fact: ValueFact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionGuardContract {
    pub id: RegionGuardId,
    pub condition: RegionGuardCondition,
    pub deopt: DeoptimizationPointId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum RegionGuardCondition {
    ValueFact {
        value: RegionValueId,
        expected: ValueFact,
    },
    Shape {
        value: RegionValueId,
        expected: ShapeFact,
    },
    Class {
        value: RegionValueId,
        expected: ValueKindFact,
    },
    Residency {
        value: RegionValueId,
        expected: ResidencyFact,
    },
    Alias {
        value: RegionValueId,
        expected: AliasFact,
    },
    RuntimeState {
        identity: String,
        revision: String,
    },
    Capability {
        requirement: CapabilityRequirement,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum RegionProvenance {
    Source,
    Inferred,
    Profiled { profile_digest: String },
    Recovered { cache_key: String },
}

impl RegionContract {
    pub fn validate(&self) -> Result<(), SchemaValidationError> {
        if self.schema_version != REGION_CONTRACT_SCHEMA_VERSION {
            return Err(SchemaValidationError::new(
                "region.schema_version",
                format!(
                    "unsupported version {}; expected {}",
                    self.schema_version, REGION_CONTRACT_SCHEMA_VERSION
                ),
            ));
        }
        if self.span.start > self.span.end {
            return Err(SchemaValidationError::new(
                "region.span",
                "start must not exceed end",
            ));
        }
        ensure_sorted_unique("region.exits", &self.exits)?;
        ensure_sorted_unique("region.live_in", &self.live_in)?;
        ensure_sorted_unique("region.live_out", &self.live_out)?;
        ensure_sorted_unique_by("region.value_facts", &self.value_facts, |fact| fact.value)?;
        ensure_sorted_unique_by("region.guards", &self.guards, |guard| guard.id)?;
        if self.entry.function != self.id.function
            || self
                .exits
                .iter()
                .any(|point| point.function != self.id.function)
            || self
                .live_in
                .iter()
                .chain(&self.live_out)
                .any(|value| value.function != self.id.function)
            || self
                .value_facts
                .iter()
                .any(|fact| fact.value.function != self.id.function)
        {
            return Err(SchemaValidationError::new(
                "region.identity",
                "all program points and values must belong to the region function",
            ));
        }
        if self.guards.iter().any(|guard| {
            guard.id.region != self.id
                || guard.deopt.function != self.id.function
                || guard
                    .condition
                    .value()
                    .is_some_and(|value| value.function != self.id.function)
        }) {
            return Err(SchemaValidationError::new(
                "region.guards",
                "guard and deoptimization identities must belong to the region function",
            ));
        }
        for guard in &self.guards {
            if let RegionGuardCondition::RuntimeState { identity, revision } = &guard.condition {
                super::schema::validate_token(
                    "region.guards.runtime_state.identity",
                    identity,
                    128,
                )?;
                super::schema::validate_token(
                    "region.guards.runtime_state.revision",
                    revision,
                    256,
                )?;
            }
        }
        match &self.provenance {
            RegionProvenance::Profiled { profile_digest } => super::schema::validate_token(
                "region.provenance.profile_digest",
                profile_digest,
                128,
            )?,
            RegionProvenance::Recovered { cache_key } => {
                super::schema::validate_token("region.provenance.cache_key", cache_key, 256)?
            }
            RegionProvenance::Source | RegionProvenance::Inferred => {}
        }
        Ok(())
    }
}

impl RegionGuardCondition {
    fn value(&self) -> Option<RegionValueId> {
        match self {
            Self::ValueFact { value, .. }
            | Self::Shape { value, .. }
            | Self::Class { value, .. }
            | Self::Residency { value, .. }
            | Self::Alias { value, .. } => Some(*value),
            Self::RuntimeState { .. } | Self::Capability { .. } => None,
        }
    }
}

fn ensure_sorted_unique<T: Ord>(path: &str, values: &[T]) -> Result<(), SchemaValidationError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SchemaValidationError::new(
            path,
            "entries must be sorted and unique",
        ));
    }
    Ok(())
}

fn ensure_sorted_unique_by<T, K: Ord + Copy>(
    path: &str,
    values: &[T],
    key: impl Fn(&T) -> K,
) -> Result<(), SchemaValidationError> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        return Err(SchemaValidationError::new(
            path,
            "entries must be sorted and unique by identity",
        ));
    }
    Ok(())
}
