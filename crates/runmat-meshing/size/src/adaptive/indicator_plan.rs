use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::refinement::{MeshRefinementOptions, RefinementIndicatorMode, RefinementStrategy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementIndicatorStatus {
    Used,
    SkippedMissingField,
    SkippedNotApplicable,
    SkippedBudget,
    SkippedQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementIndicatorSummary {
    pub namespace: String,
    pub name: String,
    pub requested_mode: RefinementIndicatorMode,
    pub status: RefinementIndicatorStatus,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RefinementIndicatorKey {
    pub namespace: String,
    pub name: String,
}

impl RefinementIndicatorKey {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementIndicatorAvailability {
    pub key: RefinementIndicatorKey,
    pub applicable: bool,
    pub field_available: bool,
}

pub fn plan_refinement_indicators(
    options: &MeshRefinementOptions,
    defaults: &[RefinementIndicatorKey],
    availability: &[RefinementIndicatorAvailability],
    element_budget_reached: bool,
    quality_blocked: bool,
) -> Vec<RefinementIndicatorSummary> {
    if matches!(
        options.strategy,
        RefinementStrategy::None | RefinementStrategy::Uniform
    ) {
        return Vec::new();
    }

    let availability_by_key = availability
        .iter()
        .map(|item| (item.key.clone(), item))
        .collect::<BTreeMap<_, _>>();
    let overrides = options
        .indicators
        .namespaces
        .iter()
        .flat_map(|(namespace, names)| {
            names.iter().map(|(name, mode)| {
                (
                    RefinementIndicatorKey::new(namespace.clone(), name.clone()),
                    *mode,
                )
            })
        })
        .collect::<BTreeMap<_, _>>();

    let mut keys = defaults.iter().cloned().collect::<BTreeSet<_>>();
    keys.extend(overrides.keys().cloned());

    keys.into_iter()
        .map(|key| {
            let requested_mode = overrides
                .get(&key)
                .copied()
                .unwrap_or(RefinementIndicatorMode::Auto);
            let (status, detail) = if matches!(requested_mode, RefinementIndicatorMode::Off) {
                (
                    RefinementIndicatorStatus::SkippedNotApplicable,
                    Some("indicator disabled by override".to_string()),
                )
            } else if element_budget_reached {
                (
                    RefinementIndicatorStatus::SkippedBudget,
                    Some("element budget reached".to_string()),
                )
            } else if quality_blocked {
                (
                    RefinementIndicatorStatus::SkippedQuality,
                    Some("mesh quality constraint blocked refinement".to_string()),
                )
            } else if let Some(available) = availability_by_key.get(&key) {
                if !available.applicable {
                    (
                        RefinementIndicatorStatus::SkippedNotApplicable,
                        Some("indicator does not apply to the active analysis".to_string()),
                    )
                } else if !available.field_available {
                    (
                        RefinementIndicatorStatus::SkippedMissingField,
                        Some("required recovered field is unavailable".to_string()),
                    )
                } else {
                    (RefinementIndicatorStatus::Used, None)
                }
            } else if matches!(requested_mode, RefinementIndicatorMode::On) {
                (
                    RefinementIndicatorStatus::SkippedMissingField,
                    Some("required recovered field is unavailable".to_string()),
                )
            } else {
                (
                    RefinementIndicatorStatus::SkippedNotApplicable,
                    Some("indicator was not selected by the active analysis".to_string()),
                )
            };

            RefinementIndicatorSummary {
                namespace: key.namespace,
                name: key.name,
                requested_mode,
                status,
                detail,
            }
        })
        .collect()
}
