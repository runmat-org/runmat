use serde::{Deserialize, Serialize};

use crate::{
    options::RefinementIndicatorMode,
    sizing::{MeshSizingField, SizingSample},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveConvergenceStatus {
    NotStarted,
    Disabled,
    Pending,
    Converged,
    MaxIterationsReached,
    ElementBudgetReached,
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementMarker {
    pub entity_id: String,
    pub weight: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SizingFieldUpdate {
    #[serde(default)]
    pub samples: Vec<SizingSample>,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
}

impl SizingFieldUpdate {
    pub fn apply_to(self, sizing: &mut MeshSizingField) {
        if let Some(min_size_m) = self.min_size_m {
            sizing.min_size_m = Some(match sizing.min_size_m {
                Some(existing) => existing.min(min_size_m),
                None => min_size_m,
            });
        }
        if let Some(max_size_m) = self.max_size_m {
            sizing.max_size_m = Some(match sizing.max_size_m {
                Some(existing) => existing.max(max_size_m),
                None => max_size_m,
            });
        }
        sizing.samples.extend(self.samples);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveIterationSummary {
    pub iteration_index: usize,
    pub node_count: usize,
    pub element_count: usize,
    pub convergence_status: AdaptiveConvergenceStatus,
    #[serde(default)]
    pub indicators: Vec<RefinementIndicatorSummary>,
    #[serde(default)]
    pub markers: Vec<RefinementMarker>,
    #[serde(default)]
    pub sizing_update: SizingFieldUpdate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizing_field_update_merges_bounds_and_samples() {
        let mut sizing = MeshSizingField {
            global_target_size_m: Some(0.1),
            min_size_m: Some(0.02),
            max_size_m: Some(0.2),
            samples: vec![SizingSample {
                position_m: [0.0, 0.0, 0.0],
                target_size_m: 0.08,
                reason: Some("initial".to_string()),
            }],
        };

        SizingFieldUpdate {
            samples: vec![SizingSample {
                position_m: [1.0, 0.0, 0.0],
                target_size_m: 0.01,
                reason: Some("stress_gradient".to_string()),
            }],
            min_size_m: Some(0.01),
            max_size_m: Some(0.25),
        }
        .apply_to(&mut sizing);

        assert_eq!(sizing.global_target_size_m, Some(0.1));
        assert_eq!(sizing.min_size_m, Some(0.01));
        assert_eq!(sizing.max_size_m, Some(0.25));
        assert_eq!(sizing.samples.len(), 2);
    }

    #[test]
    fn adaptive_iteration_summary_round_trips() {
        let summary = AdaptiveIterationSummary {
            iteration_index: 1,
            node_count: 32,
            element_count: 96,
            convergence_status: AdaptiveConvergenceStatus::Pending,
            indicators: vec![RefinementIndicatorSummary {
                namespace: "structural".to_string(),
                name: "stress_gradient".to_string(),
                requested_mode: RefinementIndicatorMode::Auto,
                status: RefinementIndicatorStatus::Used,
                detail: Some("field available".to_string()),
            }],
            markers: vec![RefinementMarker {
                entity_id: "tet_1".to_string(),
                weight: 1.0,
                reason: "stress_gradient".to_string(),
            }],
            sizing_update: SizingFieldUpdate::default(),
        };

        let encoded = serde_json::to_string(&summary).expect("summary should serialize");
        let decoded: AdaptiveIterationSummary =
            serde_json::from_str(&encoded).expect("summary should deserialize");

        assert_eq!(decoded, summary);
    }
}
