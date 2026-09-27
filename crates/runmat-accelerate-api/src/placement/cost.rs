use runmat_execution::{
    CandidatePreparationState, EstimateConfidence, EstimateSource, ExecutionCostComponents,
    ExecutionCostEstimate,
};
use serde::{Deserialize, Serialize};

use super::ProviderFeasibilityQuery;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostContext {
    pub preparation: Option<super::ProviderPreparationSnapshot>,
    /// Present only when this selected route will request completion evidence.
    pub observation: Option<crate::observation::ProviderObservationOverhead>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostQuery {
    #[serde(default)]
    pub context: ProviderCostContext,
    /// The operation contract that placement has already proven feasible.
    pub operation: ProviderFeasibilityQuery,
    pub preparation: CandidatePreparationState,
    pub required_upload_bytes: u64,
    pub required_download_bytes: u64,
    pub downstream_materialization: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostEstimate {
    pub cost: ExecutionCostEstimate,
}

impl ProviderCostEstimate {
    pub fn static_prior(
        components: ExecutionCostComponents,
        scratch_bytes: u64,
    ) -> ProviderCostEstimate {
        Self {
            cost: ExecutionCostEstimate {
                components,
                scratch_bytes,
                confidence: EstimateConfidence::Prior,
                source: EstimateSource::StaticPrior,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderWorkload,
    };

    #[test]
    fn cost_query_round_trips_without_losing_transfer_intent() {
        let query = ProviderCostQuery {
            operation: ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new("test.fusion"),
                family: ProviderOperationFamily::Fusion,
                inputs: Vec::new(),
                outputs: Vec::new(),
                workload: ProviderWorkload {
                    elements: Some(32),
                    ..ProviderWorkload::default()
                },
            },
            preparation: CandidatePreparationState::Cold,
            context: Default::default(),
            required_upload_bytes: 256,
            required_download_bytes: 128,
            downstream_materialization: true,
        };
        let encoded = serde_json::to_string(&query).unwrap();
        assert_eq!(
            serde_json::from_str::<ProviderCostQuery>(&encoded).unwrap(),
            query
        );
        let mut legacy = serde_json::to_value(&query).unwrap();
        legacy.as_object_mut().unwrap().remove("context");
        assert_eq!(
            serde_json::from_value::<ProviderCostQuery>(legacy).unwrap(),
            query
        );
        let mut observed = query;
        observed.context.preparation = Some(super::super::ProviderPreparationSnapshot {
            identity: runmat_execution::Digest::sha256(b"pipeline inventory"),
            state: None,
            cold_pipelines: 0,
            warm_pipelines: 0,
        });
        observed.context.observation = Some(crate::observation::ProviderObservationOverhead {
            submissions: 2,
            requested_peak_buffer_bytes: 16,
            clear_commands: 2,
            map_callbacks: 2,
        });
        assert_eq!(
            serde_json::from_value::<ProviderCostQuery>(serde_json::to_value(&observed).unwrap())
                .unwrap(),
            observed
        );
    }
}
