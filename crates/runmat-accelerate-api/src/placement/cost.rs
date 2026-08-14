use runmat_execution::{
    CandidatePreparationState, EstimateConfidence, EstimateSource, ExecutionCostComponents,
    ExecutionCostEstimate,
};
use serde::{Deserialize, Serialize};

use super::ProviderFeasibilityQuery;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostQuery {
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
            required_upload_bytes: 256,
            required_download_bytes: 128,
            downstream_materialization: true,
        };
        let encoded = serde_json::to_string(&query).unwrap();
        assert_eq!(
            serde_json::from_str::<ProviderCostQuery>(&encoded).unwrap(),
            query
        );
    }
}
