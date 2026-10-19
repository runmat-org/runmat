use serde::{Deserialize, Serialize};

use super::{
    Count, PreparedInputResourceFacts, PreparedOutputResourceFacts,
    StructuralOperatorResourceDemand,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRandomResponseIntervalResourceDemand {
    pub operator: StructuralOperatorResourceDemand,
    pub interval_count: Count,
    pub channel_count: Count,
    pub spectral_matrix_complex_value_count: Count,
    pub selected_mode_count: Count,
    pub physical_dof_count: Count,
    pub linear_field_value_count: Count,
    pub response_value_count: Count,
    pub maximum_frequency_evaluation_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
