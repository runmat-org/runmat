use serde::{Deserialize, Serialize};

use super::{
    quantity::{Damping, Stiffness},
    target::RegionTarget,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisConnector {
    pub connector_id: String,
    pub primary: RegionTarget,
    pub secondary: Option<RegionTarget>,
    pub kind: ConnectorKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorKind {
    TranslationalSpring { stiffness: Stiffness },
    TranslationalDashpot { damping: Damping },
    ElasticFoundation { stiffness: Stiffness },
}
