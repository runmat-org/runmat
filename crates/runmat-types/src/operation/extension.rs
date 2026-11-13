use super::{
    OperationCatalogExtensionIdentity, OperationDistribution, OperationEffectFacts,
    OperationFrontend, OperationIdentity, OperationLoweringId, OperationOverloadIdentity,
    OperationQuantityRule, OperationSignature, TopologyApplicability,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationOverloadFacts {
    pub identity: OperationOverloadIdentity,
    pub frontends: Vec<OperationFrontend>,
    pub signature: OperationSignature,
    pub quantity: OperationQuantityRule,
    pub topology: TopologyApplicability,
    pub effects: OperationEffectFacts,
    pub distribution: OperationDistribution,
    pub lowering: OperationLoweringId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationFacts {
    pub identity: OperationIdentity,
    pub category: String,
    pub summary: String,
    pub overloads: Vec<OperationOverloadFacts>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationCatalogExtension {
    pub identity: OperationCatalogExtensionIdentity,
    pub operations: Vec<OperationFacts>,
}
