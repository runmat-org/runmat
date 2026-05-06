use crate::AsyncBehaviorFact;
use runmat_hir::{EnvironmentEffect, ShapeFact, TypeFact, ValueFlowFact, WorkspaceEffect};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocalFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub value_flow: ValueFlowFact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EffectSummary {
    pub workspace: Vec<WorkspaceEffect>,
    pub environment: Vec<EnvironmentEffect>,
    pub async_behavior: Option<AsyncBehaviorFact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusibilityFact {
    Unknown,
    Fusible,
    NonFusible(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParallelSafetyFact {
    Unknown,
    Safe,
    ReadsSharedState,
    WritesSharedState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccelEligibilityFact {
    Unknown,
    Ineligible(String),
    Eligible,
    Preferred,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataMovementPolicyHint {
    Unknown,
    KeepHost,
    KeepDeviceIfAlreadyThere,
    PreferDeviceForLargeInputs,
}
