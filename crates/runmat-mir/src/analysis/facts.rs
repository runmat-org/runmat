use crate::AsyncBehaviorFact;
use runmat_hir::{EnvironmentEffect, ShapeFact, TypeFact, ValueFlowFact, WorkspaceEffect};
use serde::{Deserialize, Serialize};

use super::InitFact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocalFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub value_flow: ValueFlowFact,
    pub initialized: InitFact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EffectSummary {
    pub workspace: Vec<WorkspaceEffect>,
    pub environment: Vec<EnvironmentEffect>,
    pub async_behavior: Option<AsyncBehaviorFact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum FusibilityFact {
    #[default]
    Unknown,
    Fusible,
    NonFusible(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ParallelSafetyFact {
    #[default]
    Unknown,
    Safe,
    ReadsSharedState,
    WritesSharedState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum AccelEligibilityFact {
    #[default]
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
