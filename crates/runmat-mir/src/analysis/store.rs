use runmat_hir::{
    AsyncValueFact, BindingId, EnvironmentEffect, ExprId, FunctionId, ModuleId, ShapeFact,
    TypeFact, ValueFlowFact, WorkspaceEffect,
};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

use crate::{MirDiagnostic, MirLocalId};

use super::{FunctionSummary, LivenessFacts, MirLocalFact};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AnalysisStore {
    pub bindings: HashMap<BindingId, BindingFact>,
    pub expressions: HashMap<ExprId, ExprFact>,
    pub mir_locals: HashMap<MirLocalKey, MirLocalFact>,
    pub functions: HashMap<FunctionId, FunctionSummary>,
    pub modules: HashMap<ModuleId, ModuleSummary>,
    pub liveness: HashMap<FunctionId, LivenessFacts>,
    pub spawn_boundaries: HashMap<FunctionId, Vec<crate::SpawnBoundary>>,
    pub diagnostics: Vec<MirDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleSummary {
    pub module: ModuleId,
    pub functions: Vec<FunctionId>,
    pub workspace: Vec<WorkspaceEffect>,
    pub environment: Vec<EnvironmentEffect>,
    pub may_call_unknown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirLocalKey {
    pub function: FunctionId,
    pub local: MirLocalId,
}

impl Serialize for MirLocalKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}:{}", self.function.0, self.local.0))
    }
}

impl<'de> Deserialize<'de> for MirLocalKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let Some((function, local)) = value.split_once(':') else {
            return Err(de::Error::custom(
                "expected MIR local key as function:local",
            ));
        };
        Ok(Self {
            function: FunctionId(function.parse().map_err(de::Error::custom)?),
            local: MirLocalId(local.parse().map_err(de::Error::custom)?),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingFact {
    pub ty: TypeFact,
    pub initialized: InitFact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExprFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub value_flow: ValueFlowFact,
    pub async_value: Option<AsyncValueFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
