use runmat_hir::{BindingId, ExprId, FunctionId, TypeFact};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{FunctionSummary, LivenessFacts, MirLocalFact};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AnalysisStore {
    pub bindings: HashMap<BindingId, BindingFact>,
    pub expressions: HashMap<ExprId, ExprFact>,
    pub mir_locals: HashMap<crate::MirLocalId, MirLocalFact>,
    pub functions: HashMap<FunctionId, FunctionSummary>,
    pub liveness: HashMap<FunctionId, LivenessFacts>,
    pub spawn_boundaries: HashMap<FunctionId, Vec<crate::SpawnBoundary>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingFact {
    pub ty: TypeFact,
    pub initialized: InitFact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExprFact {
    pub ty: TypeFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
