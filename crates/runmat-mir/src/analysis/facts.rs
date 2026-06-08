use runmat_hir::{AsyncValueFact, ShapeFact, TypeFact, ValueFlowFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirLocalFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub value_flow: ValueFlowFact,
    pub async_value: Option<AsyncValueFact>,
}
