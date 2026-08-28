use super::NativeLocalId;
use runmat_mir::MirOperand;
use runmat_runtime::indexing::EndExpr;
use serde::{Deserialize, Serialize};

/// Self-contained context-dependent selector metadata derived during MIR
/// lowering. The host never has to rediscover `end` provenance from executed
/// sentinel values or from bytecode layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeIndexExpression {
    pub local: NativeLocalId,
    pub kind: NativeIndexExpressionKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum NativeIndexExpressionKind {
    Scalar(EndExpr),
    Range(Box<NativeRangeExpression>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRangeExpression {
    pub start: NativeIndexBound,
    pub step: Option<NativeIndexBound>,
    pub end: EndExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum NativeIndexBound {
    Operand(MirOperand),
    Expression(EndExpr),
}
