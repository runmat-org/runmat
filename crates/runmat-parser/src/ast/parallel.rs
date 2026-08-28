use super::Expr;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpmdHeader {
    Default,
    One(Expr),
    Two(Expr, Expr),
    Three(Expr, Expr, Expr),
}
