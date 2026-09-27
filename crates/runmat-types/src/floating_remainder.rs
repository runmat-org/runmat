//! Floating remainder strategy; integer arithmetic keeps its own engine.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatingRemainderKind {
    Modulus,
    Remainder,
}
