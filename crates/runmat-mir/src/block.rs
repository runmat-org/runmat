use crate::{BasicBlockId, MirStmt, MirTerminator};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<MirStmt>,
    pub terminator: MirTerminator,
}
