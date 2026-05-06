use crate::{BasicBlockId, MirLocalId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LivenessFacts {
    pub live_across_await: Vec<(BasicBlockId, Vec<MirLocalId>)>,
}
