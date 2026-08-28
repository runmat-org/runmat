use serde::{Deserialize, Serialize};

use crate::driver::DriverSnapshot;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DriverCheckpoint {
    pub schema_version: u16,
    pub generation: u64,
    pub snapshot: DriverSnapshot,
}

impl DriverCheckpoint {
    pub fn new(generation: u64, snapshot: DriverSnapshot) -> Self {
        Self {
            schema_version: 1,
            generation,
            snapshot,
        }
    }
}
