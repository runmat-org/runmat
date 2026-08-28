use serde::{Deserialize, Serialize};

use crate::descriptor::{FixtureScope, ProcedureDescriptor};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct FixtureScopeKey {
    pub scope: FixtureScope,
    pub identity: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LifecycleStep {
    pub scope: FixtureScopeKey,
    pub procedure: ProcedureDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisteredTeardown {
    pub scope: FixtureScopeKey,
    pub procedure: ProcedureDescriptor,
    pub registration_order: u64,
}
