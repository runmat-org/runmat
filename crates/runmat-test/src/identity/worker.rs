use serde::{Deserialize, Serialize};

/// Identity of one native worker process/session, not a particular test case.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct WorkerSessionId(pub String);
