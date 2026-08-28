use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocalEndpoint {
    Stdio,
    #[cfg(unix)]
    UnixSocket {
        path: String,
    },
    #[cfg(windows)]
    NamedPipe {
        name: String,
    },
}

impl LocalEndpoint {
    pub const fn is_network(&self) -> bool {
        false
    }
}
