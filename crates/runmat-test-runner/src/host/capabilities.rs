use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    Auto,
    Process,
    Worker,
    Session,
    None,
}

impl IsolationMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Process => "process",
            Self::Worker => "worker",
            Self::Session => "session",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCapabilities {
    isolation: BTreeSet<IsolationMode>,
    pub max_workers: usize,
}

impl HostCapabilities {
    pub fn new(
        isolation: impl IntoIterator<Item = IsolationMode>,
        max_workers: usize,
    ) -> RunnerResult<Self> {
        let isolation = isolation
            .into_iter()
            .filter(|mode| *mode != IsolationMode::Auto)
            .collect::<BTreeSet<_>>();
        if isolation.is_empty() {
            return Err(RunnerError::InvalidConfiguration(
                "host must expose at least one concrete isolation mode".into(),
            ));
        }
        if max_workers == 0 {
            return Err(RunnerError::InvalidConfiguration(
                "host max_workers must be greater than zero".into(),
            ));
        }
        Ok(Self {
            isolation,
            max_workers,
        })
    }

    pub fn supports(&self, mode: IsolationMode) -> bool {
        self.isolation.contains(&mode)
    }

    pub fn resolve(&self, requested: IsolationMode) -> RunnerResult<IsolationMode> {
        if requested == IsolationMode::Auto {
            return [
                IsolationMode::Process,
                IsolationMode::Worker,
                IsolationMode::Session,
                IsolationMode::None,
            ]
            .into_iter()
            .find(|mode| self.supports(*mode))
            .ok_or_else(|| {
                RunnerError::InvalidConfiguration("host has no usable isolation mode".into())
            });
        }
        if self.supports(requested) {
            return Ok(requested);
        }
        Err(RunnerError::IsolationUnavailable {
            requested: requested.as_str().into(),
            available: self
                .isolation
                .iter()
                .map(|mode| mode.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })
    }
}
