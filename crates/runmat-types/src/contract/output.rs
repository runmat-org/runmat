use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestedOutputCount {
    Zero,
    One,
    Exactly(usize),
    CurrentFunctionNargout,
}

impl RequestedOutputCount {
    /// Return the statically known requested count. `CurrentFunctionNargout`
    /// remains dynamic instead of being silently treated as one output.
    pub fn known_count(self) -> Option<usize> {
        match self {
            Self::Zero => Some(0),
            Self::One => Some(1),
            Self::Exactly(count) => Some(count),
            Self::CurrentFunctionNargout => None,
        }
    }

    /// Compatibility accessor used by executable lowering, where dynamic
    /// nargout is represented by the existing one-output carrier.
    pub fn fixed_count(self) -> usize {
        self.known_count().unwrap_or(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OutputSelection {
    pub requested: RequestedOutputCount,
    /// Zero-based result slots that are validated but intentionally discarded.
    pub discarded: BTreeSet<usize>,
}

impl OutputSelection {
    pub fn new(requested: RequestedOutputCount) -> Self {
        Self {
            requested,
            discarded: BTreeSet::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let Some(count) = self.requested.known_count() else {
            return Ok(());
        };
        if let Some(index) = self.discarded.iter().find(|index| **index >= count) {
            return Err(format!(
                "discarded output slot {index} is outside requested output count {count}"
            ));
        }
        Ok(())
    }
}

impl Default for RequestedOutputCount {
    fn default() -> Self {
        Self::One
    }
}
