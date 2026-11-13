use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestedOutputCount {
    Zero,
    One,
    Exactly(usize),
    CurrentFunctionNargout,
    /// Determined from the evaluated comma-separated assignment destination.
    DestinationSequenceCardinality,
}

/// Describes how an expression context consumes a value sequence.
///
/// Most expressions require one value. Bracketed output lists consume a
/// statically requested prefix, call arguments expand the complete sequence,
/// and suppressed expression statements validate the source without retaining
/// any of its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequenceUse {
    RequireSingle,
    /// Require at least `count` values and return exactly that prefix.
    SelectPrefix {
        count: usize,
    },
    /// Select the prefix requested by the active function invocation.
    SelectCurrentFunctionOutputs,
    /// Select the prefix determined by an evaluated sequence destination.
    SelectDestinationCardinality,
    ExpandAll,
    Discard,
}

impl SequenceUse {
    pub const fn from_requested_outputs(requested: RequestedOutputCount) -> Self {
        match requested {
            RequestedOutputCount::Zero | RequestedOutputCount::Exactly(0) => Self::Discard,
            RequestedOutputCount::One | RequestedOutputCount::Exactly(1) => Self::RequireSingle,
            RequestedOutputCount::Exactly(count) => Self::SelectPrefix { count },
            RequestedOutputCount::CurrentFunctionNargout => Self::SelectCurrentFunctionOutputs,
            RequestedOutputCount::DestinationSequenceCardinality => {
                Self::SelectDestinationCardinality
            }
        }
    }

    pub const fn output_count(self) -> Option<usize> {
        match self {
            Self::RequireSingle => Some(1),
            Self::SelectPrefix { count } => Some(count),
            Self::SelectCurrentFunctionOutputs
            | Self::SelectDestinationCardinality
            | Self::ExpandAll => None,
            Self::Discard => Some(0),
        }
    }
}

impl RequestedOutputCount {
    /// Return the statically known requested count. `CurrentFunctionNargout`
    /// remains dynamic instead of being silently treated as one output.
    pub const fn known_count(self) -> Option<usize> {
        match self {
            Self::Zero => Some(0),
            Self::One => Some(1),
            Self::Exactly(count) => Some(count),
            Self::CurrentFunctionNargout | Self::DestinationSequenceCardinality => None,
        }
    }

    /// Count represented by the legacy fixed-width executor carrier.
    /// Dynamic function `nargout` uses its established one-value carrier;
    /// destination cardinality has no such representation and must be resolved
    /// from a prepared destination plan.
    pub const fn executor_carrier_count(self) -> Option<usize> {
        match self {
            Self::CurrentFunctionNargout => Some(1),
            Self::DestinationSequenceCardinality => None,
            _ => self.known_count(),
        }
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

#[cfg(test)]
mod tests {
    use super::RequestedOutputCount;

    #[test]
    fn destination_cardinality_cannot_be_guessed_as_one() {
        assert_eq!(
            RequestedOutputCount::DestinationSequenceCardinality.executor_carrier_count(),
            None
        );
    }
}
