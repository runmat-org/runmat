use serde::{Deserialize, Serialize};

use super::ValueFact;

/// Static facts for a transient source-level value sequence.
///
/// This is deliberately separate from [`ValueFact`]: a sequence can be
/// selected or expanded by an expression context, but cannot inhabit a value
/// local or aggregate element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueSequenceFact {
    pub outputs: Vec<ValueFact>,
    pub variadic: bool,
}

impl ValueSequenceFact {
    pub fn fixed(outputs: Vec<ValueFact>) -> Self {
        Self {
            outputs,
            variadic: false,
        }
    }

    pub fn single(output: ValueFact) -> Self {
        Self::fixed(vec![output])
    }

    pub fn dynamic() -> Self {
        Self {
            outputs: Vec::new(),
            variadic: true,
        }
    }

    pub fn first_or_else(&self, fallback: impl FnOnce() -> ValueFact) -> ValueFact {
        self.outputs.first().cloned().unwrap_or_else(fallback)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceFactInference {
    pub sequence: ValueSequenceFact,
    pub diagnostics: Vec<crate::InferenceDiagnostic>,
}

impl SequenceFactInference {
    pub fn exact(sequence: ValueSequenceFact) -> Self {
        Self {
            sequence,
            diagnostics: Vec::new(),
        }
    }
}
