use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeSubscriptSelector {
    Value,
    Colon,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeSubscriptStep {
    Parentheses {
        selectors: Vec<BytecodeSubscriptSelector>,
    },
    Braces {
        selectors: Vec<BytecodeSubscriptSelector>,
    },
    Member(runmat_types::MemberName),
    DynamicMember,
    DottedInvoke {
        member: runmat_types::MemberName,
        arguments: Vec<BytecodeSubscriptSelector>,
    },
}

impl BytecodeSubscriptStep {
    pub fn operand_count(&self) -> usize {
        match self {
            Self::Parentheses { selectors } | Self::Braces { selectors } => selectors
                .iter()
                .filter(|selector| matches!(selector, BytecodeSubscriptSelector::Value))
                .count(),
            Self::DynamicMember => 1,
            Self::DottedInvoke { arguments, .. } => arguments
                .iter()
                .filter(|selector| matches!(selector, BytecodeSubscriptSelector::Value))
                .count(),
            Self::Member(_) => 0,
        }
    }
}
