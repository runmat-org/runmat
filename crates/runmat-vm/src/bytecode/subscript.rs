use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeSubscriptSelector {
    Value,
    Colon,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeDottedInvokeArgument {
    Colon,
    Call(runmat_runtime::call::arguments::ArgumentSpec),
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
        arguments: Vec<BytecodeDottedInvokeArgument>,
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
                .map(|argument| match argument {
                    BytecodeDottedInvokeArgument::Colon => 0,
                    BytecodeDottedInvokeArgument::Call(spec) => spec.stack_operand_count(),
                })
                .sum(),
            Self::Member(_) => 0,
        }
    }
}
