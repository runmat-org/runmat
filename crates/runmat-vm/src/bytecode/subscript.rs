use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BytecodeSubscriptSelector {
    Value,
    Colon,
    /// A prepared source argument whose contents become parentheses selectors.
    Expansion(runmat_runtime::call::arguments::ArgumentSpec),
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
        semantic_fallback: Option<std::sync::Arc<runmat_types::CallableIdentity>>,
    },
}

impl BytecodeSubscriptStep {
    pub fn operand_count(&self) -> usize {
        match self {
            Self::Parentheses { selectors } | Self::Braces { selectors } => selectors
                .iter()
                .map(|selector| match selector {
                    BytecodeSubscriptSelector::Value => 1,
                    BytecodeSubscriptSelector::Colon => 0,
                    BytecodeSubscriptSelector::Expansion(spec) => spec.stack_operand_count(),
                })
                .sum(),
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
