use crate::{AsyncBehaviorFact, MirOperand, MirSequenceLocalId};
use runmat_builtins::{
    BuiltinEffects, BuiltinEnvironmentEffect, BuiltinPurity, BuiltinSemanticKind,
    BuiltinWorkspaceEffect,
};
use runmat_hir::{
    CallSyntax, CallableFallbackPolicy, CallableIdentity, RequestedOutputCount, Span,
};
use runmat_types::{ClassIdentity, MethodName};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirCall {
    pub callee: MirCallee,
    pub args: Vec<MirCallArg>,
    pub arg_spans: Vec<Span>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
    pub fallback_policy: CallableFallbackPolicy,
    #[serde(default)]
    pub workspace_first_name: Option<runmat_hir::SymbolName>,
    #[serde(default)]
    pub bare_identifier: bool,
    pub async_behavior: AsyncBehaviorFact,
    pub effects: BuiltinEffects,
    pub workspace_effect: Option<BuiltinWorkspaceEffect>,
    pub environment_effect: Option<BuiltinEnvironmentEffect>,
    pub purity: BuiltinPurity,
    pub semantic_kind: BuiltinSemanticKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCallee {
    Static(CallableIdentity),
    SuperConstructor {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
    },
    SuperMethod {
        current_class: ClassIdentity,
        super_class: ClassIdentity,
        method: MethodName,
    },
    Dynamic(MirOperand),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCallArg {
    Single(MirOperand),
    Expansion(MirExpansionSource),
    CapturedSequence(MirSequenceLocalId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirExpansionSource {
    SubscriptChain(crate::MirSubscriptChain),
    CellContents {
        base: MirOperand,
        indexing: crate::MirIndexing,
    },
    ReturnedOutputs(MirOperand),
    Member {
        base: MirOperand,
        member: runmat_hir::MemberName,
    },
    DynamicMember {
        base: MirOperand,
        member: MirOperand,
    },
}

impl MirCallArg {
    pub fn contains_contextual_end(&self) -> bool {
        matches!(self, Self::Expansion(source) if source.contains_contextual_end())
    }
    pub fn operand(&self) -> Option<&MirOperand> {
        match self {
            MirCallArg::Single(operand) => Some(operand),
            MirCallArg::Expansion(source) => Some(source.base()),
            MirCallArg::CapturedSequence(_) => None,
        }
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        match self {
            Self::Single(operand) => visitor(operand),
            Self::Expansion(source) => source.visit_operands(visitor),
            Self::CapturedSequence(_) => {}
        }
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        match self {
            Self::Single(operand) => visitor(operand),
            Self::Expansion(source) => source.visit_operands_mut(visitor),
            Self::CapturedSequence(_) => {}
        }
    }

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        if let Self::Expansion(source) = self {
            source.visit_direct_expression_regions_dyn(visitor);
        }
    }
}

impl MirExpansionSource {
    pub fn contains_contextual_end(&self) -> bool {
        match self {
            Self::SubscriptChain(chain) => chain.contains_contextual_end(),
            Self::CellContents { indexing, .. } => indexing.components.iter().any(|component| {
                matches!(component, crate::MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end())
            }),
            Self::ReturnedOutputs(_) | Self::Member { .. } | Self::DynamicMember { .. } => false,
        }
    }

    pub fn base(&self) -> &MirOperand {
        match self {
            Self::SubscriptChain(chain) => &chain.root,
            Self::CellContents { base, .. }
            | Self::Member { base, .. }
            | Self::DynamicMember { base, .. } => base,
            Self::ReturnedOutputs(base) => base,
        }
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        match self {
            Self::SubscriptChain(chain) => chain.visit_operands(visitor),
            Self::CellContents { base, indexing } => {
                visitor(base);
                indexing.visit_operands(visitor);
            }
            Self::ReturnedOutputs(base) | Self::Member { base, .. } => visitor(base),
            Self::DynamicMember { base, member } => {
                visitor(base);
                visitor(member);
            }
        }
    }

    /// Visits only operands supplied by the owning MIR body. Locals defined
    /// inside contextual selector regions remain private to those regions.
    pub fn visit_outer_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        self.visit_outer_operands_dyn(&mut visitor);
    }

    pub(crate) fn visit_outer_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        match self {
            Self::SubscriptChain(chain) => chain.visit_outer_operands(visitor),
            Self::CellContents { base, indexing } => {
                visitor(base);
                indexing.visit_outer_operands_dyn(visitor);
            }
            Self::ReturnedOutputs(base) | Self::Member { base, .. } => visitor(base),
            Self::DynamicMember { base, member } => {
                visitor(base);
                visitor(member);
            }
        }
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        match self {
            Self::SubscriptChain(chain) => chain.visit_operands_mut(visitor),
            Self::CellContents { base, indexing } => {
                visitor(base);
                indexing.visit_operands_mut(visitor);
            }
            Self::ReturnedOutputs(base) | Self::Member { base, .. } => visitor(base),
            Self::DynamicMember { base, member } => {
                visitor(base);
                visitor(member);
            }
        }
    }

    pub(crate) fn visit_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        match self {
            Self::SubscriptChain(chain) => chain.visit_expression_regions(visitor),
            Self::CellContents { indexing, .. } => indexing.visit_expression_regions_dyn(visitor),
            Self::ReturnedOutputs(_) | Self::Member { .. } | Self::DynamicMember { .. } => {}
        }
    }

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        match self {
            Self::SubscriptChain(chain) => chain.visit_direct_expression_regions_dyn(visitor),
            Self::CellContents { indexing, .. } => {
                indexing.visit_direct_expression_regions_dyn(visitor)
            }
            Self::ReturnedOutputs(_) | Self::Member { .. } | Self::DynamicMember { .. } => {}
        }
    }

    pub fn visit_expression_regions(&self, mut visitor: impl FnMut(&crate::MirExpressionRegion)) {
        self.visit_expression_regions_dyn(&mut visitor);
    }
}
