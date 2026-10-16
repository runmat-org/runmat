use crate::{MirExpressionStep, MirOperand, MirSequenceLocalId};
use runmat_types::{IndexKind, IndexResultContext};
use serde::{Deserialize, Serialize};
use std::fmt;

mod dotted_argument;
pub use dotted_argument::{MirCallArgumentRegion, MirDottedInvokeArgument};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirSubscriptChain {
    pub root: MirOperand,
    pub steps: Vec<MirSubscriptStep>,
    pub sequence_use: runmat_types::SequenceUse,
    pub context: runmat_types::ObjectIndexingContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirSubscriptStep {
    Index(MirIndexing),
    Member(runmat_types::MemberName),
    DynamicMember(MirOperand),
    DottedInvoke {
        member: runmat_types::MemberName,
        arguments: Vec<MirDottedInvokeArgument>,
        semantic_fallback: Option<runmat_types::CallableIdentity>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirSubscriptChainError {
    Empty,
    InvalidReadContext,
    ParenthesesWithCellPlan,
    BracesWithoutCellPlan,
    InvalidCellExpandAll,
    InvalidDottedArgumentRegion,
}

impl fmt::Display for MirSubscriptChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "subscript chain has no path steps",
            Self::InvalidReadContext => {
                "subscript chain index step does not have a read result context"
            }
            Self::ParenthesesWithCellPlan => "parentheses subscript step uses a cell index plan",
            Self::BracesWithoutCellPlan => "braces subscript step does not use a cell index plan",
            Self::InvalidCellExpandAll => {
                "cell expansion marker does not match an all-colon braces step"
            }
            Self::InvalidDottedArgumentRegion => {
                "dotted invocation argument preparation is invalid"
            }
        })
    }
}

impl MirSubscriptStep {
    /// Call syntax represented directly by this ordered path step.
    pub fn call_syntax(&self) -> Option<runmat_types::CallSyntax> {
        match self {
            Self::DottedInvoke { .. } => Some(runmat_types::CallSyntax::DottedInvoke),
            Self::Index(_) | Self::Member(_) | Self::DynamicMember(_) => None,
        }
    }

    /// Fallback policy attached to a call-shaped ordered path step.
    pub fn fallback_policy(&self) -> Option<runmat_types::CallableFallbackPolicy> {
        match self {
            Self::DottedInvoke { .. } => Some(runmat_types::CallableFallbackPolicy::ObjectDispatch),
            Self::Index(_) | Self::Member(_) | Self::DynamicMember(_) => None,
        }
    }
}

impl MirSubscriptChain {
    pub(crate) fn visit_sequence_locals(&self, visitor: &mut dyn FnMut(&MirSequenceLocalId)) {
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing.visit_sequence_locals(visitor),
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_sequence_locals(visitor);
                    }
                }
                MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => {}
            }
        }
    }
    pub fn visit_direct_expression_steps(&self, mut visitor: impl FnMut(&[MirExpressionStep])) {
        self.visit_direct_expression_steps_dyn(&mut visitor);
    }

    pub(crate) fn visit_direct_expression_steps_dyn(
        &self,
        visitor: &mut dyn FnMut(&[MirExpressionStep]),
    ) {
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing
                    .visit_direct_expression_regions_dyn(&mut |region| visitor(region.steps())),
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        match argument {
                            MirDottedInvokeArgument::Colon => {}
                            MirDottedInvokeArgument::Contextual(region) => visitor(region.steps()),
                            MirDottedInvokeArgument::Call(region) => {
                                region.visit_direct_preparation_steps(visitor)
                            }
                        }
                    }
                }
                MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => {}
            }
        }
    }

    pub fn validate(&self) -> Result<(), MirSubscriptChainError> {
        if self.steps.is_empty() {
            return Err(MirSubscriptChainError::Empty);
        }
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => validate_index_step(indexing)?,
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    if arguments
                        .iter()
                        .any(|argument| argument.validate().is_err())
                    {
                        return Err(MirSubscriptChainError::InvalidDottedArgumentRegion);
                    }
                }
                MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => {}
            }
        }
        Ok(())
    }

    pub fn contains_contextual_end(&self) -> bool {
        self.steps.iter().any(|step| match step {
            MirSubscriptStep::Index(indexing) => indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end())),
            MirSubscriptStep::DottedInvoke { arguments, .. } =>
                arguments.iter().any(MirDottedInvokeArgument::contains_contextual_end),
            MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => false,
        })
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        visitor(&self.root);
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing.visit_operands(&mut visitor),
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_operands(&mut visitor);
                    }
                }
                MirSubscriptStep::DynamicMember(member) => visitor(member),
                MirSubscriptStep::Member(_) => {}
            }
        }
    }

    pub fn visit_outer_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        visitor(&self.root);
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing.visit_outer_operands(&mut visitor),
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_outer_operands(&mut visitor);
                    }
                }
                MirSubscriptStep::DynamicMember(member) => visitor(member),
                MirSubscriptStep::Member(_) => {}
            }
        }
    }

    pub fn visit_expression_regions(&self, mut visitor: impl FnMut(&crate::MirExpressionRegion)) {
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => {
                    indexing.visit_expression_regions(&mut visitor)
                }
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_expression_regions(&mut visitor);
                    }
                }
                MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => {}
            }
        }
    }

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => {
                    indexing.visit_direct_expression_regions_dyn(visitor)
                }
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_direct_expression_regions_dyn(visitor);
                    }
                }
                MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => {}
            }
        }
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        visitor(&mut self.root);
        for step in &mut self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing.visit_operands_mut(&mut visitor),
                MirSubscriptStep::DottedInvoke { arguments, .. } => {
                    for argument in arguments {
                        argument.visit_operands_mut(&mut visitor);
                    }
                }
                MirSubscriptStep::DynamicMember(member) => visitor(member),
                MirSubscriptStep::Member(_) => {}
            }
        }
    }
}

fn validate_index_step(indexing: &MirIndexing) -> Result<(), MirSubscriptChainError> {
    if !matches!(
        indexing.result_context,
        IndexResultContext::ReadSingle | IndexResultContext::ReadCommaList
    ) {
        return Err(MirSubscriptChainError::InvalidReadContext);
    }
    validate_index_plan(indexing)
}

fn validate_index_plan(indexing: &MirIndexing) -> Result<(), MirSubscriptChainError> {
    match (indexing.kind, indexing.plan) {
        (IndexKind::Paren, MirIndexPlan::Cell) => {
            return Err(MirSubscriptChainError::ParenthesesWithCellPlan)
        }
        (IndexKind::Brace, MirIndexPlan::Scalar | MirIndexPlan::Slice) => {
            return Err(MirSubscriptChainError::BracesWithoutCellPlan)
        }
        (IndexKind::Paren, MirIndexPlan::Scalar | MirIndexPlan::Slice)
        | (IndexKind::Brace, MirIndexPlan::Cell) => {}
    }
    let all_colon = indexing
        .components
        .iter()
        .all(|component| matches!(component, MirIndexComponent::Colon));
    if indexing.cell_expand_all != (indexing.kind == IndexKind::Brace && all_colon) {
        return Err(MirSubscriptChainError::InvalidCellExpandAll);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirIndexing {
    pub kind: IndexKind,
    pub plan: MirIndexPlan,
    pub components: Vec<MirIndexComponent>,
    pub result_context: IndexResultContext,
    #[serde(default)]
    pub cell_expand_all: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirIndexPlan {
    Scalar,
    Slice,
    Cell,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirIndexComponent {
    Colon,
    Expr(MirOperand),
    ContextualExpr(crate::MirExpressionRegion),
}

impl MirIndexing {
    pub(crate) fn visit_sequence_locals(&self, visitor: &mut dyn FnMut(&MirSequenceLocalId)) {
        for component in &self.components {
            if let MirIndexComponent::ContextualExpr(region) = component {
                let local_sequences = region
                    .defined_sequences()
                    .collect::<std::collections::BTreeSet<_>>();
                region.visit_sequence_locals(|sequence| {
                    if !local_sequences.contains(sequence) {
                        visitor(sequence);
                    }
                });
            }
        }
    }
    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        for component in &self.components {
            if let MirIndexComponent::ContextualExpr(region) = component {
                visitor(region);
            }
        }
    }

    pub fn visit_expression_regions(&self, mut visitor: impl FnMut(&crate::MirExpressionRegion)) {
        self.visit_expression_regions_dyn(&mut visitor);
    }

    pub(crate) fn visit_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        self.visit_direct_expression_regions_dyn(&mut |region| {
            visitor(region);
            region.visit_nested_regions(visitor);
        });
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        self.visit_operands_dyn(&mut visitor);
    }

    pub(crate) fn visit_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        for component in &self.components {
            if let MirIndexComponent::Expr(operand) = component {
                visitor(operand);
            } else if let MirIndexComponent::ContextualExpr(region) = component {
                region.visit_operands_dyn(visitor);
            }
        }
    }

    pub(crate) fn visit_outer_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        for component in &self.components {
            if let MirIndexComponent::Expr(operand) = component {
                visitor(operand);
            } else if let MirIndexComponent::ContextualExpr(region) = component {
                region.visit_external_operands(&mut *visitor);
            }
        }
    }

    pub fn visit_outer_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        self.visit_outer_operands_dyn(&mut visitor);
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        self.visit_operands_mut_dyn(&mut visitor);
    }

    pub(crate) fn visit_operands_mut_dyn(&mut self, visitor: &mut dyn FnMut(&mut MirOperand)) {
        for component in &mut self.components {
            if let MirIndexComponent::Expr(operand) = component {
                visitor(operand);
            } else if let MirIndexComponent::ContextualExpr(region) = component {
                region.visit_operands_mut_dyn(visitor);
            }
        }
    }
}
