use crate::MirOperand;
use runmat_hir::{IndexKind, IndexResultContext};
use serde::{Deserialize, Serialize};

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
    Member(runmat_hir::MemberName),
    DynamicMember(MirOperand),
    DottedInvoke {
        member: runmat_hir::MemberName,
        indexing: MirIndexing,
    },
}

impl MirSubscriptChain {
    pub fn contains_contextual_end(&self) -> bool {
        self.steps.iter().any(|step| match step {
            MirSubscriptStep::Index(indexing)
            | MirSubscriptStep::DottedInvoke { indexing, .. } => indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end())),
            MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => false,
        })
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        visitor(&self.root);
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing) => indexing.visit_operands(&mut visitor),
                MirSubscriptStep::DottedInvoke { indexing, .. } => {
                    indexing.visit_operands(&mut visitor)
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
                MirSubscriptStep::DottedInvoke { indexing, .. } => {
                    indexing.visit_outer_operands(&mut visitor)
                }
                MirSubscriptStep::DynamicMember(member) => visitor(member),
                MirSubscriptStep::Member(_) => {}
            }
        }
    }

    pub fn visit_expression_regions(&self, mut visitor: impl FnMut(&crate::MirExpressionRegion)) {
        for step in &self.steps {
            match step {
                MirSubscriptStep::Index(indexing)
                | MirSubscriptStep::DottedInvoke { indexing, .. } => {
                    indexing.visit_expression_regions(&mut visitor)
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
                MirSubscriptStep::Index(indexing)
                | MirSubscriptStep::DottedInvoke { indexing, .. } => {
                    indexing.visit_direct_expression_regions_dyn(visitor)
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
                MirSubscriptStep::DottedInvoke { indexing, .. } => {
                    indexing.visit_operands_mut(&mut visitor)
                }
                MirSubscriptStep::DynamicMember(member) => visitor(member),
                MirSubscriptStep::Member(_) => {}
            }
        }
    }
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
