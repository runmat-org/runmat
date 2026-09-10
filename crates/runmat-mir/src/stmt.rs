use crate::{MirExpansionSource, MirLocalId, MirPlace, MirRvalue, MirSequenceLocalId};
use runmat_hir::{
    AssignmentCreationPolicy, AssignmentShapePolicy, EnvironmentEffect, PlaceMutationKind,
    RequestedOutputCount, Span, WorkspaceEffect,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirStmt {
    pub kind: MirStmtKind,
    pub span: Span,
}

impl MirStmt {
    pub fn visit_expression_regions(&self, mut visitor: impl FnMut(&crate::MirExpressionRegion)) {
        self.kind.visit_expression_regions_dyn(&mut visitor);
    }
}

impl MirStmtKind {
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

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        fn visit_place(
            place_value: &MirPlace,
            visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
        ) {
            match place_value {
                MirPlace::Local(_) | MirPlace::Binding(_) => {}
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => {
                    visit_place(base, visitor)
                }
                MirPlace::Index(base, indexing) => {
                    visit_place(base, visitor);
                    indexing.visit_direct_expression_regions_dyn(visitor);
                }
            }
        }
        match self {
            MirStmtKind::CaptureSequence { source, .. } => {
                source.visit_direct_expression_regions_dyn(visitor)
            }
            MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
            MirStmtKind::Assign {
                place: target,
                value,
            } => {
                visit_place(target, visitor);
                value.visit_direct_expression_regions_dyn(visitor);
            }
            MirStmtKind::MultiAssign { targets, value } => {
                for target in &targets.targets {
                    match target {
                        MirOutputTarget::Place(target) => visit_place(target, visitor),
                        MirOutputTarget::Sequence(target) => {
                            visit_place(target.base(), visitor);
                            if let MirSequenceTarget::CellContents { indexing, .. } = target {
                                indexing.visit_direct_expression_regions_dyn(visitor);
                            }
                        }
                        MirOutputTarget::Discard => {}
                    }
                }
                value.visit_direct_expression_regions_dyn(visitor);
            }
            MirStmtKind::SequenceAssign { target, value } => {
                visit_place(target.base(), visitor);
                if let MirSequenceTarget::CellContents { indexing, .. } = target {
                    indexing.visit_direct_expression_regions_dyn(visitor);
                }
                value.visit_direct_expression_regions_dyn(visitor);
            }
            MirStmtKind::Expr(value) => value.visit_direct_expression_regions_dyn(visitor),
            // PlaceMutation is a contract annotation for the immediately
            // following store. The store is the sole owner of contextual
            // selector regions; traversing the cloned annotation would count
            // the same private definitions twice and make execution metadata
            // look like a second evaluation site.
            MirStmtKind::PlaceMutation(_) => {}
        }
    }

    /// Visits contextual regions executed by the statement phase itself.
    /// RHS regions belong to the preceding rvalue phase and are deliberately
    /// excluded so effects and safepoints have one owner.
    pub(crate) fn visit_statement_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        fn visit_place(place: &MirPlace, visitor: &mut dyn FnMut(&crate::MirExpressionRegion)) {
            match place {
                MirPlace::Local(_) | MirPlace::Binding(_) => {}
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => {
                    visit_place(base, visitor)
                }
                MirPlace::Index(base, indexing) => {
                    visit_place(base, visitor);
                    indexing.visit_direct_expression_regions_dyn(visitor);
                }
            }
        }
        match self {
            MirStmtKind::CaptureSequence { source, .. } => {
                source.visit_direct_expression_regions_dyn(visitor)
            }
            MirStmtKind::Assign { place, .. } => visit_place(place, visitor),
            MirStmtKind::MultiAssign { targets, .. } => {
                for target in &targets.targets {
                    match target {
                        MirOutputTarget::Place(place) => visit_place(place, visitor),
                        MirOutputTarget::Sequence(target) => {
                            visit_place(target.base(), visitor);
                            if let MirSequenceTarget::CellContents { indexing, .. } = target {
                                indexing.visit_direct_expression_regions_dyn(visitor);
                            }
                        }
                        MirOutputTarget::Discard => {}
                    }
                }
            }
            MirStmtKind::SequenceAssign { target, .. } => {
                visit_place(target.base(), visitor);
                if let MirSequenceTarget::CellContents { indexing, .. } = target {
                    indexing.visit_direct_expression_regions_dyn(visitor);
                }
            }
            MirStmtKind::Expr(_)
            | MirStmtKind::PlaceMutation(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }
}

impl MirStmt {
    pub(crate) fn visit_outer_operands_dyn(&self, visitor: &mut dyn FnMut(&crate::MirOperand)) {
        fn visit_place(place: &MirPlace, visitor: &mut dyn FnMut(&crate::MirOperand)) {
            match place {
                MirPlace::Local(_) | MirPlace::Binding(_) => {}
                MirPlace::Member(base, _) => visit_place(base, visitor),
                MirPlace::DynamicMember(base, member) => {
                    visit_place(base, visitor);
                    visitor(member);
                }
                MirPlace::Index(base, indexing) => {
                    visit_place(base, visitor);
                    indexing.visit_outer_operands_dyn(visitor);
                }
            }
        }
        match &self.kind {
            MirStmtKind::CaptureSequence { source, .. } => source.visit_outer_operands_dyn(visitor),
            MirStmtKind::Assign { place, value } => {
                visit_place(place, visitor);
                value.visit_outer_operands_dyn(visitor);
            }
            MirStmtKind::MultiAssign { targets, value } => {
                for target in &targets.targets {
                    match target {
                        MirOutputTarget::Place(place) => visit_place(place, visitor),
                        MirOutputTarget::Sequence(target) => {
                            visit_place(target.base(), visitor);
                            target.visit_outer_operands_dyn(visitor);
                        }
                        MirOutputTarget::Discard => {}
                    }
                }
                value.visit_outer_operands_dyn(visitor);
            }
            MirStmtKind::SequenceAssign { target, value } => {
                visit_place(target.base(), visitor);
                target.visit_outer_operands_dyn(visitor);
                value.visit_outer_operands_dyn(visitor);
            }
            MirStmtKind::Expr(value) => value.visit_outer_operands_dyn(visitor),
            MirStmtKind::PlaceMutation(mutation) => visit_place(&mutation.place, visitor),
            MirStmtKind::WorkspaceEffect { .. } | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirStmtKind {
    CaptureSequence {
        destination: MirSequenceLocalId,
        source: MirExpansionSource,
    },
    Assign {
        place: MirPlace,
        value: MirRvalue,
    },
    MultiAssign {
        targets: MirOutputTargetList,
        value: MirRvalue,
    },
    SequenceAssign {
        target: MirSequenceTarget,
        value: MirRvalue,
    },
    Expr(MirRvalue),
    PlaceMutation(MirPlaceMutation),
    WorkspaceEffect {
        effect: WorkspaceEffect,
        bindings: Vec<MirLocalId>,
    },
    EnvironmentEffect(EnvironmentEffect),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirPlaceMutation {
    pub place: MirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirOutputTargetList {
    pub targets: Vec<MirOutputTarget>,
    pub requested_outputs: RequestedOutputCount,
}

impl MirOutputTargetList {
    pub fn validate_fixed_arity(&self, context: &str) -> Result<usize, String> {
        if self
            .targets
            .iter()
            .any(|target| matches!(target, MirOutputTarget::Sequence(_)))
        {
            return Err(format!(
                "{context} contains runtime-cardinality sequence targets"
            ));
        }
        let expected = self.targets.len();
        let count = self
            .requested_outputs
            .known_count()
            .ok_or_else(|| format!("{context} requires a statically known output count"))?;
        if count != expected {
            return Err(format!(
                "{context} output target count mismatch: requested {count}, targets {expected}"
            ));
        }
        Ok(count)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirOutputTarget {
    Place(MirPlace),
    Sequence(MirSequenceTarget),
    Discard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirSequenceTarget {
    Member {
        base: MirPlace,
        member: runmat_hir::MemberName,
    },
    DynamicMember {
        base: MirPlace,
        member: crate::MirOperand,
    },
    CellContents {
        base: MirPlace,
        indexing: crate::MirIndexing,
    },
}

impl MirSequenceTarget {
    pub fn base(&self) -> &MirPlace {
        match self {
            Self::Member { base, .. }
            | Self::DynamicMember { base, .. }
            | Self::CellContents { base, .. } => base,
        }
    }

    pub fn base_mut(&mut self) -> &mut MirPlace {
        match self {
            Self::Member { base, .. }
            | Self::DynamicMember { base, .. }
            | Self::CellContents { base, .. } => base,
        }
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        match self {
            Self::Member { .. } => {}
            Self::DynamicMember { member, .. } => visitor(member),
            Self::CellContents { indexing, .. } => indexing.visit_operands(visitor),
        }
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut crate::MirOperand)) {
        match self {
            Self::Member { .. } => {}
            Self::DynamicMember { member, .. } => visitor(member),
            Self::CellContents { indexing, .. } => indexing.visit_operands_mut(visitor),
        }
    }

    pub(crate) fn visit_outer_operands_dyn(&self, visitor: &mut dyn FnMut(&crate::MirOperand)) {
        match self {
            Self::Member { .. } => {}
            Self::DynamicMember { member, .. } => visitor(member),
            Self::CellContents { indexing, .. } => indexing.visit_outer_operands_dyn(visitor),
        }
    }
}
