use crate::{MirExpansionSource, MirLocalId, MirPlace, MirRvalue, MirSequenceLocalId};
use runmat_types::{
    AssignmentCreationPolicy, AssignmentShapePolicy, EnvironmentEffect, PlaceMutationKind,
    RequestedOutputCount, Span, WorkspaceEffect,
};
use serde::{Deserialize, Serialize};

mod fixed_preparation;
mod assignment_index_context;
pub use assignment_index_context::{MirAssignmentIndexContextError, MirAssignmentIndexRole};
pub use fixed_preparation::{
    MirFixedPreparation, MirFixedPreparationPhase, MirFixedPreparationState,
    MirFixedSelectorDestination,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirStmt {
    pub kind: MirStmtKind,
    pub span: Span,
}

impl MirStmt {
    pub fn visit_expression_regions(
        &self,
        mut visitor: impl FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        self.kind.visit_expression_regions_dyn(&mut visitor);
    }
}

impl MirStmtKind {
    /// Visits every operand in this statement, including owned selector and
    /// expression-preparation regions. Native continuation geometry uses the
    /// same canonical traversal as MIR's statement semantics.
    pub fn visit_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        crate::rvalue::visit_stmt_kind_operands(self, &mut visitor);
    }

    pub fn visit_expression_regions(
        &self,
        mut visitor: impl FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        self.visit_expression_regions_dyn(&mut visitor);
    }

    pub(crate) fn visit_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        self.visit_direct_expression_regions_dyn(&mut |region| {
            visitor(region);
            region.visit_nested_regions(visitor);
        });
    }

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        fn visit_place(
            place_value: &MirPlace,
            visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
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
            MirStmtKind::PrepareFixedAssignment(preparation) => {
                preparation.visit_direct_expression_regions_dyn(visitor);
            }
            MirStmtKind::PreparedAssign { value, .. } => {
                value.visit_direct_expression_regions_dyn(visitor)
            }
            MirStmtKind::CaptureSequence { source, .. } => {
                source.visit_direct_expression_regions_dyn(visitor)
            }
            MirStmtKind::StatementOutput(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
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
        visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        fn visit_place(
            place: &MirPlace,
            visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
        ) {
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
            MirStmtKind::PrepareFixedAssignment(preparation) => {
                preparation.visit_direct_expression_regions_dyn(visitor)
            }
            MirStmtKind::PreparedAssign { .. } => {}
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
            | MirStmtKind::StatementOutput(_)
            | MirStmtKind::PlaceMutation(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }

    pub(crate) fn visit_statement_expression_steps_dyn(
        &self,
        visitor: &mut dyn FnMut(&[crate::MirExpressionStep]),
    ) {
        match self {
            MirStmtKind::PrepareFixedAssignment(preparation) => {
                preparation.visit_preparation_steps_dyn(visitor);
            }
            MirStmtKind::CaptureSequence { source, .. } => {
                source.visit_direct_expression_steps_dyn(visitor)
            }
            _ => self.visit_statement_expression_regions_dyn(&mut |region| visitor(region.steps())),
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
            MirStmtKind::PrepareFixedAssignment(preparation) => {
                preparation.visit_external_operands(&mut *visitor)
            }
            MirStmtKind::PreparedAssign { value, .. } => value.visit_outer_operands_dyn(visitor),
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
            MirStmtKind::StatementOutput(_)
            | MirStmtKind::WorkspaceEffect { .. }
            | MirStmtKind::EnvironmentEffect(_) => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirStmtKind {
    PrepareFixedAssignment(MirFixedPreparation),
    /// Ordinary one-output assignment to an already retained destination.
    /// Selector/receiver preparation is owned only by the matching producer.
    PreparedAssign {
        preparation: MirPreparedAssignment,
        value: MirRvalue,
    },
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
    /// An ordered, source-level console effect after a successful statement.
    StatementOutput(MirStatementOutput),
    PlaceMutation(MirPlaceMutation),
    WorkspaceEffect {
        effect: WorkspaceEffect,
        bindings: Vec<MirLocalId>,
    },
    EnvironmentEffect(EnvironmentEffect),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirStatementOutput {
    pub source: MirStatementOutputSource,
    pub label: MirStatementOutputLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirStatementOutputSource {
    /// The immediately preceding expression statement's completed value.
    Expression,
    /// The complete binding after assignment, including indexed/member writes.
    Local(MirLocalId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirStatementOutputLabel {
    Ans,
    Binding(runmat_types::BindingId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirPlaceMutation {
    pub place: MirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
}

/// One source occurrence in this function, not a textual-place identity.
/// The invocation-owned prepared destination is matched to this entire typed
/// contract; its selectors are never evaluated by the consumer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirPreparedAssignment {
    pub source_statement: runmat_types::StmtId,
    pub mutation: MirPlaceMutation,
    pub steps: Vec<crate::MirExpressionStep>,
}

impl MirPreparedAssignment {
    pub fn validate(&self) -> Result<(), String> {
        self.validate_contract(false)
    }

    pub fn validate_phased(&self) -> Result<(), String> {
        if !self.steps.is_empty() {
            return Err("phased fixed assignment cannot retain eager target work".into());
        }
        self.validate_contract(true)
    }

    fn validate_contract(&self, phased: bool) -> Result<(), String> {
        use runmat_types::{IndexKind, PlaceMutationKind};
        self.validate_index_contexts()
            .map_err(|error| error.to_string())?;
        let place = &self.mutation.place;
        if !place.requires_prepared_assignment() {
            return Err(
                "prepared fixed assignment requires expanding selectors or an indexed member path"
                    .into(),
            );
        }
        let delete = self.mutation.kind == PlaceMutationKind::Delete;
        let expected = match place {
            MirPlace::Member(_, _) | MirPlace::DynamicMember(_, _) => {
                PlaceMutationKind::MemberAssign
            }
            MirPlace::Index(_, indexing) if indexing.kind == IndexKind::Brace => {
                PlaceMutationKind::CellAssign
            }
            MirPlace::Index(_, _) => {
                if delete {
                    PlaceMutationKind::Delete
                } else {
                    PlaceMutationKind::IndexedAssign
                }
            }
            _ => return Err("prepared fixed assignment requires a nonlocal endpoint".into()),
        };
        if self.mutation.kind != expected {
            return Err("prepared fixed assignment mutation disagrees with its endpoint".into());
        }
        let canonical = place.assignment_creation_policy(delete);
        if self.mutation.creation_policy != canonical
            && (delete || self.mutation.creation_policy != AssignmentCreationPolicy::ExistingOnly)
        {
            return Err(
                "prepared fixed assignment creation authority disagrees with its target".into(),
            );
        }
        fn validate_place(place: &MirPlace, phased: bool) -> Result<(), String> {
            match place {
                MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) => {
                    validate_place(base, phased)
                }
                MirPlace::Index(base, indexing) => {
                    validate_place(base, phased)?;
                    if phased {
                        for component in &indexing.components {
                            match component {
                                crate::MirIndexComponent::Colon
                                | crate::MirIndexComponent::Expr(crate::MirOperand::Local(_)) => {}
                                crate::MirIndexComponent::Expansion(region)
                                    if region.steps.is_empty()
                                        && matches!(
                                            region.argument,
                                            crate::MirCallArg::CapturedSequence(_)
                                        ) => {}
                                _ => return Err(
                                    "phased target descriptor contains executable selector work"
                                        .into(),
                                ),
                            }
                        }
                        return Ok(());
                    }
                    indexing
                        .validate_target_selectors()
                        .map_err(|error| error.to_string())
                }
                _ => Ok(()),
            }
        }
        validate_place(place, phased)?;
        crate::expression_region::validate_preparation_steps(&self.steps)?;
        if self.steps.iter().any(|step| matches!(step, crate::MirExpressionStep::Let { value, .. } if value.contains_unscoped_end())) {
            return Err("prepared fixed assignment contains end without its own indexing receiver".into());
        }
        Ok(())
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut crate::MirOperand)) {
        for step in &mut self.steps {
            match step {
                crate::MirExpressionStep::Let { value, .. } => {
                    value.visit_operands_mut(&mut visitor)
                }
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands_mut(&mut visitor)
                }
            }
        }
        crate::rvalue::visit_place_operands_mut(&mut self.mutation.place, &mut visitor);
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        for step in &self.steps {
            match step {
                crate::MirExpressionStep::Let { value, .. } => value.visit_operands(&mut visitor),
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_operands(&mut visitor)
                }
            }
        }
        crate::rvalue::visit_place_operands(&self.mutation.place, &mut visitor);
    }

    /// Private selector temporaries are not invocation-frame inputs.
    pub fn visit_external_operands(&self, mut visitor: impl FnMut(&crate::MirOperand)) {
        let mut defined = std::collections::BTreeSet::new();
        for step in &self.steps {
            if let crate::MirExpressionStep::Let { local, .. } = step {
                defined.insert(*local);
            }
        }
        self.visit_direct_expression_regions_dyn(&mut |region| {
            defined.extend(region.defined_locals());
            region.visit_nested_regions(&mut |nested| defined.extend(nested.defined_locals()));
        });
        self.visit_operands(|operand| {
            if !matches!(operand, crate::MirOperand::Local(local) if defined.contains(local)) {
                visitor(operand);
            }
        });
    }

    fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(crate::MirExpressionRegionRef<'_>),
    ) {
        for step in &self.steps {
            match step {
                crate::MirExpressionStep::Let { value, .. } => {
                    value.visit_direct_expression_regions_dyn(visitor)
                }
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_direct_expression_regions_dyn(visitor)
                }
            }
        }
        self.mutation.place.visit_selector_regions(visitor);
    }
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
        member: runmat_types::MemberName,
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

/// The root carrier permitted by the first indexed step of a sequence write.
/// This is only a creation recipe; the executor must separately prove that
/// the binding is absent before using it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SequenceRootCreation {
    Existing,
    EmptyCell,
    /// Ordinary fixed-RHS assignment retains its already-admitted original
    /// creation policy; this is not destination-cardinality output selection.
    Ordinary(runmat_types::AssignmentCreationPolicy),
    OrdinaryCell,
    OrdinaryContainer(runmat_types::AssignmentChildCreation),
}

impl SequenceRootCreation {
    pub fn retains_original_receiver(self) -> bool {
        matches!(
            self,
            Self::Ordinary(_) | Self::OrdinaryCell | Self::OrdinaryContainer(_)
        )
    }
}

impl MirSequenceTarget {
    pub fn root_creation(&self) -> SequenceRootCreation {
        match self.base().first_root_step() {
            Some(crate::MirRootStep::Index(runmat_types::IndexKind::Brace)) => {
                SequenceRootCreation::EmptyCell
            }
            None if matches!(self, Self::CellContents { .. }) => SequenceRootCreation::EmptyCell,
            _ => SequenceRootCreation::Existing,
        }
    }

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
