use crate::{MirCallArg, MirIndexing, MirOperand, MirStmt};
use runmat_hir::{
    CallSyntax, FunctionId, MemberName, OperatorKind, QualifiedName, RequestedOutputCount,
    SymbolName,
};
use runmat_types::ClassIdentity;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirRvalue {
    Use(MirOperand),
    Unary(OperatorKind, MirOperand),
    Binary(MirOperand, OperatorKind, MirOperand),
    ShortCircuit {
        left: MirOperand,
        op: MirShortCircuitOp,
        right_temps: Vec<MirStmt>,
        right: MirOperand,
    },
    Range {
        start: MirOperand,
        step: Option<MirOperand>,
        end: MirOperand,
    },
    Call(crate::MirCall),
    Aggregate {
        kind: MirAggregateKind,
        /// Number of syntactic elements in each source row. Realized widths
        /// are validated after comma-separated sequences and nested array
        /// concatenation expand.
        row_lengths: Vec<usize>,
        elements: Vec<MirAggregateElement>,
    },
    StructLiteral {
        fields: Vec<(MemberName, MirOperand)>,
    },
    ObjectLiteral {
        class_name: QualifiedName,
        fields: Vec<(MemberName, MirOperand)>,
    },
    Index {
        base: MirOperand,
        indexing: MirIndexing,
    },
    SubscriptChain(crate::MirSubscriptChain),
    Member {
        base: MirOperand,
        member: MemberName,
        sequence_use: runmat_types::SequenceUse,
    },
    DynamicMember {
        base: MirOperand,
        member: MirOperand,
        sequence_use: runmat_types::SequenceUse,
    },
    WorkspaceFirstStaticProperty {
        workspace_name: SymbolName,
        class_name: ClassIdentity,
        property: MemberName,
    },
    MetaClass(QualifiedName),
    Colon,
    End,
    Future {
        function: FunctionId,
        args: Vec<MirCallArg>,
        syntax: CallSyntax,
        requested_outputs: RequestedOutputCount,
    },
    Spawn(MirOperand),
    Distributed(crate::parallel::MirDistributedOp),
    Collective(crate::parallel::MirCollectiveOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MirShortCircuitOp {
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirAggregateKind {
    Tensor,
    Cell,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirAggregateElement {
    Single(MirOperand),
    CapturedSequence(crate::MirSequenceLocalId),
}

impl MirAggregateElement {
    pub fn operand(&self) -> Option<&MirOperand> {
        match self {
            Self::Single(operand) => Some(operand),
            Self::CapturedSequence(_) => None,
        }
    }

    pub fn operand_mut(&mut self) -> Option<&mut MirOperand> {
        match self {
            Self::Single(operand) => Some(operand),
            Self::CapturedSequence(_) => None,
        }
    }
}

impl MirRvalue {
    pub(crate) fn contains_unscoped_end(&self) -> bool {
        match self {
            MirRvalue::End => true,
            MirRvalue::ShortCircuit { right_temps, .. } => {
                right_temps.iter().any(|statement| match &statement.kind {
                    crate::MirStmtKind::Assign { value, .. }
                    | crate::MirStmtKind::MultiAssign { value, .. }
                    | crate::MirStmtKind::SequenceAssign { value, .. }
                    | crate::MirStmtKind::Expr(value) => value.contains_unscoped_end(),
                    _ => false,
                })
            }
            _ => false,
        }
    }

    pub fn contains_contextual_end(&self) -> bool {
        match self {
            MirRvalue::End => true,
            MirRvalue::ShortCircuit { right_temps, .. } => right_temps.iter().any(|statement| {
                match &statement.kind {
                    crate::MirStmtKind::Assign { value, .. }
                    | crate::MirStmtKind::MultiAssign { value, .. }
                    | crate::MirStmtKind::SequenceAssign { value, .. }
                    | crate::MirStmtKind::Expr(value) => value.contains_contextual_end(),
                    _ => false,
                }
            }),
            MirRvalue::Index { indexing, .. } => indexing.components.iter().any(|component| {
                matches!(component, crate::MirIndexComponent::ContextualExpr(region) if region.contains_contextual_end())
            }),
            MirRvalue::SubscriptChain(chain) => {
                let mut found = false;
                chain.visit_expression_regions(|region| found |= region.contains_contextual_end());
                found
            }
            MirRvalue::Call(call) => call.args.iter().any(MirCallArg::contains_contextual_end),
            MirRvalue::Future { args, .. } => {
                args.iter().any(MirCallArg::contains_contextual_end)
            }
            _ => false,
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

    pub(crate) fn visit_direct_expression_regions_dyn(
        &self,
        visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
    ) {
        match self {
            MirRvalue::Index { indexing, .. } => {
                indexing.visit_direct_expression_regions_dyn(visitor)
            }
            MirRvalue::SubscriptChain(chain) => chain.visit_direct_expression_regions_dyn(visitor),
            MirRvalue::Call(call) => {
                for argument in &call.args {
                    argument.visit_direct_expression_regions_dyn(visitor);
                }
            }
            MirRvalue::Future { args, .. } => {
                for argument in args {
                    argument.visit_direct_expression_regions_dyn(visitor);
                }
            }
            MirRvalue::ShortCircuit { right_temps, .. } => {
                for statement in right_temps {
                    statement.kind.visit_direct_expression_regions_dyn(visitor);
                }
            }
            _ => {}
        }
    }

    pub fn visit_operands(&self, mut visitor: impl FnMut(&MirOperand)) {
        visit_rvalue_operands(self, &mut visitor, true);
    }

    pub(crate) fn visit_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        visit_rvalue_operands(self, visitor, true);
    }

    pub(crate) fn visit_outer_operands_dyn(&self, visitor: &mut dyn FnMut(&MirOperand)) {
        visit_rvalue_operands(self, visitor, false);
    }

    pub fn visit_operands_mut(&mut self, mut visitor: impl FnMut(&mut MirOperand)) {
        visit_rvalue_operands_mut(self, &mut visitor);
    }

    pub(crate) fn visit_operands_mut_dyn(&mut self, visitor: &mut dyn FnMut(&mut MirOperand)) {
        visit_rvalue_operands_mut(self, visitor);
    }

    pub fn visit_sequence_locals(&self, mut visitor: impl FnMut(&crate::MirSequenceLocalId)) {
        visit_rvalue_sequences(self, &mut visitor);
    }
}

fn visit_rvalue_operands(
    value: &MirRvalue,
    visitor: &mut dyn FnMut(&MirOperand),
    descend_expression_regions: bool,
) {
    match value {
        MirRvalue::Use(value) | MirRvalue::Unary(_, value) | MirRvalue::Spawn(value) => {
            visitor(value)
        }
        MirRvalue::Binary(left, _, right) => {
            visitor(left);
            visitor(right);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            visitor(left);
            for statement in right_temps {
                visit_stmt_operands(statement, visitor, descend_expression_regions);
            }
            visitor(right);
        }
        MirRvalue::Range { start, step, end } => {
            visitor(start);
            if let Some(step) = step {
                visitor(step);
            }
            visitor(end);
        }
        MirRvalue::Call(call) => {
            if let crate::MirCallee::Dynamic(callee) = &call.callee {
                visitor(callee);
            }
            for argument in &call.args {
                argument.visit_operands(&mut *visitor);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(value) = element.operand() {
                    visitor(value);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                visitor(value);
            }
        }
        MirRvalue::Index { base, indexing } => {
            visitor(base);
            if descend_expression_regions {
                indexing.visit_operands(&mut *visitor);
            } else {
                indexing.visit_outer_operands_dyn(visitor);
            }
        }
        MirRvalue::SubscriptChain(chain) => {
            if descend_expression_regions {
                chain.visit_operands(&mut *visitor);
            } else {
                chain.visit_outer_operands(&mut *visitor);
            }
        }
        MirRvalue::Member { base, .. } => visitor(base),
        MirRvalue::DynamicMember { base, member, .. } => {
            visitor(base);
            visitor(member);
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                argument.visit_operands(&mut *visitor);
            }
        }
        MirRvalue::Distributed(operation) => {
            for operand in operation.operands() {
                visitor(operand);
            }
        }
        MirRvalue::Collective(operation) => operation.for_each_operand(&mut *visitor),
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => {}
    }
}

fn visit_rvalue_operands_mut(value: &mut MirRvalue, visitor: &mut dyn FnMut(&mut MirOperand)) {
    match value {
        MirRvalue::Use(value) | MirRvalue::Unary(_, value) | MirRvalue::Spawn(value) => {
            visitor(value)
        }
        MirRvalue::Binary(left, _, right) => {
            visitor(left);
            visitor(right);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            visitor(left);
            for statement in right_temps {
                visit_stmt_operands_mut(statement, visitor);
            }
            visitor(right);
        }
        MirRvalue::Range { start, step, end } => {
            visitor(start);
            if let Some(step) = step {
                visitor(step);
            }
            visitor(end);
        }
        MirRvalue::Call(call) => {
            if let crate::MirCallee::Dynamic(callee) = &mut call.callee {
                visitor(callee);
            }
            for argument in &mut call.args {
                argument.visit_operands_mut(&mut *visitor);
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(value) = element.operand_mut() {
                    visitor(value);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                visitor(value);
            }
        }
        MirRvalue::Index { base, indexing } => {
            visitor(base);
            indexing.visit_operands_mut(&mut *visitor);
        }
        MirRvalue::SubscriptChain(chain) => chain.visit_operands_mut(&mut *visitor),
        MirRvalue::Member { base, .. } => visitor(base),
        MirRvalue::DynamicMember { base, member, .. } => {
            visitor(base);
            visitor(member);
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                argument.visit_operands_mut(&mut *visitor);
            }
        }
        MirRvalue::Distributed(operation) => operation.for_each_operand_mut(&mut *visitor),
        MirRvalue::Collective(operation) => operation.for_each_operand_mut(&mut *visitor),
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => {}
    }
}

fn visit_stmt_operands(
    statement: &MirStmt,
    visitor: &mut dyn FnMut(&MirOperand),
    descend_expression_regions: bool,
) {
    match &statement.kind {
        crate::MirStmtKind::CaptureSequence { source, .. } => source.visit_operands(&mut *visitor),
        crate::MirStmtKind::Assign { place, value } => {
            visit_place_operands(place, visitor);
            visit_rvalue_operands(value, visitor, descend_expression_regions);
        }
        crate::MirStmtKind::MultiAssign { targets, value } => {
            for target in &targets.targets {
                match target {
                    crate::MirOutputTarget::Place(place) => visit_place_operands(place, visitor),
                    crate::MirOutputTarget::Sequence(target) => {
                        visit_place_operands(target.base(), visitor);
                        target.visit_operands(&mut *visitor);
                    }
                    crate::MirOutputTarget::Discard => {}
                }
            }
            visit_rvalue_operands(value, visitor, descend_expression_regions);
        }
        crate::MirStmtKind::SequenceAssign { target, value } => {
            visit_place_operands(target.base(), visitor);
            target.visit_operands(&mut *visitor);
            visit_rvalue_operands(value, visitor, descend_expression_regions);
        }
        crate::MirStmtKind::Expr(value) => {
            visit_rvalue_operands(value, visitor, descend_expression_regions)
        }
        crate::MirStmtKind::PlaceMutation(mutation) => {
            visit_place_operands(&mutation.place, visitor)
        }
        crate::MirStmtKind::WorkspaceEffect { .. } | crate::MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn visit_stmt_operands_mut(statement: &mut MirStmt, visitor: &mut dyn FnMut(&mut MirOperand)) {
    match &mut statement.kind {
        crate::MirStmtKind::CaptureSequence { source, .. } => {
            source.visit_operands_mut(&mut *visitor)
        }
        crate::MirStmtKind::Assign { place, value } => {
            visit_place_operands_mut(place, visitor);
            visit_rvalue_operands_mut(value, visitor);
        }
        crate::MirStmtKind::MultiAssign { targets, value } => {
            for target in &mut targets.targets {
                match target {
                    crate::MirOutputTarget::Place(place) => {
                        visit_place_operands_mut(place, visitor)
                    }
                    crate::MirOutputTarget::Sequence(target) => {
                        visit_place_operands_mut(target.base_mut(), visitor);
                        target.visit_operands_mut(&mut *visitor);
                    }
                    crate::MirOutputTarget::Discard => {}
                }
            }
            visit_rvalue_operands_mut(value, visitor);
        }
        crate::MirStmtKind::SequenceAssign { target, value } => {
            visit_place_operands_mut(target.base_mut(), visitor);
            target.visit_operands_mut(&mut *visitor);
            visit_rvalue_operands_mut(value, visitor);
        }
        crate::MirStmtKind::Expr(value) => visit_rvalue_operands_mut(value, visitor),
        crate::MirStmtKind::PlaceMutation(mutation) => {
            visit_place_operands_mut(&mut mutation.place, visitor)
        }
        crate::MirStmtKind::WorkspaceEffect { .. } | crate::MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn visit_place_operands(place: &crate::MirPlace, visitor: &mut dyn FnMut(&MirOperand)) {
    match place {
        crate::MirPlace::Local(_) | crate::MirPlace::Binding(_) => {}
        crate::MirPlace::Member(base, _) => visit_place_operands(base, visitor),
        crate::MirPlace::DynamicMember(base, member) => {
            visit_place_operands(base, visitor);
            visitor(member);
        }
        crate::MirPlace::Index(base, indexing) => {
            visit_place_operands(base, visitor);
            indexing.visit_operands(&mut *visitor);
        }
    }
}

fn visit_place_operands_mut(place: &mut crate::MirPlace, visitor: &mut dyn FnMut(&mut MirOperand)) {
    match place {
        crate::MirPlace::Local(_) | crate::MirPlace::Binding(_) => {}
        crate::MirPlace::Member(base, _) => visit_place_operands_mut(base, visitor),
        crate::MirPlace::DynamicMember(base, member) => {
            visit_place_operands_mut(base, visitor);
            visitor(member);
        }
        crate::MirPlace::Index(base, indexing) => {
            visit_place_operands_mut(base, visitor);
            indexing.visit_operands_mut(&mut *visitor);
        }
    }
}

fn visit_rvalue_sequences(value: &MirRvalue, visitor: &mut impl FnMut(&crate::MirSequenceLocalId)) {
    match value {
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let MirAggregateElement::CapturedSequence(sequence) = element {
                    visitor(sequence);
                }
            }
        }
        MirRvalue::Call(call) => {
            for argument in &call.args {
                if let MirCallArg::CapturedSequence(sequence) = argument {
                    visitor(sequence);
                }
            }
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                if let MirCallArg::CapturedSequence(sequence) = argument {
                    visitor(sequence);
                }
            }
        }
        MirRvalue::ShortCircuit { right_temps, .. } => {
            for statement in right_temps {
                visit_stmt_sequences(statement, visitor);
            }
        }
        _ => {}
    }
}

fn visit_stmt_sequences(statement: &MirStmt, visitor: &mut impl FnMut(&crate::MirSequenceLocalId)) {
    match &statement.kind {
        crate::MirStmtKind::Assign { value, .. }
        | crate::MirStmtKind::Expr(value)
        | crate::MirStmtKind::MultiAssign { value, .. }
        | crate::MirStmtKind::SequenceAssign { value, .. } => {
            visit_rvalue_sequences(value, visitor)
        }
        _ => {}
    }
}
