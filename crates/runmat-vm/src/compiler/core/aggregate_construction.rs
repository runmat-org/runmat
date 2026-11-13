//! Fixed matrix construction requires primitive scalar stack values, not
//! shape-one tensors. Index definitions once and prove temporaries iteratively.

use runmat_hir::OperatorKind;
use runmat_mir::{
    MirAggregateElement, MirBody, MirConstant, MirLocalId, MirLocalKind, MirOperand,
    MirOutputTarget, MirPlace, MirRvalue, MirStmtKind,
};
use runmat_value::Value;

#[derive(Clone, Copy)]
enum Definition {
    Missing,
    Unique { block: usize, statement: usize },
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proof {
    Unseen,
    Visiting,
    Primitive,
    NotPrimitive,
}

/// Shared by all aggregate expressions in one function body. Neither deep
/// temporary chains nor repeated operands rescan statements or recurse.
#[derive(Clone)]
pub(super) struct ProofIndex {
    definitions: Vec<Definition>,
    proofs: Vec<Proof>,
}

impl ProofIndex {
    pub(super) fn new(body: &MirBody) -> Self {
        let mut index = Self {
            definitions: vec![Definition::Missing; body.locals.len()],
            proofs: vec![Proof::Unseen; body.locals.len()],
        };
        for (block, basic_block) in body.blocks.iter().enumerate() {
            for (statement, item) in basic_block.statements.iter().enumerate() {
                match &item.kind {
                    MirStmtKind::Assign { place, .. } => {
                        if let Some(local) = place_root_local(place) {
                            if matches!(place, MirPlace::Local(_)) {
                                index.record_unique(local, block, statement);
                            } else {
                                index.invalidate(local);
                            }
                        }
                    }
                    MirStmtKind::MultiAssign { targets, .. } => {
                        for target in &targets.targets {
                            let local = match target {
                                MirOutputTarget::Place(place) => place_root_local(place),
                                MirOutputTarget::Sequence(target) => {
                                    place_root_local(target.base())
                                }
                                MirOutputTarget::Discard => None,
                            };
                            if let Some(local) = local {
                                index.invalidate(local);
                            }
                        }
                    }
                    MirStmtKind::SequenceAssign { target, .. } => {
                        if let Some(local) = place_root_local(target.base()) {
                            index.invalidate(local);
                        }
                    }
                    MirStmtKind::PlaceMutation(mutation) => {
                        if let Some(local) = place_root_local(&mutation.place) {
                            index.invalidate(local);
                        }
                    }
                    MirStmtKind::WorkspaceEffect { bindings, .. } => {
                        for local in bindings {
                            index.invalidate(*local);
                        }
                    }
                    MirStmtKind::PreparedAssign { preparation, .. } => {
                        if let Some(root) = preparation.mutation.place.root_local() {
                            index.invalidate(root);
                        }
                    }
                    MirStmtKind::PrepareFixedAssignment(_)
                    | MirStmtKind::CaptureSequence { .. }
                    | MirStmtKind::Expr(_)
                    | MirStmtKind::StatementOutput(_)
                    | MirStmtKind::EnvironmentEffect(_) => {}
                }
            }
        }
        index
    }

    pub(super) fn needs_dynamic_concat(
        &mut self,
        body: &MirBody,
        elements: &[MirAggregateElement],
    ) -> bool {
        elements.iter().any(|element| {
            element
                .operand()
                .is_none_or(|operand| !self.primitive_operand(body, operand))
        })
    }

    fn record_unique(&mut self, local: MirLocalId, block: usize, statement: usize) {
        if let Some(definition) = self.definitions.get_mut(local.0) {
            *definition = match definition {
                Definition::Missing => Definition::Unique { block, statement },
                Definition::Unique { .. } | Definition::Invalid => Definition::Invalid,
            };
        }
    }

    fn invalidate(&mut self, local: MirLocalId) {
        if let Some(definition) = self.definitions.get_mut(local.0) {
            *definition = Definition::Invalid;
        }
    }

    fn definition<'a>(&self, body: &'a MirBody, local: MirLocalId) -> Option<&'a MirRvalue> {
        if !body
            .locals
            .get(local.0)
            .is_some_and(|slot| slot.id == local && matches!(slot.kind, MirLocalKind::Temporary))
        {
            return None;
        }
        let Definition::Unique { block, statement } = *self.definitions.get(local.0)? else {
            return None;
        };
        match &body.blocks.get(block)?.statements.get(statement)?.kind {
            MirStmtKind::Assign {
                place: MirPlace::Local(candidate),
                value,
            } if *candidate == local => Some(value),
            _ => None,
        }
    }

    fn primitive_operand(&mut self, body: &MirBody, operand: &MirOperand) -> bool {
        match operand {
            MirOperand::Local(local) => self.prove_local(body, *local),
            other => primitive_constant(other),
        }
    }

    fn prove_local(&mut self, body: &MirBody, root: MirLocalId) -> bool {
        let mut work = vec![(root, false)];
        while let Some((local, finish)) = work.pop() {
            let Some(&proof) = self.proofs.get(local.0) else {
                continue;
            };
            if finish {
                if proof == Proof::Visiting {
                    let proven = self
                        .definition(body, local)
                        .and_then(rvalue_operands)
                        .is_some_and(|operands| {
                            operands
                                .into_iter()
                                .flatten()
                                .all(|operand| self.operand_is_complete(operand))
                        });
                    self.proofs[local.0] = if proven {
                        Proof::Primitive
                    } else {
                        Proof::NotPrimitive
                    };
                }
                continue;
            }
            if proof != Proof::Unseen {
                continue;
            }
            let Some(operands) = self.definition(body, local).and_then(rvalue_operands) else {
                self.proofs[local.0] = Proof::NotPrimitive;
                continue;
            };
            self.proofs[local.0] = Proof::Visiting;
            work.push((local, true));
            for operand in operands.into_iter().flatten().rev() {
                match operand {
                    MirOperand::Local(child) => match self.proofs.get(child.0) {
                        Some(Proof::Unseen) => work.push((*child, false)),
                        Some(Proof::Visiting) | None => {
                            self.proofs[local.0] = Proof::NotPrimitive;
                        }
                        Some(Proof::Primitive | Proof::NotPrimitive) => {}
                    },
                    other if !primitive_constant(other) => {
                        self.proofs[local.0] = Proof::NotPrimitive;
                    }
                    _ => {}
                }
            }
        }
        self.proofs.get(root.0) == Some(&Proof::Primitive)
    }

    fn operand_is_complete(&self, operand: &MirOperand) -> bool {
        match operand {
            MirOperand::Local(local) => self.proofs.get(local.0) == Some(&Proof::Primitive),
            other => primitive_constant(other),
        }
    }
}

fn rvalue_operands(value: &MirRvalue) -> Option<[Option<&MirOperand>; 2]> {
    match value {
        MirRvalue::Use(operand) => Some([Some(operand), None]),
        MirRvalue::Unary(
            OperatorKind::UnaryPlus | OperatorKind::UnaryMinus | OperatorKind::Not,
            operand,
        ) => Some([Some(operand), None]),
        MirRvalue::Binary(left, operator, right) if scalar_binary(*operator) => {
            Some([Some(left), Some(right)])
        }
        _ => None,
    }
}

fn primitive_constant(operand: &MirOperand) -> bool {
    match operand {
        MirOperand::Constant(
            MirConstant::Number(_)
            | MirConstant::ImaginaryNumber(_)
            | MirConstant::IntegerLiteral(_)
            | MirConstant::Bool(_),
        ) => true,
        MirOperand::Constant(MirConstant::Symbol(name)) => matches!(
            runmat_runtime::builtin::builtin_constant_value(&name.0),
            Some(Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _))
        ),
        MirOperand::Constant(MirConstant::String(_) | MirConstant::EmptyArray)
        | MirOperand::FunctionHandle(_)
        | MirOperand::Local(_) => false,
    }
}

fn scalar_binary(operator: OperatorKind) -> bool {
    matches!(
        operator,
        OperatorKind::Add
            | OperatorKind::Subtract
            | OperatorKind::MatrixMultiply
            | OperatorKind::ElementwiseMultiply
            | OperatorKind::MatrixPower
            | OperatorKind::ElementwisePower
            | OperatorKind::Mldivide
            | OperatorKind::Mrdivide
            | OperatorKind::ElementwiseDivide
            | OperatorKind::ElementwiseLeftDivide
            | OperatorKind::Equal
            | OperatorKind::NotEqual
            | OperatorKind::Less
            | OperatorKind::LessEqual
            | OperatorKind::Greater
            | OperatorKind::GreaterEqual
            | OperatorKind::ElementwiseAnd
            | OperatorKind::ElementwiseOr
    )
}

fn place_root_local(place: &MirPlace) -> Option<MirLocalId> {
    let mut current = place;
    loop {
        match current {
            MirPlace::Local(local) => return Some(*local),
            MirPlace::Member(base, _)
            | MirPlace::DynamicMember(base, _)
            | MirPlace::Index(base, _) => current = base,
            MirPlace::Binding(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::{FunctionAbi, FunctionId, RequestedOutputCount, Span};
    use runmat_mir::{
        BasicBlock, BasicBlockId, MirLocal, MirOutputTargetList, MirStmt, MirTerminator,
        MirTerminatorKind,
    };

    fn body(values: Vec<MirRvalue>) -> MirBody {
        let span = Span::default();
        let locals = (0..values.len())
            .map(|id| MirLocal {
                id: MirLocalId(id),
                binding: None,
                kind: MirLocalKind::Temporary,
                span,
            })
            .collect();
        let statements = values
            .into_iter()
            .enumerate()
            .map(|(id, value)| MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(id)),
                    value,
                },
                span,
            })
            .collect();
        MirBody {
            catch_scopes: Default::default(),
            function: FunctionId(0),
            abi: FunctionAbi {
                fixed_inputs: Vec::new(),
                varargin: None,
                fixed_outputs: Vec::new(),
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            locals,
            blocks: vec![BasicBlock {
                id: BasicBlockId(0),
                statements,
                terminator: MirTerminator {
                    kind: MirTerminatorKind::Return(Vec::new()),
                    span,
                },
            }],
        }
    }

    fn aggregate(local: usize, count: usize) -> Vec<MirAggregateElement> {
        vec![MirAggregateElement::Single(MirOperand::Local(MirLocalId(local))); count]
    }

    #[test]
    fn deep_chain_and_repeated_aggregate_operand_are_iterative_and_memoized() {
        let mut values = vec![MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
            "2".into(),
        )))];
        for id in 1..512 {
            values.push(MirRvalue::Binary(
                MirOperand::Local(MirLocalId(id - 1)),
                OperatorKind::Add,
                MirOperand::Constant(MirConstant::Number("1".into())),
            ));
        }
        let body = body(values);
        let mut index = ProofIndex::new(&body);
        assert!(!index.needs_dynamic_concat(&body, &aggregate(511, 512)));
        assert_eq!(index.proofs[511], Proof::Primitive);
        assert!(!index.needs_dynamic_concat(&body, &aggregate(511, 512)));
    }

    #[test]
    fn duplicate_pending_child_is_not_a_cycle() {
        let shared = body(vec![
            MirRvalue::Use(MirOperand::Constant(MirConstant::Number("2".into()))),
            MirRvalue::Binary(
                MirOperand::Local(MirLocalId(0)),
                OperatorKind::Add,
                MirOperand::Local(MirLocalId(0)),
            ),
        ]);
        let mut index = ProofIndex::new(&shared);
        assert!(!index.needs_dynamic_concat(&shared, &aggregate(1, 2)));
        assert_eq!(index.proofs[0], Proof::Primitive);
        assert_eq!(index.proofs[1], Proof::Primitive);
    }

    #[test]
    fn cycles_redefinitions_and_multioutput_destinations_are_not_primitive() {
        let cycle = body(vec![
            MirRvalue::Use(MirOperand::Local(MirLocalId(1))),
            MirRvalue::Use(MirOperand::Local(MirLocalId(0))),
        ]);
        assert!(ProofIndex::new(&cycle).needs_dynamic_concat(&cycle, &aggregate(0, 1)));

        let mut changed = body(vec![MirRvalue::Use(MirOperand::Constant(
            MirConstant::Number("1".into()),
        ))]);
        let span = Span::default();
        changed.blocks[0].statements.push(MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("2".into()))),
            },
            span,
        });
        assert!(ProofIndex::new(&changed).needs_dynamic_concat(&changed, &aggregate(0, 1)));

        changed.blocks[0].statements.pop();
        changed.blocks[0].statements.push(MirStmt {
            kind: MirStmtKind::MultiAssign {
                targets: MirOutputTargetList {
                    targets: vec![MirOutputTarget::Place(MirPlace::Local(MirLocalId(0)))],
                    requested_outputs: RequestedOutputCount::One,
                },
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("2".into()))),
            },
            span,
        });
        assert!(ProofIndex::new(&changed).needs_dynamic_concat(&changed, &aggregate(0, 1)));
    }
}
