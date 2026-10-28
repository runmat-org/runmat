use super::*;
use crate::{MirStmt, MirStmtKind};

#[path = "body/captures.rs"]
mod captures;
#[path = "body/mutation.rs"]
mod mutation;

fn body_with_region(region: MirExpressionRegion) -> crate::MirBody {
    use crate::{
        BasicBlock, BasicBlockId, MirIndexPlan, MirIndexing, MirLocal, MirLocalKind, MirPlace,
        MirTerminator, MirTerminatorKind,
    };
    use runmat_hir::{FunctionAbi, FunctionId, IndexKind, IndexResultContext};
    let span = Span::default();
    crate::MirBody {
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
        locals: (0..2)
            .map(|id| MirLocal {
                id: MirLocalId(id),
                binding: None,
                kind: MirLocalKind::Temporary,
                span,
            })
            .collect(),
        blocks: vec![BasicBlock {
            id: BasicBlockId(0),
            statements: vec![MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Index {
                        base: MirOperand::Constant(crate::MirConstant::EmptyArray),
                        indexing: MirIndexing {
                            argument_spans: None,
                            kind: IndexKind::Paren,
                            plan: MirIndexPlan::Slice,
                            components: vec![crate::MirIndexComponent::ContextualExpr(region)],
                            result_context: IndexResultContext::ReadSingle,
                            cell_expand_all: false,
                        },
                    },
                },
                span,
            }],
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                span,
            },
        }],
    }
}

#[test]
fn body_admission_confines_contextual_temporaries() {
    let region = MirExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let body = body_with_region(region.clone());
    body.validate_expression_regions().unwrap();

    let mut wrong_kind = body.clone();
    wrong_kind.locals[1].kind = crate::MirLocalKind::Binding;
    assert!(wrong_kind
        .validate_expression_regions()
        .unwrap_err()
        .contains("must be a temporary"));

    let mut escaped = body;
    escaped.blocks[0].terminator.kind =
        crate::MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]);
    assert!(escaped
        .validate_expression_regions()
        .unwrap_err()
        .contains("used outside its region"));

    let context_free = MirExpressionRegion {
        steps: vec![let_step(
            1,
            MirRvalue::Use(MirOperand::Constant(crate::MirConstant::Bool(true))),
        )],
        result: MirOperand::Local(MirLocalId(1)),
    };
    assert!(context_free
        .validate()
        .unwrap_err()
        .contains("does not depend on the indexing context"));
}

#[test]
fn body_admission_confines_contextual_sequences() {
    let sequence = MirSequenceLocalId(7);
    let region = MirExpressionRegion {
        steps: vec![
            let_step(1, MirRvalue::End),
            MirExpressionStep::CaptureSequence {
                destination: sequence,
                source: crate::MirExpansionSource::ReturnedOutputs(MirOperand::Local(MirLocalId(
                    1,
                ))),
                span: Span::default(),
            },
            let_step(
                2,
                MirRvalue::Aggregate {
                    kind: MirAggregateKind::Tensor,
                    row_lengths: vec![1],
                    elements: vec![MirAggregateElement::CapturedSequence(sequence)],
                },
            ),
        ],
        result: MirOperand::Local(MirLocalId(2)),
    };
    let mut body = body_with_region(region);
    body.locals.push(crate::MirLocal {
        id: MirLocalId(2),
        binding: None,
        kind: crate::MirLocalKind::Temporary,
        span: Span::default(),
    });
    body.validate_expression_regions().unwrap();

    body.blocks[0].statements.push(MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Aggregate {
            kind: MirAggregateKind::Cell,
            row_lengths: vec![1],
            elements: vec![MirAggregateElement::CapturedSequence(sequence)],
        }),
        span: Span::default(),
    });
    assert!(body
        .validate_expression_regions()
        .unwrap_err()
        .contains("expected its single use inside the defining region"));
}
