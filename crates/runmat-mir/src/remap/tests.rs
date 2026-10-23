use super::*;

use runmat_types::{FunctionAbi, MemberName, Span};
use runmat_types::{ObjectIndexingContext, SequenceUse};

fn external(function: FunctionId) -> CallableIdentity {
    CallableIdentity::ExternalFunction {
        function,
        display_name: "helper".into(),
    }
}

fn assembly() -> MirAssembly {
    let source = FunctionId(70);
    let span = Span { start: 0, end: 1 };
    let body = crate::MirBody {
        function: source,
        abi: FunctionAbi {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        locals: vec![],
        blocks: vec![crate::BasicBlock {
            id: crate::BasicBlockId(0),
            statements: vec![crate::MirStmt {
                kind: MirStmtKind::Expr(MirRvalue::SubscriptChain(crate::MirSubscriptChain {
                    root: MirOperand::FunctionHandle(external(source)),
                    steps: vec![crate::MirSubscriptStep::DottedInvoke {
                        member: MemberName("helper".into()),
                        arguments: vec![],
                        semantic_fallback: Some(std::sync::Arc::new(external(source))),
                    }],
                    sequence_use: SequenceUse::RequireSingle,
                    context: ObjectIndexingContext::Expression,
                })),
                span,
            }],
            terminator: crate::MirTerminator {
                kind: MirTerminatorKind::Return(vec![MirOperand::FunctionHandle(external(source))]),
                span,
            },
        }],
        catch_scopes: BTreeMap::new(),
    };
    MirAssembly {
        bodies: BTreeMap::from([(source, body)]),
        ..Default::default()
    }
}

fn references(assembly: &MirAssembly, owner: FunctionId) -> [FunctionId; 3] {
    let body = &assembly.bodies[&owner];
    let MirStmtKind::Expr(MirRvalue::SubscriptChain(chain)) = &body.blocks[0].statements[0].kind
    else {
        panic!("dotted chain");
    };
    let MirOperand::FunctionHandle(CallableIdentity::ExternalFunction { function: root, .. }) =
        &chain.root
    else {
        panic!("external root");
    };
    let crate::MirSubscriptStep::DottedInvoke {
        semantic_fallback: Some(fallback),
        ..
    } = &chain.steps[0]
    else {
        panic!("external dotted fallback");
    };
    let CallableIdentity::ExternalFunction {
        function: fallback, ..
    } = fallback.as_ref()
    else {
        panic!("external dotted fallback identity");
    };
    let MirTerminatorKind::Return(values) = &body.blocks[0].terminator.kind else {
        panic!("return");
    };
    let MirOperand::FunctionHandle(CallableIdentity::ExternalFunction {
        function: returned, ..
    }) = &values[0]
    else {
        panic!("external return");
    };
    [*root, *fallback, *returned]
}

#[test]
fn eval_local_keeps_parent_external_refs_but_closed_unit_rebinds_all_edges() {
    let ids = HashMap::from([(FunctionId(70), FunctionId(7))]);
    let mut local = assembly();
    remap_function_ids(&mut local, &ids).unwrap();
    assert_eq!(references(&local, FunctionId(7)), [FunctionId(70); 3]);

    let mut closed = assembly();
    remap_closed_function_ids(&mut closed, &ids).unwrap();
    assert_eq!(references(&closed, FunctionId(7)), [FunctionId(7); 3]);
}
