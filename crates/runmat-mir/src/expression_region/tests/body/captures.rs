use super::*;

#[test]
fn nested_call_expansion_keeps_selector_locals_private_but_rejects_real_escape() {
    use runmat_hir::{CallSyntax, CallableFallbackPolicy, RequestedOutputCount};

    let region = MirExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let mut body = body_with_region(region);
    let MirStmtKind::Assign { value, .. } = &mut body.blocks[0].statements[0].kind else {
        unreachable!()
    };
    let MirRvalue::Index { base, indexing } = value.clone() else {
        unreachable!()
    };
    let argument =
        crate::MirCallArg::Expansion(crate::MirExpansionSource::CellContents { base, indexing });
    let call = crate::MirCall {
        callee: crate::MirCallee::Dynamic(MirOperand::Constant(crate::MirConstant::Bool(true))),
        args: vec![argument.clone()],
        arg_spans: vec![Span::default()],
        syntax: CallSyntax::Command,
        requested_outputs: RequestedOutputCount::Exactly(1),
        fallback_policy: CallableFallbackPolicy::None,
        workspace_first_name: None,
        bare_identifier: false,
        async_behavior: crate::AsyncBehaviorFact::MaySuspend,
        effects: runmat_builtins::BuiltinEffects::none(),
        workspace_effect: None,
        environment_effect: None,
        purity: runmat_builtins::BuiltinPurity::Impure,
        semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
    };
    for nested in [
        MirRvalue::Call(call),
        MirRvalue::Future {
            function: runmat_hir::FunctionId(9),
            args: vec![argument],
            syntax: CallSyntax::Command,
            requested_outputs: RequestedOutputCount::Exactly(1),
        },
    ] {
        let mut candidate = body.clone();
        let MirStmtKind::Assign { value, .. } = &mut candidate.blocks[0].statements[0].kind else {
            unreachable!()
        };
        *value = MirRvalue::ShortCircuit {
            op: crate::MirShortCircuitOp::And,
            left: MirOperand::Constant(crate::MirConstant::Bool(true)),
            right_temps: vec![MirStmt {
                kind: MirStmtKind::Expr(nested),
                span: Span::default(),
            }],
            right: MirOperand::Constant(crate::MirConstant::Bool(true)),
        };
        // Full traversal still sees the selector result; only outer traversal
        // omits it. Validation must not hide a genuinely escaping argument.
        let mut full_locals = Vec::new();
        value.visit_operands(|operand| {
            if let MirOperand::Local(local) = operand {
                full_locals.push(*local);
            }
        });
        assert!(full_locals.contains(&MirLocalId(1)));
        candidate.validate_expression_regions().unwrap();

        let MirStmtKind::Assign { value, .. } = &mut candidate.blocks[0].statements[0].kind else {
            unreachable!()
        };
        let MirRvalue::ShortCircuit { right_temps, .. } = value else {
            unreachable!()
        };
        let MirStmtKind::Expr(nested) = &mut right_temps[0].kind else {
            unreachable!()
        };
        let args = match nested {
            MirRvalue::Call(call) => {
                call.arg_spans.push(Span::default());
                &mut call.args
            }
            MirRvalue::Future { args, .. } => args,
            _ => unreachable!(),
        };
        args.push(crate::MirCallArg::Single(MirOperand::Local(MirLocalId(1))));
        assert!(candidate
            .validate_expression_regions()
            .unwrap_err()
            .contains("used outside its region"));
    }
}

#[test]
fn short_circuit_capture_outer_traversal_excludes_contextual_selector_locals() {
    let value = MirRvalue::ShortCircuit {
        op: crate::MirShortCircuitOp::And,
        left: MirOperand::Constant(crate::MirConstant::Bool(true)),
        right_temps: vec![MirStmt {
            kind: MirStmtKind::CaptureSequence {
                destination: MirSequenceLocalId(0),
                source: crate::MirExpansionSource::CellContents {
                    base: MirOperand::Local(MirLocalId(0)),
                    indexing: crate::MirIndexing {
                        argument_spans: None,
                        kind: runmat_hir::IndexKind::Brace,
                        plan: crate::MirIndexPlan::Slice,
                        components: vec![crate::MirIndexComponent::ContextualExpr(
                            MirExpressionRegion {
                                steps: vec![let_step(1, MirRvalue::End)],
                                result: MirOperand::Local(MirLocalId(1)),
                            },
                        )],
                        result_context: runmat_hir::IndexResultContext::ReadSingle,
                        cell_expand_all: false,
                    },
                },
            },
            span: Span::default(),
        }],
        right: MirOperand::Constant(crate::MirConstant::Bool(true)),
    };
    let mut outer = Vec::new();
    value.visit_outer_operands_dyn(&mut |operand| {
        if let MirOperand::Local(local) = operand {
            outer.push(*local);
        }
    });
    assert_eq!(outer, vec![MirLocalId(0)]);
    let mut full = Vec::new();
    value.visit_operands(|operand| {
        if let MirOperand::Local(local) = operand {
            full.push(*local);
        }
    });
    assert_eq!(full, vec![MirLocalId(0), MirLocalId(1)]);
}
