use super::*;

pub(super) fn nested_argument_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 12 };
    let selector = runmat_mir::MirExpressionRegion::from_lowered(
        vec![
            statement(1, nested_call(8, 2, span), span),
            statement(8, MirRvalue::End, span),
            statement(
                3,
                contextual_call(9, vec![MirOperand::Local(MirLocalId(8))], span),
                span,
            ),
        ],
        MirOperand::Local(MirLocalId(3)),
    )
    .unwrap();
    let MirRvalue::Call(mut outer) = contextual_call(11, vec![], span) else {
        unreachable!()
    };
    outer.args = vec![MirCallArg::Expansion(
        runmat_mir::MirExpansionSource::CellContents {
            base: MirOperand::Local(MirLocalId(0)),
            indexing: MirIndexing {
                argument_spans: None,
                kind: runmat_types::IndexKind::Brace,
                plan: MirIndexPlan::Cell,
                components: vec![MirIndexComponent::ContextualExpr(selector)],
                result_context: runmat_types::IndexResultContext::ReadCommaList,
                cell_expand_all: false,
            },
        },
    )];
    outer.arg_spans = vec![span];
    lower_body(
        "short_nested_argument",
        9,
        vec![
            statement(
                0,
                MirRvalue::Aggregate {
                    kind: MirAggregateKind::Cell,
                    row_lengths: vec![1],
                    elements: vec![MirAggregateElement::Single(boolean(true))],
                },
                span,
            ),
            statement(
                7,
                MirRvalue::ShortCircuit {
                    left: boolean(true),
                    op: runmat_mir::MirShortCircuitOp::And,
                    right_temps: vec![
                        statement(4, MirRvalue::Call(outer), span),
                        statement(5, nested_call(10, 6, span), span),
                    ],
                    right: MirOperand::Local(MirLocalId(5)),
                },
                span,
            ),
        ],
        MirOperand::Local(MirLocalId(7)),
        span,
    )
}

pub(super) fn caught_capture_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 5 };
    let function = FunctionId(0);
    let capture = MirStmt {
        kind: MirStmtKind::CaptureSequence {
            destination: runmat_mir::MirSequenceLocalId(0),
            source: runmat_mir::MirExpansionSource::CellContents {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    argument_spans: None,
                    kind: runmat_types::IndexKind::Brace,
                    plan: MirIndexPlan::Cell,
                    components: vec![MirIndexComponent::Colon],
                    result_context: runmat_types::IndexResultContext::ReadCommaList,
                    cell_expand_all: true,
                },
            },
        },
        span,
    };
    let MirRvalue::Call(mut consume) = contextual_call(10, vec![], span) else {
        unreachable!()
    };
    consume.args = vec![MirCallArg::CapturedSequence(
        runmat_mir::MirSequenceLocalId(0),
    )];
    consume.arg_spans = vec![span];
    let failure = MirRvalue::ShortCircuit {
        left: boolean(true),
        op: runmat_mir::MirShortCircuitOp::And,
        right_temps: vec![
            capture.clone(),
            statement(2, contextual_call(9, vec![], span), span),
            // The normal path consumes the capture; the injected failure above
            // leaves it live for exceptional cleanup instead.
            statement(2, MirRvalue::Call(consume.clone()), span),
        ],
        right: MirOperand::Local(MirLocalId(2)),
    };
    let recovery = MirRvalue::ShortCircuit {
        left: boolean(true),
        op: runmat_mir::MirShortCircuitOp::And,
        right_temps: vec![capture, statement(2, MirRvalue::Call(consume), span)],
        right: MirOperand::Local(MirLocalId(2)),
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                catch_scopes: [(BasicBlockId(2), vec![runmat_mir::CatchScopeId(0)])]
                    .into_iter()
                    .collect(),
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: (0..5)
                    .map(|id| MirLocal {
                        id: MirLocalId(id),
                        binding: None,
                        kind: MirLocalKind::Temporary,
                        span,
                    })
                    .collect(),
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: vec![statement(
                            0,
                            MirRvalue::Aggregate {
                                kind: MirAggregateKind::Cell,
                                row_lengths: vec![2],
                                elements: vec![
                                    MirAggregateElement::Single(boolean(true)),
                                    MirAggregateElement::Single(boolean(false)),
                                ],
                            },
                            span,
                        )],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::TryCatch {
                                try_block: BasicBlockId(1),
                                catch_block: BasicBlockId(2),
                                catch_binding: None,
                            },
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(1),
                        statements: vec![statement(4, failure, span)],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(3)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(2),
                        statements: vec![statement(4, recovery, span)],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(3)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(3),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(4))]),
                            span,
                        },
                    },
                ],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("caught_short_capture".into()),
                parent: None,
                enclosing_class: None,
                class_method_owner: None,
                kind: FunctionKind::SyntheticEntrypoint,
                source_role: runmat_types::FunctionSourceRole::Synthetic,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        class_property_defaults: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        entrypoint: runmat_native_codegen::NativeEntrypointBinding::new(
            &mir,
            *mir.entrypoints.first().expect("fixture entrypoint"),
            manifest.identity.entrypoint_function.unwrap(),
        )
        .unwrap(),
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}
