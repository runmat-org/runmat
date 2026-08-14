#![cfg(not(target_arch = "wasm32"))]

use std::rc::Rc;

use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitManifest, ProgramEnvironment, ProgramRevision, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_hir::{FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, Span};
use runmat_jit::{GenericExecutor, JitError};
use runmat_mir::{
    AsyncBehaviorFact, BasicBlock, BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall,
    MirCallArg, MirCallee, MirConstant, MirFunctionMetadata, MirLocal, MirLocalId, MirLocalKind,
    MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminator, MirTerminatorKind,
};
use runmat_native_codegen::{lower_executable, NativeLoweringInput, NativeTarget};
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use runmat_types::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, CapabilitySet, InteropManifest,
    ParallelManifest, ProgramFunctionId, ProgramSourceId, RequestedOutputCount,
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};
use runmat_value::Value;

#[test]
fn forced_generic_entry_executes_literal_assignment_and_transactional_return() {
    let executor = GenericExecutor::compile(fixture()).unwrap();
    let execution = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap();
    assert_eq!(execution.outputs, vec![Value::Num(41.0)]);
}

#[test]
fn cancellation_exits_without_committing_speculative_outputs() {
    let executor = GenericExecutor::compile(fixture()).unwrap();
    let runtime = runtime_context();
    runtime
        .cancellation()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(matches!(
        executor.invoke(ProgramFunctionId(0), Vec::new(), 1, runtime),
        Err(JitError::Cancelled)
    ));
}

#[test]
fn generated_branch_uses_shared_truth_semantics_and_exact_edge_values() {
    let executor = GenericExecutor::compile(branch_fixture()).unwrap();
    let execution = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap();
    assert_eq!(execution.outputs, vec![Value::Num(7.0)]);
}

#[test]
fn generic_operator_and_range_sites_use_canonical_runtime_builtins() {
    let operator = GenericExecutor::compile(binary_fixture()).unwrap();
    assert_eq!(
        operator
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );

    let range = GenericExecutor::compile(range_fixture()).unwrap();
    let output = range
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap();
    let Value::Tensor(range) = output else {
        panic!("expected colon to produce a tensor")
    };
    assert_eq!(range.materialize_f64(), vec![1.0, 2.0, 3.0]);

    let call = GenericExecutor::compile(call_fixture()).unwrap();
    assert_eq!(
        call.invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(9.0)]
    );

    let aggregate = GenericExecutor::compile(aggregate_fixture()).unwrap();
    let Value::Tensor(matrix) = aggregate
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected a tensor aggregate")
    };
    assert_eq!(matrix.shape, vec![1, 2]);
    assert_eq!(matrix.materialize_f64(), vec![1.0, 2.0]);
}

#[test]
fn generic_short_circuit_and_switch_preserve_compiled_control_flow() {
    let short = GenericExecutor::compile(short_circuit_fixture()).unwrap();
    assert_eq!(
        short
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Bool(true)]
    );

    let switch = GenericExecutor::compile(switch_fixture()).unwrap();
    assert_eq!(
        switch
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(20.0)]
    );
}

#[test]
fn generic_for_iterates_captured_columns_through_compiled_backedges() {
    let executor = GenericExecutor::compile(for_fixture()).unwrap();
    assert_eq!(
        executor
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(6.0)]
    );
}

fn runtime_context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 2 };
    let function = FunctionId(0);
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![MirLocal {
                    id: MirLocalId(0),
                    binding: None,
                    kind: MirLocalKind::Temporary,
                    span,
                }],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![MirStmt {
                        kind: MirStmtKind::Assign {
                            place: MirPlace::Local(MirLocalId(0)),
                            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                "41".into(),
                            ))),
                        },
                        span,
                    }],
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("main".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn branch_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let local = |id| MirLocal {
        id: MirLocalId(id),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    };
    let assign = |local, number: &str| MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(number.into()))),
        },
        span,
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![local(0), local(1)],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: vec![assign(0, "1")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Branch {
                                cond: MirOperand::Local(MirLocalId(0)),
                                then_block: BasicBlockId(1),
                                else_block: BasicBlockId(2),
                            },
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(1),
                        statements: vec![assign(1, "7")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(2),
                        statements: vec![assign(1, "9")],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
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
                name: FunctionName("branch".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn binary_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let statement = |local, value| MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value,
        },
        span,
    };
    lower_body(
        "binary",
        3,
        vec![
            statement(
                0,
                MirRvalue::Use(MirOperand::Constant(MirConstant::Number("40".into()))),
            ),
            statement(
                1,
                MirRvalue::Use(MirOperand::Constant(MirConstant::Number("2".into()))),
            ),
            statement(
                2,
                MirRvalue::Binary(
                    MirOperand::Local(MirLocalId(0)),
                    runmat_types::OperatorKind::Add,
                    MirOperand::Local(MirLocalId(1)),
                ),
            ),
        ],
        MirOperand::Local(MirLocalId(2)),
        span,
    )
}

fn range_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "range",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Range {
                    start: MirOperand::Constant(MirConstant::Number("1".into())),
                    step: None,
                    end: MirOperand::Constant(MirConstant::Number("3".into())),
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "call",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::Static(CallableIdentity::Builtin(BuiltinId("sqrt".into()))),
                    args: vec![MirCallArg::Single(MirOperand::Constant(
                        MirConstant::Number("81".into()),
                    ))],
                    arg_spans: vec![span],
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: AsyncBehaviorFact::NeverSuspends,
                    effects: runmat_builtins::BuiltinEffects::none(),
                    workspace_effect: None,
                    environment_effect: None,
                    purity: runmat_builtins::BuiltinPurity::Pure,
                    semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn aggregate_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "aggregate",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Aggregate {
                    kind: MirAggregateKind::Tensor,
                    rows: 1,
                    cols: 2,
                    elements: vec![
                        MirOperand::Constant(MirConstant::Number("1".into())),
                        MirOperand::Constant(MirConstant::Number("2".into())),
                    ],
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn short_circuit_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "short",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::ShortCircuit {
                    left: MirOperand::Constant(MirConstant::Bool(true)),
                    op: runmat_mir::MirShortCircuitOp::Or,
                    right_temps: Vec::new(),
                    right: MirOperand::Constant(MirConstant::Bool(false)),
                },
            },
            span,
        }],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn switch_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let function = FunctionId(0);
    let return_block = |id, number: &str| BasicBlock {
        id: BasicBlockId(id),
        statements: vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(number.into()))),
            },
            span,
        }],
        terminator: MirTerminator {
            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
            span,
        },
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![MirLocal {
                    id: MirLocalId(0),
                    binding: None,
                    kind: MirLocalKind::Temporary,
                    span,
                }],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Switch {
                                discr: MirOperand::Constant(MirConstant::Number("2".into())),
                                cases: vec![
                                    (
                                        MirOperand::Constant(MirConstant::Number("1".into())),
                                        BasicBlockId(1),
                                    ),
                                    (
                                        MirOperand::Constant(MirConstant::Number("2".into())),
                                        BasicBlockId(2),
                                    ),
                                ],
                                otherwise: BasicBlockId(3),
                            },
                            span,
                        },
                    },
                    return_block(1, "10"),
                    return_block(2, "20"),
                    return_block(3, "30"),
                ],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName("switch".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn for_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let function = FunctionId(0);
    let local = |id| MirLocal {
        id: MirLocalId(id),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    };
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: vec![local(0), local(1)],
                blocks: vec![
                    BasicBlock {
                        id: BasicBlockId(0),
                        statements: vec![MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                    "0".into(),
                                ))),
                            },
                            span,
                        }],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(1)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(1),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::For {
                                binding: MirLocalId(0),
                                iterable: MirRvalue::Range {
                                    start: MirOperand::Constant(MirConstant::Number("1".into())),
                                    step: None,
                                    end: MirOperand::Constant(MirConstant::Number("3".into())),
                                },
                                body_block: BasicBlockId(2),
                                exit_block: BasicBlockId(3),
                            },
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(2),
                        statements: vec![MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::Binary(
                                    MirOperand::Local(MirLocalId(1)),
                                    runmat_types::OperatorKind::Add,
                                    MirOperand::Local(MirLocalId(0)),
                                ),
                            },
                            span,
                        }],
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Goto(BasicBlockId(1)),
                            span,
                        },
                    },
                    BasicBlock {
                        id: BasicBlockId(3),
                        statements: Vec::new(),
                        terminator: MirTerminator {
                            kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
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
                name: FunctionName("for_loop".into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn lower_body(
    name: &str,
    local_count: usize,
    statements: Vec<MirStmt>,
    returned: MirOperand,
    span: Span,
) -> runmat_native_codegen::NativeAssembly {
    let function = FunctionId(0);
    let mir = MirAssembly {
        bodies: [(
            function,
            MirBody {
                function,
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                locals: (0..local_count)
                    .map(|id| MirLocal {
                        id: MirLocalId(id),
                        binding: None,
                        kind: MirLocalKind::Temporary,
                        span,
                    })
                    .collect(),
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements,
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![returned]),
                        span,
                    },
                }],
            },
        )]
        .into_iter()
        .collect(),
        functions: [(
            function,
            MirFunctionMetadata {
                source: ProgramSourceId(0),
                name: FunctionName(name.into()),
                parent: None,
                enclosing_class: None,
                kind: FunctionKind::SyntheticEntrypoint,
                argument_validations: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                span,
            },
        )]
        .into_iter()
        .collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    };
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn manifest(analysis_schema: u16) -> ExecutableUnitManifest {
    let program = ProgramRevision::new(
        Digest::sha256(b"r13-graph"),
        Digest::sha256(b"r13-sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"r13-runtime"),
            Digest::sha256(b"r13-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let revisions = ExecutableComponentRevisions {
        catalog_schema: 1,
        catalog_fingerprint: *program.catalog_fingerprint(),
        contract_schema: 1,
        contract_fingerprint: Digest::sha256(b"r13-contracts"),
        analysis_schema,
        mir_schema: runmat_mir::MIR_SCHEMA_VERSION,
        bytecode_schema: 1,
        vm_layout_schema: 1,
        function_registry_schema: 1,
        source_map_schema: 1,
        region_schema: REGION_CONTRACT_SCHEMA_VERSION,
        interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
        parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
    };
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}-r13").into_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            let schema = match payload.kind {
                ExecutableComponentKind::Mir => revisions.mir_schema,
                ExecutableComponentKind::Analysis => revisions.analysis_schema,
                ExecutableComponentKind::Bytecode => revisions.bytecode_schema,
                ExecutableComponentKind::VmLayout => revisions.vm_layout_schema,
                ExecutableComponentKind::FunctionRegistry => revisions.function_registry_schema,
                ExecutableComponentKind::SourceMap => revisions.source_map_schema,
            };
            ExecutableComponentDescriptor::from_payload(payload.kind, schema, &payload.bytes)
                .unwrap()
        })
        .collect();
    ExecutableUnitManifest {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        identity: ExecutableIdentity {
            program,
            root_package: "r13-fixture@1.0.0".into(),
            entrypoint: "main".into(),
            entrypoint_function: ProgramFunctionId(0),
            entrypoint_kind: ExecutableEntrypointKind::Function,
            source_digest: Digest::sha256(b"r13-main.m"),
        },
        revisions,
        components,
        capabilities: CapabilitySet::default(),
        regions: Vec::new(),
        interop: InteropManifest::default(),
        parallel: ParallelManifest::default(),
        optional_sections: Vec::new(),
    }
}
