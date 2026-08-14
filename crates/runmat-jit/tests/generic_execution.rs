#![cfg(not(target_arch = "wasm32"))]

use std::{
    collections::BTreeMap,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitManifest, ProgramEnvironment, ProgramRevision, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_hir::{
    FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, Span, WorkspaceEffect,
};
use runmat_jit::{GenericExecutor, JitError};
use runmat_mir::{
    AsyncBehaviorFact, BasicBlock, BasicBlockId, MirAggregateKind, MirAssembly, MirBody, MirCall,
    MirCallArg, MirCallee, MirConstant, MirFunctionMetadata, MirIndexComponent, MirIndexPlan,
    MirIndexing, MirLocal, MirLocalId, MirLocalKind, MirOperand, MirOutputTarget,
    MirOutputTargetList, MirPlace, MirPlaceMutation, MirRvalue, MirStmt, MirStmtKind,
    MirTerminator, MirTerminatorKind,
};
use runmat_native_codegen::{lower_executable, NativeLoweringInput, NativeTarget};
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use runmat_types::{
    BindingId, BuiltinId, CallableFallbackPolicy, CallableIdentity, CapabilitySet, InteropManifest,
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
fn suspending_semantic_call_exits_explicitly_without_nested_executor_or_replay() {
    let runtime = runtime_context();
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(|function, _arguments, requested_outputs| {
            assert_eq!(function, 9);
            assert_eq!(requested_outputs, 1);
            Box::pin(std::future::pending::<
                Result<Value, runmat_runtime::RuntimeError>,
            >())
        }),
    ));
    drop(activation);
    let executor = GenericExecutor::compile(semantic_call_fixture()).unwrap();
    let error = executor
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
        .expect_err("R13 must reject a real suspension boundary");
    let JitError::UnsupportedSite(message) = error else {
        panic!("suspension must remain a typed unsupported-site exit");
    };
    assert!(message.contains("requires the R14 continuation cohort"));
    drop(invoker);
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

#[test]
fn generic_indexing_and_member_reads_use_shared_runtime_semantics() {
    let indexing = GenericExecutor::compile(index_fixture()).unwrap();
    assert_eq!(
        indexing
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(20.0)]
    );

    let member = GenericExecutor::compile(member_fixture()).unwrap();
    assert_eq!(
        member
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );
}

#[test]
fn generic_index_and_member_mutations_publish_updated_root_values() {
    let indexing = GenericExecutor::compile(index_assignment_fixture()).unwrap();
    let Value::Tensor(updated) = indexing
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected updated tensor")
    };
    assert_eq!(updated.materialize_f64(), vec![10.0, 99.0, 30.0]);

    let member = GenericExecutor::compile(member_assignment_fixture()).unwrap();
    let Value::Struct(updated) = member
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected updated structure")
    };
    assert_eq!(updated.fields.get("answer"), Some(&Value::Num(42.0)));

    let multi = GenericExecutor::compile(multi_assignment_fixture()).unwrap();
    assert_eq!(
        multi
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(33.0)]
    );
}

#[test]
fn generic_global_and_persistent_declarations_use_semantic_session_names() {
    let runtime = runtime_context();
    futures::executor::block_on(runtime.scope(async {
        runmat_runtime::workspace::session::store_global_named("shared", Value::Num(41.0));
        runmat_runtime::workspace::session::store_persistent_named(
            "persistent_counter",
            "calls",
            Value::Num(9.0),
        );
    }));

    let global = GenericExecutor::compile(workspace_binding_fixture(
        "global_counter",
        "shared",
        WorkspaceEffect::MutatesGlobal,
    ))
    .unwrap();
    assert_eq!(
        global
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime.clone())
            .unwrap()
            .outputs,
        vec![Value::Num(42.0)]
    );
    futures::executor::block_on(runtime.scope(async {
        assert_eq!(
            runmat_runtime::workspace::session::global_value("shared"),
            Some(Value::Num(42.0))
        );
    }));

    let persistent = GenericExecutor::compile(workspace_binding_fixture(
        "persistent_counter",
        "calls",
        WorkspaceEffect::MutatesPersistent,
    ))
    .unwrap();
    assert_eq!(
        persistent
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime.clone())
            .unwrap()
            .outputs,
        vec![Value::Num(10.0)]
    );
    futures::executor::block_on(runtime.scope(async {
        assert_eq!(
            runmat_runtime::workspace::session::persistent_named_value(
                "persistent_counter",
                "calls"
            ),
            Some(Value::Num(10.0))
        );
    }));
}

#[test]
fn generic_workspace_first_static_property_prefers_named_local_binding() {
    let executor = GenericExecutor::compile(workspace_first_static_fixture()).unwrap();
    assert_eq!(
        executor
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(55.0)]
    );
}

#[test]
fn generic_context_dependent_end_selectors_execute_once_against_the_base_shape() {
    let scalar = GenericExecutor::compile(end_scalar_fixture()).unwrap();
    assert_eq!(
        scalar
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
            .unwrap()
            .outputs,
        vec![Value::Num(30.0)]
    );

    let range = GenericExecutor::compile(end_range_fixture()).unwrap();
    let Value::Tensor(selected) = range
        .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime_context())
        .unwrap()
        .outputs
        .pop()
        .unwrap()
    else {
        panic!("expected end-relative range selection to return a tensor")
    };
    assert_eq!(selected.materialize_f64(), vec![10.0, 20.0, 30.0]);

    let runtime = runtime_context();
    let calls = Arc::new(AtomicUsize::new(0));
    let invoker_calls = Arc::clone(&calls);
    let activation = runtime.enter();
    let invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
        Arc::new(move |function, arguments, requested_outputs| {
            assert_eq!(function, 9);
            assert_eq!(requested_outputs, 1);
            invoker_calls.fetch_add(1, Ordering::SeqCst);
            let value = arguments[0].clone();
            Box::pin(async move { Ok(value) })
        }),
    ));
    drop(activation);
    let called = GenericExecutor::compile(end_call_fixture()).unwrap();
    assert_eq!(
        called
            .invoke(ProgramFunctionId(0), Vec::new(), 1, runtime)
            .unwrap()
            .outputs,
        vec![Value::Num(40.0)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(invoker);
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
        binding_names: None,
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
        binding_names: None,
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

fn semantic_call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "semantic_call",
        1,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::Static(CallableIdentity::BoundFunction(
                        runmat_types::FunctionId(9),
                    )),
                    args: vec![MirCallArg::Single(MirOperand::Constant(
                        MirConstant::Number("1".into()),
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
        binding_names: None,
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
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn index_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "index",
        2,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Tensor,
                        rows: 1,
                        cols: 3,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("10".into())),
                            MirOperand::Constant(MirConstant::Number("20".into())),
                            MirOperand::Constant(MirConstant::Number("30".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::Index {
                        base: MirOperand::Local(MirLocalId(0)),
                        indexing: MirIndexing {
                            kind: runmat_types::IndexKind::Paren,
                            plan: MirIndexPlan::Scalar,
                            components: vec![MirIndexComponent::Expr(MirOperand::Constant(
                                MirConstant::Number("2".into()),
                            ))],
                            result_context: runmat_types::IndexResultContext::ReadSingle,
                            cell_expand_all: false,
                        },
                    },
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(1)),
        span,
    )
}

fn member_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    lower_body(
        "member",
        2,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::StructLiteral {
                        fields: vec![(
                            runmat_types::MemberName("answer".into()),
                            MirOperand::Constant(MirConstant::Number("42".into())),
                        )],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::Member {
                        base: MirOperand::Local(MirLocalId(0)),
                        member: runmat_types::MemberName("answer".into()),
                    },
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(1)),
        span,
    )
}

fn index_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let indexing = MirIndexing {
        kind: runmat_types::IndexKind::Paren,
        plan: MirIndexPlan::Scalar,
        components: vec![MirIndexComponent::Expr(MirOperand::Constant(
            MirConstant::Number("2".into()),
        ))],
        result_context: runmat_types::IndexResultContext::AssignmentTarget,
        cell_expand_all: false,
    };
    let place = MirPlace::Index(Box::new(MirPlace::Local(MirLocalId(0))), indexing);
    lower_body(
        "index_assignment",
        1,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Tensor,
                        rows: 1,
                        cols: 3,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("10".into())),
                            MirOperand::Constant(MirConstant::Number("20".into())),
                            MirOperand::Constant(MirConstant::Number("30".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::PlaceMutation(MirPlaceMutation {
                    place: place.clone(),
                    kind: runmat_types::PlaceMutationKind::IndexedAssign,
                    creation_policy: runmat_types::AssignmentCreationPolicy::CreateArrayByIndex,
                    shape_policy: runmat_types::AssignmentShapePolicy::MatlabCompatible,
                }),
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place,
                    value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("99".into()))),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn member_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let place = MirPlace::Member(
        Box::new(MirPlace::Local(MirLocalId(0))),
        runmat_types::MemberName("answer".into()),
    );
    lower_body(
        "member_assignment",
        1,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::StructLiteral {
                        fields: vec![(
                            runmat_types::MemberName("answer".into()),
                            MirOperand::Constant(MirConstant::Number("1".into())),
                        )],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::PlaceMutation(MirPlaceMutation {
                    place: place.clone(),
                    kind: runmat_types::PlaceMutationKind::MemberAssign,
                    creation_policy: runmat_types::AssignmentCreationPolicy::CreateStructFieldPath,
                    shape_policy: runmat_types::AssignmentShapePolicy::MatlabCompatible,
                }),
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place,
                    value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("42".into()))),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(0)),
        span,
    )
}

fn multi_assignment_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    lower_body(
        "multi_assignment",
        4,
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Aggregate {
                        kind: MirAggregateKind::Cell,
                        rows: 1,
                        cols: 2,
                        elements: vec![
                            MirOperand::Constant(MirConstant::Number("11".into())),
                            MirOperand::Constant(MirConstant::Number("22".into())),
                        ],
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::MultiAssign {
                    targets: MirOutputTargetList {
                        targets: vec![
                            MirOutputTarget::Place(MirPlace::Local(MirLocalId(1))),
                            MirOutputTarget::Place(MirPlace::Local(MirLocalId(2))),
                        ],
                        requested_outputs: RequestedOutputCount::Exactly(2),
                    },
                    value: MirRvalue::Index {
                        base: MirOperand::Local(MirLocalId(0)),
                        indexing: MirIndexing {
                            kind: runmat_types::IndexKind::Brace,
                            plan: MirIndexPlan::Cell,
                            components: vec![MirIndexComponent::Colon],
                            result_context: runmat_types::IndexResultContext::ReadCommaList,
                            cell_expand_all: true,
                        },
                    },
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(3)),
                    value: MirRvalue::Binary(
                        MirOperand::Local(MirLocalId(1)),
                        runmat_types::OperatorKind::Add,
                        MirOperand::Local(MirLocalId(2)),
                    ),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(3)),
        span,
    )
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
        binding_names: None,
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn workspace_binding_fixture(
    name: &str,
    binding_name: &str,
    effect: WorkspaceEffect,
) -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let binding = BindingId(0);
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
                    binding: Some(binding),
                    kind: MirLocalKind::Binding,
                    span,
                }],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![
                        MirStmt {
                            kind: MirStmtKind::WorkspaceEffect {
                                effect,
                                bindings: vec![MirLocalId(0)],
                            },
                            span,
                        },
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(0)),
                                value: MirRvalue::Binary(
                                    MirOperand::Local(MirLocalId(0)),
                                    runmat_types::OperatorKind::Add,
                                    MirOperand::Constant(MirConstant::Number("1".into())),
                                ),
                            },
                            span,
                        },
                    ],
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
    let binding_names = BTreeMap::from([(binding, binding_name.to_string())]);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: Some(&binding_names),
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn workspace_first_static_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 3 };
    let function = FunctionId(0);
    let binding = BindingId(0);
    let local = |id, binding, kind| MirLocal {
        id: MirLocalId(id),
        binding,
        kind,
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
                locals: vec![
                    local(0, Some(binding), MirLocalKind::Binding),
                    local(1, None, MirLocalKind::Temporary),
                ],
                blocks: vec![BasicBlock {
                    id: BasicBlockId(0),
                    statements: vec![
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(0)),
                                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                                    "55".into(),
                                ))),
                            },
                            span,
                        },
                        MirStmt {
                            kind: MirStmtKind::Assign {
                                place: MirPlace::Local(MirLocalId(1)),
                                value: MirRvalue::WorkspaceFirstStaticProperty {
                                    workspace_name: runmat_hir::SymbolName("Candidate".into()),
                                    class_name: "Candidate".into(),
                                    property: runmat_hir::MemberName("Missing".into()),
                                },
                            },
                            span,
                        },
                    ],
                    terminator: MirTerminator {
                        kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]),
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
                name: FunctionName("workspace_first".into()),
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
    let binding_names = BTreeMap::from([(binding, "Candidate".to_string())]);
    lower_executable(NativeLoweringInput {
        mir: &mir,
        analysis: &analysis,
        manifest: &manifest,
        binding_names: Some(&binding_names),
        target: NativeTarget::current(),
    })
    .unwrap()
}

fn end_scalar_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Binary(
                MirOperand::Local(MirLocalId(1)),
                runmat_types::OperatorKind::Subtract,
                MirOperand::Constant(MirConstant::Number("1".into())),
            ),
            span,
        ),
        statement(
            3,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(2)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_scalar",
        4,
        statements,
        MirOperand::Local(MirLocalId(3)),
        span,
    )
}

fn end_range_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 5 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Binary(
                MirOperand::Local(MirLocalId(1)),
                runmat_types::OperatorKind::Subtract,
                MirOperand::Constant(MirConstant::Number("1".into())),
            ),
            span,
        ),
        statement(
            3,
            MirRvalue::Range {
                start: MirOperand::Constant(MirConstant::Number("1".into())),
                step: None,
                end: MirOperand::Local(MirLocalId(2)),
            },
            span,
        ),
        statement(
            4,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(3)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_range",
        5,
        statements,
        MirOperand::Local(MirLocalId(4)),
        span,
    )
}

fn end_call_fixture() -> runmat_native_codegen::NativeAssembly {
    let span = Span { start: 0, end: 4 };
    let mut statements = tensor_assignment(span);
    statements.extend([
        statement(1, MirRvalue::End, span),
        statement(
            2,
            MirRvalue::Call(MirCall {
                callee: MirCallee::Static(CallableIdentity::BoundFunction(
                    runmat_types::FunctionId(9),
                )),
                args: vec![MirCallArg::Single(MirOperand::Local(MirLocalId(1)))],
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
            span,
        ),
        statement(
            3,
            MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: MirIndexing {
                    kind: runmat_types::IndexKind::Paren,
                    plan: MirIndexPlan::SliceExpr,
                    components: vec![MirIndexComponent::Expr(MirOperand::Local(MirLocalId(2)))],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
            span,
        ),
    ]);
    lower_body(
        "end_call",
        4,
        statements,
        MirOperand::Local(MirLocalId(3)),
        span,
    )
}

fn tensor_assignment(span: Span) -> Vec<MirStmt> {
    vec![statement(
        0,
        MirRvalue::Aggregate {
            kind: MirAggregateKind::Tensor,
            rows: 1,
            cols: 4,
            elements: ["10", "20", "30", "40"]
                .into_iter()
                .map(|value| MirOperand::Constant(MirConstant::Number(value.into())))
                .collect(),
        },
        span,
    )]
}

fn statement(local: usize, value: MirRvalue, span: Span) -> MirStmt {
    MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(local)),
            value,
        },
        span,
    }
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
