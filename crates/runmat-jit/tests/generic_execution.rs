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
    BasicBlock, BasicBlockId, MirAssembly, MirBody, MirConstant, MirFunctionMetadata, MirLocal,
    MirLocalId, MirLocalKind, MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminator,
    MirTerminatorKind,
};
use runmat_native_codegen::{lower_executable, NativeLoweringInput, NativeTarget};
use runmat_runtime::{context::RuntimeContext, execution::RuntimeExecutionService};
use runmat_types::{
    CapabilitySet, InteropManifest, ParallelManifest, ProgramFunctionId, ProgramSourceId,
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
