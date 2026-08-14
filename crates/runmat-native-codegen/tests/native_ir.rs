use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitManifest, ProgramEnvironment, ProgramRevision, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_hir::{
    FunctionAbi, FunctionArgumentValidation, FunctionId, FunctionKind, FunctionModifiers,
    FunctionName, WorkspaceEffect,
};
use runmat_mir::{
    BasicBlock, BasicBlockId, MirAssembly, MirBody, MirConstant, MirFunctionMetadata, MirLocal,
    MirLocalId, MirLocalKind, MirOperand, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminator,
    MirTerminatorKind,
};
use runmat_native_codegen::{
    analyze_liveness, lower_executable, print_native_ir, verify_against_manifest,
    verify_against_mir, NativeLoweringInput, NativeTarget,
};
use runmat_types::{
    BindingId, CapabilityRequirement, CapabilitySet, CollectiveId, DeoptimizationPointId,
    DistributedValueId, DynamicReason, ForeignAffinity, ForeignCapability, ForeignLifetime,
    ForeignOwnership, ForeignRequirement, ForeignTypeIdentity, FunctionArgDefaultValue,
    FunctionArgDim, FunctionArgSizeSpec, FunctionArgValidator, InteropManifest, LabCount,
    ParallelAccess, ParallelManifest, ParallelRandomnessPolicy, ParallelRegionId,
    ParallelVariableContract, ParallelVariableRole, ParforContract, ProgramFunctionId,
    ProgramPointId, ProgramSourceId, ProgramSpan, RegionContract, RegionGuardCondition,
    RegionGuardContract, RegionGuardId, RegionId, RegionProvenance, RegionValueId, Span,
    SpmdContract, SpmdLabRequirement, ValueFact, WasmInteropPolicy,
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};

fn component_payloads() -> Vec<ExecutableComponentPayload> {
    ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}-r12").into_bytes()).unwrap()
        })
        .collect()
}

fn manifest(analysis_schema: u16) -> ExecutableUnitManifest {
    let program = ProgramRevision::new(
        Digest::sha256(b"r12-graph"),
        Digest::sha256(b"r12-sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"r12-runtime"),
            Digest::sha256(b"r12-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let revisions = ExecutableComponentRevisions {
        catalog_schema: 1,
        catalog_fingerprint: *program.catalog_fingerprint(),
        contract_schema: 1,
        contract_fingerprint: Digest::sha256(b"r12-contracts"),
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
    let components = component_payloads()
        .iter()
        .map(|payload| {
            ExecutableComponentDescriptor::from_payload(
                payload.kind,
                match payload.kind {
                    ExecutableComponentKind::Mir => revisions.mir_schema,
                    ExecutableComponentKind::Analysis => revisions.analysis_schema,
                    ExecutableComponentKind::Bytecode => revisions.bytecode_schema,
                    ExecutableComponentKind::VmLayout => revisions.vm_layout_schema,
                    ExecutableComponentKind::FunctionRegistry => revisions.function_registry_schema,
                    ExecutableComponentKind::SourceMap => revisions.source_map_schema,
                },
                &payload.bytes,
            )
            .unwrap()
        })
        .collect();
    ExecutableUnitManifest {
        schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
        identity: ExecutableIdentity {
            program,
            root_package: "r12-fixture@1.0.0".into(),
            entrypoint: "main".into(),
            entrypoint_function: ProgramFunctionId(0),
            entrypoint_kind: ExecutableEntrypointKind::Function,
            source_digest: Digest::sha256(b"r12-main.m"),
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

fn function(statements: Vec<MirStmt>) -> MirAssembly {
    let span = Span { start: 0, end: 8 };
    let function = FunctionId(0);
    let body = MirBody {
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
            statements,
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
                span,
            },
        }],
    };
    let metadata = MirFunctionMetadata {
        source: ProgramSourceId(7),
        name: FunctionName("main".into()),
        parent: None,
        enclosing_class: None,
        kind: FunctionKind::SyntheticEntrypoint,
        argument_validations: Vec::new(),
        captures: Vec::new(),
        modifiers: FunctionModifiers::default(),
        span,
    };
    MirAssembly {
        bodies: [(function, body)].into_iter().collect(),
        functions: [(function, metadata)].into_iter().collect(),
        classes: Vec::new(),
        entrypoints: vec![function],
    }
}

fn assignment(number: usize) -> MirStmt {
    MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(0)),
            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
                number.to_string(),
            ))),
        },
        span: Span { start: 0, end: 1 },
    }
}

fn lower(mir: &MirAssembly) -> runmat_native_codegen::NativeAssembly {
    let analysis = runmat_mir::analysis::analyze_assembly(mir);
    let manifest = manifest(analysis.revision.schema_version);
    lower_with(mir, &analysis, &manifest).unwrap()
}

fn lower_with(
    mir: &MirAssembly,
    analysis: &runmat_mir::analysis::AnalysisStore,
    manifest: &ExecutableUnitManifest,
) -> Result<runmat_native_codegen::NativeAssembly, runmat_native_codegen::NativeCodegenError> {
    lower_with_bindings(mir, analysis, manifest, None)
}

fn lower_with_bindings(
    mir: &MirAssembly,
    analysis: &runmat_mir::analysis::AnalysisStore,
    manifest: &ExecutableUnitManifest,
    binding_names: Option<&std::collections::BTreeMap<BindingId, String>>,
) -> Result<runmat_native_codegen::NativeAssembly, runmat_native_codegen::NativeCodegenError> {
    lower_executable(NativeLoweringInput {
        mir,
        analysis,
        manifest,
        binding_names,
        target: NativeTarget::current(),
    })
}

#[test]
fn bound_locals_require_and_retain_canonical_semantic_names() {
    let mut mir = function(vec![assignment(1)]);
    let binding = BindingId(4);
    mir.bodies.get_mut(&FunctionId(0)).unwrap().locals[0].binding = Some(binding);
    mir.bodies.get_mut(&FunctionId(0)).unwrap().locals[0].kind = MirLocalKind::Binding;
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);

    let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
    assert_eq!(error.code, "native.lowering.binding_name");

    let names = std::collections::BTreeMap::from([(binding, "answer".to_string())]);
    let assembly = lower_with_bindings(&mir, &analysis, &manifest, Some(&names)).unwrap();
    let local = &assembly.functions[0].locals[0];
    assert_eq!(local.id, runmat_native_codegen::NativeLocalId(0));
    assert_eq!(local.binding, Some(binding));
    assert_eq!(local.name.as_deref(), Some("answer"));
    assert_eq!(local.kind, runmat_native_codegen::NativeLocalKind::Binding);
    assert_eq!(assembly.schema_version, 4);
    assembly.verify().unwrap();
    verify_against_mir(&assembly, &mir, Some(&names)).unwrap();

    let mut tampered = assembly;
    tampered.functions[0].locals[0].name = Some("different".into());
    assert_eq!(
        verify_against_mir(&tampered, &mir, Some(&names))
            .unwrap_err()
            .code,
        "native.ir.mir_locals"
    );
}

#[test]
fn argument_validations_round_trip_and_are_bound_to_canonical_mir() {
    let mut mir = function(vec![assignment(1)]);
    let binding = BindingId(9);
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    body.locals[0].binding = Some(binding);
    body.locals[0].kind = MirLocalKind::Parameter;
    body.abi.fixed_inputs = vec![binding];
    mir.functions
        .get_mut(&FunctionId(0))
        .unwrap()
        .argument_validations = vec![FunctionArgumentValidation {
        binding,
        size: Some(FunctionArgSizeSpec {
            rows: FunctionArgDim::Exact(1),
            cols: FunctionArgDim::Any,
        }),
        class_name: Some("double".into()),
        validators: vec![FunctionArgValidator::Positive],
        default_value: Some(FunctionArgDefaultValue::Number(1.0)),
    }];
    let names = std::collections::BTreeMap::from([(binding, "input".to_string())]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    let assembly = lower_with_bindings(&mir, &analysis, &manifest, Some(&names)).unwrap();

    let validation = &assembly.functions[0].argument_validations[0];
    assert_eq!(validation.input, runmat_native_codegen::NativeLocalId(0));
    assert_eq!(validation.validators, vec![FunctionArgValidator::Positive]);
    assert_eq!(
        validation.default_value,
        Some(FunctionArgDefaultValue::Number(1.0))
    );
    assembly.verify().unwrap();
    verify_against_mir(&assembly, &mir, Some(&names)).unwrap();

    let mut noncanonical = assembly.clone();
    noncanonical.functions[0].argument_validations[0].default_value =
        Some(FunctionArgDefaultValue::Number(2.0));
    assert_eq!(
        verify_against_mir(&noncanonical, &mir, Some(&names))
            .unwrap_err()
            .code,
        "native.ir.mir_argument_validations"
    );

    let mut abi_tampered = assembly.clone();
    abi_tampered.functions[0].abi.fixed_inputs.clear();
    abi_tampered.functions[0].argument_validations.clear();
    abi_tampered.verify().unwrap();
    assert_eq!(
        verify_against_mir(&abi_tampered, &mir, Some(&names))
            .unwrap_err()
            .code,
        "native.ir.mir_function_abi"
    );

    let mut malformed = assembly;
    malformed.functions[0].argument_validations[0].validators =
        vec![FunctionArgValidator::GreaterThan(f64::NAN)];
    assert_eq!(
        malformed.verify().unwrap_err().code,
        "native.ir.argument_validations"
    );
}

#[test]
fn end_expression_catalog_is_verified_against_canonical_mir() {
    let span = Span { start: 0, end: 3 };
    let mut mir = function(vec![
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::End,
            },
            span,
        },
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(1)),
                value: MirRvalue::Binary(
                    MirOperand::Local(MirLocalId(0)),
                    runmat_types::OperatorKind::Subtract,
                    MirOperand::Constant(MirConstant::Number("1".into())),
                ),
            },
            span,
        },
    ]);
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    body.locals.push(MirLocal {
        id: MirLocalId(1),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    });
    body.blocks[0].terminator.kind =
        MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(1))]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assert_eq!(assembly.functions[0].index_expressions.len(), 2);
    assert!(matches!(
        &assembly.functions[0].index_expressions[1].kind,
        runmat_native_codegen::NativeIndexExpressionKind::Scalar(
            runmat_runtime::indexing::EndExpr::Sub(_, _)
        )
    ));

    let mut structurally_invalid = assembly.clone();
    structurally_invalid.functions[0].index_expressions[1].kind =
        runmat_native_codegen::NativeIndexExpressionKind::Scalar(
            runmat_runtime::indexing::EndExpr::Var(99),
        );
    assert_eq!(
        structurally_invalid.verify().unwrap_err().code,
        "native.ir.index_expressions"
    );

    let mut redirected_output = assembly.clone();
    redirected_output.functions[0].blocks[0].instructions[0].outputs[0].local =
        Some(runmat_native_codegen::NativeLocalId(1));
    assert_eq!(
        redirected_output.verify().unwrap_err().code,
        "native.ir.output_local_identity"
    );

    let mut coordinated_redirect = assembly.clone();
    coordinated_redirect.functions[0].blocks[0].instructions[0].outputs[0].local =
        Some(runmat_native_codegen::NativeLocalId(1));
    coordinated_redirect.functions[0].blocks[0].instructions[1].outputs[0].local =
        Some(runmat_native_codegen::NativeLocalId(1));
    let runmat_native_codegen::NativeOperation::Statement(MirStmtKind::Assign { place, .. }) =
        &mut coordinated_redirect.functions[0].blocks[0].instructions[1].operation
    else {
        panic!("fixture statement must be an assignment");
    };
    *place = MirPlace::Local(MirLocalId(1));
    coordinated_redirect.verify().unwrap();
    assert_eq!(
        verify_against_mir(&coordinated_redirect, &mir, None)
            .unwrap_err()
            .code,
        "native.ir.mir_operation"
    );

    let mut omitted = assembly;
    omitted.functions[0].index_expressions.pop();
    omitted.verify().unwrap();
    assert_eq!(
        verify_against_mir(&omitted, &mir, None).unwrap_err().code,
        "native.ir.mir_index_expressions"
    );
}

#[test]
fn generic_ir_round_trips_prints_deterministically_and_tracks_effect_epochs() {
    let mut statements = vec![assignment(1)];
    statements.push(MirStmt {
        kind: MirStmtKind::WorkspaceEffect {
            effect: WorkspaceEffect::CreatesBinding,
            bindings: vec![MirLocalId(0)],
        },
        span: Span { start: 2, end: 3 },
    });
    let assembly = lower(&function(statements));
    assembly.verify().unwrap();

    let encoded = serde_json::to_vec(&assembly).unwrap();
    let decoded = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(assembly, decoded);
    assert_eq!(print_native_ir(&assembly), print_native_ir(&decoded));

    let block = &assembly.functions[0].blocks[0];
    assert_eq!(assembly.functions[0].source, ProgramSourceId(7));
    assert!(block
        .instructions
        .iter()
        .all(|instruction| instruction.source.source == ProgramSourceId(7)));
    assert!(matches!(
        block.instructions[0].outputs[0].value_type,
        runmat_native_codegen::NativeValueType::Analyzed(_)
    ));
    let effectful = block
        .instructions
        .iter()
        .find(|instruction| !instruction.effects.0.is_empty())
        .unwrap();
    let next_epoch = effectful.effect_epoch_output.unwrap();
    assert_eq!(block.terminator.frame_state.side_effect_epoch, next_epoch);
    assert!(analyze_liveness(&assembly).contains_key(&ProgramFunctionId(0)));
}

#[test]
fn deterministic_printer_matches_the_reviewable_snapshot() {
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
            "1".into(),
        )))),
        span: Span { start: 0, end: 1 },
    }]);
    let rendered = print_native_ir(&lower(&mir));
    let (_, target_independent_body) = rendered.split_once('\n').unwrap();
    assert_eq!(
        target_independent_body,
        include_str!("snapshots/simple.native-ir")
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn generic_cranelift_executes_the_native_block_graph_through_typed_sites() {
    use cranelift_codegen::settings::Configurable;
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{default_libcall_names, Linkage, Module};
    use runmat_runtime::native::{
        NativeCall, NativeEntryPoint, NativeExit, NativeFrame, NativeHostStatus, NativeHostVTable,
        NativeResumeState, NativeSiteOutcome, NativeSiteOutcomeKind, NativeSitePhase,
        NativeSiteRequest, NativeSourceMapEntry, NativeValueRef, NATIVE_ABI_VERSION,
    };
    use std::ffi::c_void;

    #[derive(Default)]
    struct SiteLog(Vec<NativeSiteRequest>);

    unsafe extern "C" fn retain(_: *mut c_void, _: NativeValueRef) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn release(_: *mut c_void, _: NativeValueRef) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn slow(
        _: *mut c_void,
        _: *mut NativeCall,
        _: *mut NativeExit,
    ) -> NativeHostStatus {
        NativeHostStatus::HOST_FAILURE
    }

    unsafe extern "C" fn safepoint(
        _: *mut c_void,
        _: *const runmat_runtime::native::NativeSafepoint,
        _: *mut NativeExit,
    ) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn source(
        _: *mut c_void,
        _: u32,
        output: *mut NativeSourceMapEntry,
    ) -> NativeHostStatus {
        if output.is_null() {
            return NativeHostStatus::INVALID_ARGUMENT;
        }
        // SAFETY: the ABI caller supplied the checked non-null output slot.
        unsafe { *output = NativeSourceMapEntry::default() };
        NativeHostStatus::OK
    }

    unsafe extern "C" fn site(
        context: *mut c_void,
        _: *mut NativeCall,
        request: *const NativeSiteRequest,
        outcome: *mut NativeSiteOutcome,
        exit: *mut NativeExit,
    ) -> NativeHostStatus {
        if context.is_null() || request.is_null() || outcome.is_null() || exit.is_null() {
            return NativeHostStatus::INVALID_ARGUMENT;
        }
        // SAFETY: every pointer was checked and remains borrowed for this callback.
        let request = unsafe { *request };
        if request.validate().is_err() {
            return NativeHostStatus::INVALID_ARGUMENT;
        }
        // SAFETY: context is the SiteLog installed in the vtable below.
        unsafe { &mut *context.cast::<SiteLog>() }.0.push(request);
        let decision = if request.phase == NativeSitePhase::TERMINATOR {
            match request.block {
                0 => NativeSiteOutcome::edge(0),
                1 => {
                    // SAFETY: the generated entrypoint supplied its writable exit slot.
                    unsafe { *exit = NativeExit::completed(0) };
                    NativeSiteOutcome::exit()
                }
                _ => return NativeHostStatus::HOST_FAILURE,
            }
        } else {
            NativeSiteOutcome::continue_execution()
        };
        // SAFETY: the generated entrypoint supplied its writable outcome slot.
        unsafe { *outcome = decision };
        NativeHostStatus::OK
    }

    let mut mir = function(vec![assignment(1)]);
    let span = Span { start: 0, end: 8 };
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    body.blocks[0].terminator = MirTerminator {
        kind: MirTerminatorKind::Branch {
            cond: MirOperand::Constant(MirConstant::Number("1".into())),
            then_block: BasicBlockId(1),
            else_block: BasicBlockId(2),
        },
        span,
    };
    body.blocks.extend([
        BasicBlock {
            id: BasicBlockId(1),
            statements: Vec::new(),
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(Vec::new()),
                span,
            },
        },
        BasicBlock {
            id: BasicBlockId(2),
            statements: Vec::new(),
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(Vec::new()),
                span,
            },
        },
    ]);
    let native = lower(&mir);
    let compiled =
        runmat_native_codegen::cranelift::lower_function(&native.functions[0], &native.target)
            .unwrap();

    let mut flags = cranelift_codegen::settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("enable_verifier", "true").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(cranelift_codegen::settings::Flags::new(flags))
        .unwrap();
    let builder = JITBuilder::with_isa(isa, default_libcall_names());
    let mut module = JITModule::new(builder);
    let function_id = module
        .declare_function(
            "runmat_r13_generic_test",
            Linkage::Export,
            &compiled.ir.signature,
        )
        .unwrap();
    let mut context = module.make_context();
    context.func = compiled.ir;
    module.define_function(function_id, &mut context).unwrap();
    module.clear_context(&mut context);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(function_id);
    // SAFETY: the lowered function uses the exact runtime-owned NativeEntryPoint ABI.
    let entry: NativeEntryPoint = unsafe { std::mem::transmute(code) };

    let mut log = SiteLog::default();
    let host = NativeHostVTable {
        abi_version: NATIVE_ABI_VERSION.encoded(),
        struct_size: std::mem::size_of::<NativeHostVTable>() as u32,
        context: (&mut log as *mut SiteLog).cast(),
        retain_value: Some(retain),
        release_value: Some(release),
        slow_call: Some(slow),
        poll_safepoint: Some(safepoint),
        source_lookup: Some(source),
        execute_site: Some(site),
    };
    host.validate().unwrap();
    let mut resume = NativeResumeState::default();
    let mut frame = NativeFrame {
        resume: &mut resume,
        ..NativeFrame::default()
    };
    let mut call = NativeCall {
        host: &host,
        frame: &mut frame,
        ..NativeCall::default()
    };
    let mut exit = NativeExit::completed(0);
    // SAFETY: call, frame, host table, and exit remain live for the invocation.
    let status = unsafe { entry(&mut call, &mut exit) };
    assert_eq!(status, NativeHostStatus::OK);
    call.validate_exit(&exit).unwrap();
    assert!(log.0.iter().any(|site| site.block == 0));
    assert!(log.0.iter().any(|site| site.block == 1));
    assert!(!log.0.iter().any(|site| site.block == 2));
    assert_eq!(
        log.0.last().map(|site| site.phase),
        Some(NativeSitePhase::TERMINATOR)
    );
    assert_eq!(NativeSiteOutcomeKind::EXIT.0, 2);

    // SAFETY: a null call is intentionally passed to exercise generated ABI validation.
    let invalid_status = unsafe { entry(std::ptr::null_mut(), &mut exit) };
    assert_eq!(invalid_status, NativeHostStatus::INVALID_ARGUMENT);
}

#[test]
fn verifier_rejects_omission_stale_effect_state_and_abi_drift() {
    let mut statements = vec![assignment(1)];
    statements.push(MirStmt {
        kind: MirStmtKind::WorkspaceEffect {
            effect: WorkspaceEffect::CreatesBinding,
            bindings: vec![MirLocalId(0)],
        },
        span: Span { start: 2, end: 3 },
    });
    let assembly = lower(&function(statements));

    let mut omitted = lower(&function(vec![MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
            "1".into(),
        )))),
        span: Span { start: 0, end: 1 },
    }]));
    omitted.functions[0].blocks[0].instructions.drain(0..2);
    assert_eq!(
        omitted.verify().unwrap_err().code,
        "native.ir.construct_coverage"
    );

    let mut stale = assembly.clone();
    stale.functions[0].blocks[0]
        .terminator
        .frame_state
        .side_effect_epoch = stale.functions[0].blocks[0].side_effect_epoch;
    assert_eq!(
        stale.verify().unwrap_err().code,
        "native.ir.terminator_frame_state"
    );

    let mut wrong_abi = assembly;
    wrong_abi.target.abi.encoded_version += 1;
    assert_eq!(wrong_abi.verify().unwrap_err().code, "native.abi.binding");
}

#[test]
fn native_cache_identity_binds_target_and_abi() {
    let assembly = lower(&function(vec![assignment(1)]));
    assert_ne!(assembly.native_cache_key, assembly.executable_cache_key);
    let mut tampered = assembly.clone();
    tampered.native_cache_key = Digest::sha256(b"wrong-native-key");
    assert_eq!(tampered.verify().unwrap_err().code, "native.ir.cache_key");

    let mut cross_target = assembly.target;
    cross_target.architecture = "not-the-current-architecture".into();
    assert_eq!(
        cross_target.validate().unwrap_err().code,
        "native.target.cross_layout"
    );
}

#[test]
fn executable_manifest_binding_rejects_a_coordinated_cache_key_substitution() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    let mut assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assembly.executable_cache_key = Digest::sha256(b"substituted-executable");
    assembly.native_cache_key = assembly
        .target
        .cache_key(&assembly.executable_cache_key)
        .unwrap();
    assembly.verify().unwrap();
    assert_eq!(
        verify_against_manifest(&assembly, &manifest)
            .unwrap_err()
            .code,
        "native.ir.manifest_binding"
    );
}

#[test]
fn canonical_mir_verification_rejects_coordinated_ir_and_inventory_omission() {
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Use(MirOperand::Constant(MirConstant::Number(
            "1".into(),
        )))),
        span: Span { start: 0, end: 1 },
    }]);
    let mut assembly = lower(&mir);
    assembly.functions[0].blocks[0].instructions.drain(0..2);
    assembly.functions[0].expected_sites.drain(0..2);
    assembly.verify().unwrap();
    assert_eq!(
        verify_against_mir(&assembly, &mir, None).unwrap_err().code,
        "native.ir.mir_construct_coverage"
    );
}

#[test]
fn lowering_is_deterministic_across_a_property_style_program_family() {
    for length in 1..=64 {
        let mir = function((0..length).map(assignment).collect());
        let first = lower(&mir);
        let second = lower(&mir);
        assert_eq!(first, second, "program family member {length}");
        first.verify().unwrap();
    }
}

#[test]
fn region_contracts_become_exact_native_ir_boundaries() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(analysis.revision.schema_version);
    let id = RegionId {
        function: ProgramFunctionId(0),
        ordinal: 9,
    };
    let live = RegionValueId {
        function: ProgramFunctionId(0),
        local: 0,
    };
    manifest.regions.push(RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id,
        source: ProgramSourceId(7),
        span: ProgramSpan { start: 0, end: 1 },
        entry: ProgramPointId {
            function: ProgramFunctionId(0),
            block: 0,
            position: 0,
        },
        exits: Vec::new(),
        live_in: vec![live],
        live_out: Vec::new(),
        value_facts: Vec::new(),
        effects: Default::default(),
        capabilities: Default::default(),
        guards: vec![RegionGuardContract {
            id: RegionGuardId {
                region: id,
                ordinal: 0,
            },
            condition: RegionGuardCondition::ValueFact {
                value: live,
                expected: ValueFact::unknown(DynamicReason::RuntimeValue),
            },
            deopt: DeoptimizationPointId {
                function: ProgramFunctionId(0),
                ordinal: 0,
            },
        }],
        provenance: RegionProvenance::Inferred,
    });
    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    let boundaries = &assembly.functions[0].blocks[0].region_boundaries;
    assert_eq!(boundaries.len(), 1);
    assert_eq!(boundaries[0].region, id);
    assert_eq!(boundaries[0].live_values[0].value, live);
    assert_eq!(
        boundaries[0].guards[0].value,
        Some(boundaries[0].live_values[0].ssa)
    );
    assembly.verify().unwrap();
}

#[test]
fn distributed_and_collective_constructs_fail_with_the_stable_predeclared_rejection() {
    let distributed = MirRvalue::Distributed(runmat_mir::parallel::MirDistributedOp::LocalPart {
        value: DistributedValueId {
            function: ProgramFunctionId(0),
            ordinal: 0,
        },
    });
    let region = ParallelRegionId(RegionId {
        function: ProgramFunctionId(0),
        ordinal: 0,
    });
    let collective = MirRvalue::Collective(runmat_mir::parallel::MirCollectiveOp::Barrier {
        id: CollectiveId { region, ordinal: 0 },
    });
    for value in [distributed, collective] {
        let mir = function(vec![MirStmt {
            kind: MirStmtKind::Expr(value),
            span: Span { start: 0, end: 1 },
        }]);
        let analysis = runmat_mir::analysis::analyze_assembly(&mir);
        let manifest = manifest(analysis.revision.schema_version);
        let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
        assert_eq!(error.code, "native.capability.distributed_core_pending");
        assert_eq!(
            error.construct.unwrap().native_lowering_class(),
            runmat_mir::NativeLoweringClass::CapabilityRejection
        );
    }
}

#[test]
fn capability_rejection_cannot_hide_inside_short_circuit_payloads() {
    let nested = MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Distributed(
            runmat_mir::parallel::MirDistributedOp::LocalPart {
                value: DistributedValueId {
                    function: ProgramFunctionId(0),
                    ordinal: 0,
                },
            },
        )),
        span: Span { start: 0, end: 1 },
    };
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::ShortCircuit {
            left: MirOperand::Constant(MirConstant::Bool(true)),
            op: runmat_mir::MirShortCircuitOp::And,
            right_temps: vec![nested],
            right: MirOperand::Constant(MirConstant::Bool(true)),
        }),
        span: Span { start: 0, end: 1 },
    }]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(analysis.revision.schema_version);
    assert_eq!(
        lower_with(&mir, &analysis, &manifest).unwrap_err().code,
        "native.capability.distributed_core_pending"
    );
}

#[test]
fn short_circuit_embedded_constructs_are_explicit_and_verified() {
    let nested = MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Unary(
            runmat_types::OperatorKind::Not,
            MirOperand::Constant(MirConstant::Bool(false)),
        )),
        span: Span { start: 0, end: 1 },
    };
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::ShortCircuit {
            left: MirOperand::Constant(MirConstant::Bool(true)),
            op: runmat_mir::MirShortCircuitOp::And,
            right_temps: vec![nested],
            right: MirOperand::Constant(MirConstant::Bool(true)),
        }),
        span: Span { start: 0, end: 1 },
    }]);
    let mut assembly = lower(&mir);
    let instruction = &assembly.functions[0].blocks[0].instructions[0];
    assert_eq!(
        instruction.embedded_constructs,
        [
            runmat_mir::MirConstructKind::Unary,
            runmat_mir::MirConstructKind::Expr
        ]
    );
    assembly.functions[0].blocks[0].instructions[0]
        .embedded_constructs
        .clear();
    assert_eq!(
        assembly.verify().unwrap_err().code,
        "native.ir.embedded_constructs"
    );
}

#[test]
fn canonical_construct_taxonomy_is_complete_unique_and_serializable() {
    let all = runmat_mir::MirConstructKind::ALL;
    assert_eq!(all.len(), 47);
    assert_eq!(
        all.iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        all.len()
    );
    for construct in all {
        let bytes = serde_json::to_vec(&construct).unwrap();
        assert_eq!(construct, serde_json::from_slice(&bytes).unwrap());
        let _ = construct.native_lowering_class();
    }
}

#[test]
fn parfor_and_spmd_fixtures_lower_as_structured_suspend_resume_ir() {
    for spmd in [false, true] {
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(0),
            ordinal: if spmd { 2 } else { 1 },
        });
        let terminator = if spmd {
            MirTerminatorKind::Spmd {
                region,
                header: Box::new(runmat_mir::parallel::MirSpmdHeader::One(MirRvalue::Use(
                    MirOperand::Constant(MirConstant::Number("2".into())),
                ))),
                body_block: BasicBlockId(1),
                exit_block: BasicBlockId(2),
            }
        } else {
            MirTerminatorKind::ParFor {
                region,
                binding: MirLocalId(0),
                iterable: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("1".into()))),
                maximum_workers: Some(Box::new(MirRvalue::Use(MirOperand::Constant(
                    MirConstant::Number("4".into()),
                )))),
                body_block: BasicBlockId(1),
                exit_block: BasicBlockId(2),
            }
        };
        let mut mir = function(Vec::new());
        let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
        body.blocks = vec![
            BasicBlock {
                id: BasicBlockId(0),
                statements: Vec::new(),
                terminator: MirTerminator {
                    kind: terminator,
                    span: Span { start: 0, end: 1 },
                },
            },
            return_block(1),
            return_block(2),
        ];
        let analysis = runmat_mir::analysis::analyze_assembly(&mir);
        let mut manifest = manifest(analysis.revision.schema_version);
        manifest
            .capabilities
            .0
            .insert(CapabilityRequirement::ParallelRuntime);
        manifest.regions.push(region_contract(region.0));
        let loop_value = RegionValueId {
            function: ProgramFunctionId(0),
            local: 0,
        };
        if spmd {
            manifest.parallel.spmd_regions.push(SpmdContract {
                id: region,
                labs: SpmdLabRequirement::Exact { labs: LabCount(2) },
                captures: Vec::new(),
                capabilities: CapabilitySet::default(),
            });
        } else {
            manifest.parallel.parfor_regions.push(ParforContract {
                id: region,
                loop_variable: loop_value,
                iterable: ValueFact::unknown(DynamicReason::RuntimeValue),
                variables: vec![ParallelVariableContract {
                    value: loop_value,
                    role: ParallelVariableRole::Loop,
                    access: ParallelAccess::ReadWrite,
                    fact: ValueFact::unknown(DynamicReason::RuntimeValue),
                    transferable: true,
                }],
                maximum_workers: Some(LabCount(4)),
                capabilities: CapabilitySet::default(),
                randomness: ParallelRandomnessPolicy::DeterministicSubstreams,
            });
        }
        let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
        assembly.verify().unwrap();
        let terminator = &assembly.functions[0].blocks[0].terminator;
        assert_eq!(
            terminator.class,
            runmat_mir::NativeLoweringClass::StructuredSuspendResume
        );
        assert!(terminator.safepoint.is_some());
    }
}

#[test]
fn future_spawn_and_await_carry_exact_structured_safepoints() {
    let future = MirRvalue::Future {
        function: FunctionId(0),
        args: Vec::new(),
        syntax: runmat_hir::CallSyntax::Plain,
        requested_outputs: runmat_hir::RequestedOutputCount::One,
    };
    let mut mir = function(vec![
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: future,
            },
            span: Span { start: 0, end: 1 },
        },
        MirStmt {
            kind: MirStmtKind::Expr(MirRvalue::Spawn(MirOperand::Local(MirLocalId(0)))),
            span: Span { start: 1, end: 2 },
        },
    ]);
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    body.blocks[0].terminator = MirTerminator {
        kind: MirTerminatorKind::Await {
            future: MirOperand::Local(MirLocalId(0)),
            result: Some(MirPlace::Local(MirLocalId(0))),
            resume: BasicBlockId(1),
        },
        span: Span { start: 2, end: 3 },
    };
    body.blocks.push(return_block(1));
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(analysis.revision.schema_version);
    manifest
        .capabilities
        .0
        .insert(CapabilityRequirement::ParallelRuntime);
    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    let block = &assembly.functions[0].blocks[0];
    let structured = block
        .instructions
        .iter()
        .filter(|instruction| {
            instruction.class == runmat_mir::NativeLoweringClass::StructuredSuspendResume
        })
        .collect::<Vec<_>>();
    assert_eq!(structured.len(), 2);
    assert!(structured.iter().all(|instruction| {
        instruction.safepoint.is_some() && instruction.frame_state.is_some()
    }));
    assert_eq!(
        block.terminator.class,
        runmat_mir::NativeLoweringClass::StructuredSuspendResume
    );
    assert!(block.terminator.safepoint.is_some());
    assembly.verify().unwrap();
}

#[test]
fn foreign_requirements_are_preserved_without_backend_inference() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(analysis.revision.schema_version);
    manifest
        .capabilities
        .0
        .insert(CapabilityRequirement::ForeignRuntime);
    manifest.interop.foreign_types.push(ForeignRequirement {
        type_identity: ForeignTypeIdentity {
            family: "java".into(),
            name: "java.lang.Object".into(),
            version: 1,
        },
        ownership: ForeignOwnership::Shared,
        affinity: ForeignAffinity::OriginProcess,
        lifetime: ForeignLifetime::Session,
        capabilities: vec![ForeignCapability::Invoke],
        wasm: WasmInteropPolicy::HostBridge,
    });
    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assert_eq!(assembly.requirements.interop, manifest.interop);
    assembly.verify().unwrap();
}

fn return_block(id: usize) -> BasicBlock {
    BasicBlock {
        id: BasicBlockId(id),
        statements: Vec::new(),
        terminator: MirTerminator {
            kind: MirTerminatorKind::Return(Vec::new()),
            span: Span { start: 1, end: 2 },
        },
    }
}

fn region_contract(id: RegionId) -> RegionContract {
    RegionContract {
        schema_version: REGION_CONTRACT_SCHEMA_VERSION,
        id,
        source: ProgramSourceId(7),
        span: ProgramSpan { start: 0, end: 1 },
        entry: ProgramPointId {
            function: id.function,
            block: 0,
            position: 0,
        },
        exits: Vec::new(),
        live_in: Vec::new(),
        live_out: Vec::new(),
        value_facts: Vec::new(),
        effects: Default::default(),
        capabilities: Default::default(),
        guards: Vec::new(),
        provenance: RegionProvenance::Inferred,
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_uses_the_same_generic_ir_contract() {
    let assembly = lower(&function(vec![assignment(1)]));
    assembly.verify().unwrap();
    assert_eq!(assembly.target.architecture, "wasm32");
    assert_eq!(
        assembly,
        serde_json::from_slice(&serde_json::to_vec(&assembly).unwrap()).unwrap()
    );
}
