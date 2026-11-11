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
#[cfg(not(target_arch = "wasm32"))]
use runmat_native_codegen::aot::{
    emit_relocatable_object, emit_relocatable_object_with_data, NativeObjectData,
    NativeOptimization,
};
use runmat_native_codegen::{
    analyze_liveness, lower_executable, print_native_ir, verify_against_analysis,
    verify_against_manifest, verify_against_mir, NativeLoweringInput, NativeTarget,
};
use runmat_types::{
    BindingId, CapabilityRequirement, CapabilitySet, CollectiveId, DeoptimizationPointId,
    DynamicReason, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignRequirement, ForeignTypeIdentity, FunctionArgDefaultValue, FunctionArgDim,
    FunctionArgSizeSpec, FunctionArgValidator, InteropManifest, LabCount, ParallelAccess,
    ParallelManifest, ParallelRandomnessPolicy, ParallelRegionId, ParallelVariableContract,
    ParallelVariableRole, ParforContract, ProgramFunctionId, ProgramPointId, ProgramSourceId,
    ProgramSpan, RegionContract, RegionGuardCondition, RegionGuardContract, RegionGuardId,
    RegionId, RegionProvenance, RegionValueId, SequenceUse, Span, SpmdContract, SpmdLabRequirement,
    ValueFact, WasmInteropPolicy, INTEROP_MANIFEST_SCHEMA_VERSION,
    PARALLEL_MANIFEST_SCHEMA_VERSION, REGION_CONTRACT_SCHEMA_VERSION,
};

fn component_payloads(
    analysis: &runmat_mir::analysis::AnalysisStore,
) -> Vec<ExecutableComponentPayload> {
    ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            let bytes = if kind == ExecutableComponentKind::Analysis {
                analysis.canonical_bytes().unwrap()
            } else {
                format!("{kind:?}-r12").into_bytes()
            };
            ExecutableComponentPayload::new(kind, bytes).unwrap()
        })
        .collect()
}

fn manifest(analysis: &runmat_mir::analysis::AnalysisStore) -> ExecutableUnitManifest {
    let analysis_schema = analysis.revision.schema_version;
    let program = ProgramRevision::new(
        Digest::sha256(b"r12-graph"),
        Digest::sha256(b"r12-sources"),
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
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
    let components = component_payloads(analysis)
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
            entrypoint_function: Some(ProgramFunctionId(0)),
            entrypoint_kind: ExecutableEntrypointKind::Function,
            declaration_class: None,
            source_digest: Digest::sha256(b"r12-main.m"),
        },
        revisions,
        components,
        dependencies: Vec::new(),
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
        catch_scopes: Default::default(),
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
        class_method_owner: None,
        kind: FunctionKind::SyntheticEntrypoint,
        source_role: runmat_types::FunctionSourceRole::Synthetic,
        argument_validations: Vec::new(),
        captures: Vec::new(),
        modifiers: FunctionModifiers::default(),
        span,
    };
    MirAssembly {
        bodies: [(function, body)].into_iter().collect(),
        functions: [(function, metadata)].into_iter().collect(),
        classes: Vec::new(),
        class_property_defaults: Vec::new(),
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

fn sequence_assignment_mir() -> MirAssembly {
    let span = Span { start: 0, end: 2 };
    function(vec![
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::StructLiteral {
                    fields: vec![(
                        runmat_types::MemberName("field".into()),
                        MirOperand::Constant(MirConstant::Number("1".into())),
                    )],
                },
            },
            span,
        },
        MirStmt {
            kind: MirStmtKind::SequenceAssign {
                target: runmat_mir::MirSequenceTarget::Member {
                    base: MirPlace::Local(MirLocalId(0)),
                    member: runmat_types::MemberName("field".into()),
                },
                value: MirRvalue::Member {
                    base: MirOperand::Local(MirLocalId(0)),
                    member: runmat_types::MemberName("field".into()),
                    sequence_use: SequenceUse::SelectDestinationCardinality,
                },
            },
            span,
        },
    ])
}

#[test]
fn native_sequence_register_requires_one_adjacent_matching_pair() {
    let valid = lower(&sequence_assignment_mir());
    valid.verify().unwrap();
    let producer = valid.functions[0].blocks[0]
        .instructions
        .iter()
        .position(|instruction| {
            matches!(
                instruction.operation,
                runmat_native_codegen::NativeOperation::Rvalue {
                    result: runmat_native_codegen::NativeRvalueResult::SequenceAssignment(_),
                    ..
                }
            )
        })
        .unwrap();
    let consumer = producer + 1;

    let mut missing_producer = valid.clone();
    let runmat_native_codegen::NativeOperation::Rvalue { result, .. } =
        &mut missing_producer.functions[0].blocks[0].instructions[producer].operation
    else {
        panic!("expected sequence-register producer")
    };
    *result = runmat_native_codegen::NativeRvalueResult::Discard;
    assert_eq!(
        missing_producer.verify().unwrap_err().code,
        "native.ir.sequence_register"
    );

    let mut missing_consumer = valid.clone();
    missing_consumer.functions[0].blocks[0]
        .instructions
        .truncate(consumer);
    let duplicate_id = missing_consumer.functions[0].blocks[0].instructions[0].id;
    let dangling = missing_consumer.functions[0].blocks[0]
        .instructions
        .last_mut()
        .expect("sequence producer remains");
    dangling.id = duplicate_id;
    dangling.inputs = vec![runmat_native_codegen::NativeValueId(u32::MAX)];
    assert_eq!(
        missing_consumer.verify().unwrap_err().code,
        "native.ir.sequence_register"
    );

    let mut intervening = valid.clone();
    let unrelated = intervening.functions[0].blocks[0].instructions[0].clone();
    intervening.functions[0].blocks[0]
        .instructions
        .insert(consumer, unrelated);
    assert_eq!(
        intervening.verify().unwrap_err().code,
        "native.ir.sequence_register"
    );

    let mut mismatched = valid;
    let runmat_native_codegen::NativeOperation::Statement(MirStmtKind::SequenceAssign {
        target: runmat_mir::MirSequenceTarget::Member { member, .. },
        ..
    }) = &mut mismatched.functions[0].blocks[0].instructions[consumer].operation
    else {
        panic!("expected sequence-assignment consumer")
    };
    *member = runmat_types::MemberName("different".into());
    assert_eq!(
        mismatched.verify().unwrap_err().code,
        "native.ir.sequence_register"
    );
}

#[test]
fn native_ir_rejects_forged_lexical_catch_membership() {
    let mut assembly = lower(&function(vec![assignment(1)]));
    assembly.functions[0].blocks[0]
        .catch_scopes
        .push(runmat_mir::CatchScopeId(99));
    assert_eq!(assembly.verify().unwrap_err().code, "native.ir.catch_scope");
}

#[test]
fn native_ir_rejects_repeated_and_misparented_catch_membership() {
    let mut mir = function(vec![assignment(1)]);
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    let original_return = body.blocks[0].terminator.clone();
    body.blocks[0].terminator = MirTerminator {
        kind: MirTerminatorKind::TryCatch {
            try_block: BasicBlockId(1),
            catch_block: BasicBlockId(2),
            catch_binding: None,
        },
        span: original_return.span,
    };
    body.blocks.push(BasicBlock {
        id: BasicBlockId(1),
        statements: Vec::new(),
        terminator: original_return.clone(),
    });
    body.blocks.push(BasicBlock {
        id: BasicBlockId(2),
        statements: Vec::new(),
        terminator: original_return,
    });
    body.catch_scopes
        .insert(BasicBlockId(2), vec![runmat_mir::CatchScopeId(0)]);
    let valid = lower(&mir);
    valid.verify().unwrap();

    let mut repeated = valid.clone();
    repeated.functions[0].blocks[2]
        .catch_scopes
        .push(runmat_mir::CatchScopeId(0));
    assert_eq!(repeated.verify().unwrap_err().code, "native.ir.catch_scope");

    let mut misparented = valid;
    misparented.functions[0].blocks[1]
        .catch_scopes
        .push(runmat_mir::CatchScopeId(0));
    assert_eq!(
        misparented.verify().unwrap_err().code,
        "native.ir.catch_scope"
    );

    // The repeated scope is separated by a valid inner scope, so an adjacent
    // windows(2) duplicate check would have accepted the forged ancestry.
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    let inner_return = body.blocks[2].terminator.clone();
    body.blocks[2].terminator = MirTerminator {
        kind: MirTerminatorKind::TryCatch {
            try_block: BasicBlockId(3),
            catch_block: BasicBlockId(4),
            catch_binding: None,
        },
        span: inner_return.span,
    };
    body.blocks.push(BasicBlock {
        id: BasicBlockId(3),
        statements: Vec::new(),
        terminator: inner_return.clone(),
    });
    body.blocks.push(BasicBlock {
        id: BasicBlockId(4),
        statements: Vec::new(),
        terminator: inner_return,
    });
    body.catch_scopes
        .insert(BasicBlockId(3), vec![runmat_mir::CatchScopeId(0)]);
    body.catch_scopes.insert(
        BasicBlockId(4),
        vec![runmat_mir::CatchScopeId(0), runmat_mir::CatchScopeId(2)],
    );
    let mut nonadjacent = lower(&mir);
    nonadjacent.verify().unwrap();
    nonadjacent.functions[0].blocks[4]
        .catch_scopes
        .push(runmat_mir::CatchScopeId(0));
    assert_eq!(
        nonadjacent.verify().unwrap_err().code,
        "native.ir.catch_scope"
    );
}

fn lower(mir: &MirAssembly) -> runmat_native_codegen::NativeAssembly {
    let analysis = runmat_mir::analysis::analyze_assembly(mir);
    let manifest = manifest(&analysis);
    lower_with(mir, &analysis, &manifest).unwrap()
}

#[test]
fn dotted_statement_discard_has_no_ssa_result_while_expression_keeps_one() {
    let span = Span { start: 0, end: 8 };
    for (sequence_use, expected_outputs) in
        [(SequenceUse::Discard, 0), (SequenceUse::RequireSingle, 1)]
    {
        let mir = function(vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(0)),
                    value: MirRvalue::Use(MirOperand::Constant(MirConstant::Bool(true))),
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Expr(MirRvalue::SubscriptChain(runmat_mir::MirSubscriptChain {
                    root: MirOperand::Local(MirLocalId(0)),
                    steps: vec![runmat_mir::MirSubscriptStep::DottedInvoke {
                        member: runmat_types::MemberName("method".into()),
                        arguments: Vec::new(),
                        semantic_fallback: None,
                    }],
                    sequence_use,
                    context: runmat_types::ObjectIndexingContext::Statement,
                })),
                span,
            },
        ]);
        let native = lower(&mir);
        let instructions = &native.functions[0].blocks[0].instructions;
        let call = instructions
            .iter()
            .find(|instruction| {
                matches!(
                    &instruction.operation,
                    runmat_native_codegen::NativeOperation::Rvalue {
                        value: MirRvalue::SubscriptChain(_),
                        ..
                    }
                )
            })
            .expect("retained dotted invocation");
        assert_eq!(call.outputs.len(), expected_outputs);
        let statement = instructions
            .iter()
            .find(|instruction| {
                matches!(
                    &instruction.operation,
                    runmat_native_codegen::NativeOperation::Statement(MirStmtKind::Expr(_))
                )
            })
            .expect("retained expression statement");
        assert_eq!(statement.inputs.len(), expected_outputs);
    }
}

fn phased_fixed_destination_mir() -> MirAssembly {
    let span = Span { start: 0, end: 8 };
    let place = MirPlace::Index(
        Box::new(MirPlace::Local(MirLocalId(0))),
        runmat_mir::MirIndexing {
            argument_spans: None,
            kind: runmat_types::IndexKind::Paren,
            plan: runmat_mir::MirIndexPlan::Scalar,
            components: vec![runmat_mir::MirIndexComponent::Expansion(
                runmat_mir::MirCallArgumentRegion {
                    steps: Vec::new(),
                    argument: runmat_mir::MirCallArg::CapturedSequence(
                        runmat_mir::MirSequenceLocalId(0),
                    ),
                    span,
                },
            )],
            result_context: runmat_types::IndexResultContext::AssignmentTarget,
            cell_expand_all: false,
        },
    );
    let owner = runmat_mir::MirPreparedAssignment {
        source_statement: runmat_hir::StmtId(4),
        mutation: runmat_mir::MirPlaceMutation {
            creation_policy: place.assignment_creation_policy(false),
            place,
            kind: runmat_types::PlaceMutationKind::IndexedAssign,
            shape_policy: runmat_types::AssignmentShapePolicy::MatlabCompatible,
        },
        steps: Vec::new(),
    };
    let phase = |phase| MirStmt {
        kind: MirStmtKind::PrepareFixedAssignment(runmat_mir::MirFixedPreparation {
            assignment: owner.clone(),
            phase,
        }),
        span,
    };
    function(vec![
        assignment(1),
        phase(runmat_mir::MirFixedPreparationPhase::CaptureReceiver),
        phase(runmat_mir::MirFixedPreparationPhase::BeginIndex { path_step: 0 }),
        phase(runmat_mir::MirFixedPreparationPhase::EvaluateSelector {
            path_step: 0,
            component: Some(0),
            region: Box::new(runmat_mir::MirCallArgumentRegion {
                steps: Vec::new(),
                argument: runmat_mir::MirCallArg::Expansion(
                    runmat_mir::MirExpansionSource::Member {
                        base: MirOperand::Local(MirLocalId(0)),
                        member: runmat_types::MemberName("indices".into()),
                    },
                ),
                span,
            }),
            destination: runmat_mir::MirFixedSelectorDestination::Sequence(
                runmat_mir::MirSequenceLocalId(0),
            ),
        }),
        phase(runmat_mir::MirFixedPreparationPhase::FinishStep { path_step: 0 }),
        assignment(2),
        MirStmt {
            kind: MirStmtKind::PreparedAssign {
                preparation: owner.clone(),
                value: MirRvalue::Use(MirOperand::Local(MirLocalId(0))),
            },
            span,
        },
    ])
}

#[test]
fn fixed_destination_geometry_sums_live_owner_and_rejects_source_mismatch() {
    let native = lower(&phased_fixed_destination_mir());
    native.verify().unwrap();
    assert!(
        native.functions[0]
            .expression_continuation_geometry
            .retained_input_capacity
            >= 3
    );
    let mut underquoted = native.clone();
    underquoted.functions[0]
        .expression_continuation_geometry
        .retained_input_capacity -= 1;
    assert!(underquoted.verify().is_err());
    let mut mismatched = native;
    for instruction in &mut mismatched.functions[0].blocks[0].instructions {
        if let runmat_native_codegen::NativeOperation::Statement(MirStmtKind::PreparedAssign {
            preparation,
            ..
        }) = &mut instruction.operation
        {
            preparation.source_statement.0 += 1;
        }
    }
    assert!(mismatched.verify().is_err());
}

#[test]
fn fixed_selector_scalar_phase_defines_only_its_typed_frame_output() {
    let mut mir = phased_fixed_destination_mir();
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    body.locals.push(MirLocal {
        id: MirLocalId(1),
        binding: None,
        kind: MirLocalKind::Temporary,
        span: Span { start: 0, end: 8 },
    });
    // Canonical fixed preparation is selected for an expanding target. Keep
    // that CSL receipt and add the scalar selector beside it, rather than
    // forging a scalar-only target that the ordinary Assign owner handles.
    let mut scalar_phase = body.blocks[0].statements[3].clone();
    let MirStmtKind::PrepareFixedAssignment(phase) = &mut scalar_phase.kind else {
        unreachable!()
    };
    let runmat_mir::MirFixedPreparationPhase::EvaluateSelector {
        component,
        region,
        destination,
        ..
    } = &mut phase.phase
    else {
        unreachable!()
    };
    *component = Some(1);
    region.argument =
        runmat_mir::MirCallArg::Single(MirOperand::Constant(MirConstant::Number("1".into())));
    *destination = runmat_mir::MirFixedSelectorDestination::Scalar(MirLocalId(1));
    body.blocks[0].statements.insert(4, scalar_phase);
    for statement in &mut body.blocks[0].statements {
        let preparation = match &mut statement.kind {
            MirStmtKind::PrepareFixedAssignment(phase) => &mut phase.assignment,
            MirStmtKind::PreparedAssign { preparation, .. } => preparation,
            _ => continue,
        };
        let MirPlace::Index(_, indexing) = &mut preparation.mutation.place else {
            unreachable!()
        };
        indexing
            .components
            .push(runmat_mir::MirIndexComponent::Expr(MirOperand::Local(
                MirLocalId(1),
            )));
    }
    let native = lower(&mir);
    native.verify().unwrap();
    let mut scalar_outputs = 0;
    for instruction in &native.functions[0].blocks[0].instructions {
        if let runmat_native_codegen::NativeOperation::Statement(
            MirStmtKind::PrepareFixedAssignment(phase),
        ) = &instruction.operation
        {
            if matches!(
                phase.output(),
                Some(runmat_mir::MirFixedSelectorDestination::Scalar(_))
            ) {
                assert_eq!(instruction.outputs.len(), 1);
                assert_eq!(
                    instruction.outputs[0].local,
                    Some(runmat_native_codegen::NativeLocalId(1))
                );
                scalar_outputs += 1;
            } else {
                assert!(instruction.outputs.is_empty());
            }
        }
    }
    assert_eq!(scalar_outputs, 1);
}

#[test]
fn fixed_selector_receipt_and_phase_cursor_survive_normal_await() {
    let mut mir = phased_fixed_destination_mir();
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    let span = Span { start: 0, end: 8 };
    body.locals.push(MirLocal {
        id: MirLocalId(1),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    });
    body.locals.push(MirLocal {
        id: MirLocalId(2),
        binding: None,
        kind: MirLocalKind::Temporary,
        span,
    });
    let mut final_selector = body.blocks[0].statements[3].clone();
    let MirStmtKind::PrepareFixedAssignment(prefix) = &mut body.blocks[0].statements[3].kind else {
        unreachable!()
    };
    let source_assignment = prefix.assignment.clone();
    let runmat_mir::MirFixedPreparationPhase::EvaluateSelector { destination, .. } =
        &mut prefix.phase
    else {
        unreachable!()
    };
    *destination =
        runmat_mir::MirFixedSelectorDestination::Sequence(runmat_mir::MirSequenceLocalId(1));
    let MirStmtKind::PrepareFixedAssignment(final_phase) = &mut final_selector.kind else {
        unreachable!()
    };
    let runmat_mir::MirFixedPreparationPhase::EvaluateSelector { region, .. } =
        &mut final_phase.phase
    else {
        unreachable!()
    };
    region.argument = runmat_mir::MirCallArg::CapturedSequence(runmat_mir::MirSequenceLocalId(1));
    let mut rest = body.blocks[0].statements.split_off(4);
    rest.insert(0, final_selector);
    let terminal = body.blocks[0].terminator.clone();
    body.blocks[0].statements.push(MirStmt {
        kind: MirStmtKind::PrepareFixedAssignment(runmat_mir::MirFixedPreparation {
            assignment: source_assignment,
            phase: runmat_mir::MirFixedPreparationPhase::EvaluateSelector {
                path_step: 0,
                component: Some(0),
                region: Box::new(runmat_mir::MirCallArgumentRegion {
                    steps: vec![runmat_mir::MirExpressionStep::Let {
                        local: MirLocalId(2),
                        value: MirRvalue::Future {
                            function: FunctionId(0),
                            args: Vec::new(),
                            syntax: runmat_hir::CallSyntax::Plain,
                            requested_outputs: runmat_hir::RequestedOutputCount::One,
                        },
                        span,
                    }],
                    argument: runmat_mir::MirCallArg::Single(MirOperand::Local(MirLocalId(2))),
                    span,
                }),
                destination: runmat_mir::MirFixedSelectorDestination::Scalar(MirLocalId(1)),
            },
        }),
        span,
    });
    body.blocks[0].terminator = MirTerminator {
        kind: MirTerminatorKind::Await {
            future: MirOperand::Local(MirLocalId(1)),
            result: None,
            resume: BasicBlockId(1),
        },
        span,
    };
    body.blocks.push(BasicBlock {
        id: BasicBlockId(1),
        statements: rest,
        terminator: terminal,
    });
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(&analysis);
    manifest
        .capabilities
        .0
        .insert(CapabilityRequirement::ParallelRuntime);
    let native = lower_with(&mir, &analysis, &manifest).unwrap();
    native.verify().unwrap();
    let mut missing_finish = native;
    for instruction in &mut missing_finish.functions[0].blocks[1].instructions {
        if let runmat_native_codegen::NativeOperation::Statement(
            MirStmtKind::PrepareFixedAssignment(phase),
        ) = &mut instruction.operation
        {
            if matches!(
                phase.phase,
                runmat_mir::MirFixedPreparationPhase::FinishStep { .. }
            ) {
                phase.phase = runmat_mir::MirFixedPreparationPhase::BeginIndex { path_step: 0 };
            }
        }
    }
    assert!(missing_finish.verify().is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn relocatable_object_is_bound_to_the_exact_native_product() {
    let assembly = lower(&function(vec![assignment(1)]));
    let first = emit_relocatable_object(&assembly, NativeOptimization::Speed).unwrap();
    let second = emit_relocatable_object(&assembly, NativeOptimization::Speed).unwrap();

    first.validate().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.manifest.target, assembly.target);
    assert_eq!(
        first.manifest.object_format,
        runmat_native_codegen::aot::native_object_format_for_target(&assembly.target).unwrap()
    );
    assert_eq!(
        first.manifest.executable_cache_key,
        assembly.executable_cache_key
    );
    assert_eq!(first.manifest.native_cache_key, assembly.native_cache_key);
    assert_eq!(first.manifest.object_bytes, first.bytes.len() as u64);
    assert_eq!(first.manifest.functions.len(), 1);
    assert_eq!(first.manifest.functions[0].function, FunctionId(0));
    assert_eq!(first.manifest.entrypoint, FunctionId(0));
    assert_eq!(first.manifest.functions[0].symbol, "runmat_aot_entry");
    assert!(first.manifest.data.is_empty());
    assert_eq!(
        first.manifest.product,
        runmat_execution::host::NativeProductMode::HostUnselected
    );
    let mut untagged = serde_json::to_value(&first.manifest).unwrap();
    untagged.as_object_mut().unwrap().remove("product");
    assert!(
        serde_json::from_value::<runmat_native_codegen::aot::NativeObjectManifest>(untagged)
            .is_err()
    );

    let mut tampered = first;
    tampered.bytes.push(0);
    assert_eq!(
        tampered.validate().unwrap_err().code,
        "native.object.identity"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn relocatable_object_embeds_canonical_bounded_runtime_data() {
    let mut mir = function(vec![assignment(1)]);
    mir.functions.get_mut(&FunctionId(0)).unwrap().source = ProgramSourceId(0);
    let assembly = lower(&mir);
    let native_ir = assembly.canonical_bytes().unwrap();
    let text = "value = 1;";
    let source_map = runmat_execution::ExecutableSourceMapV2::new(vec![
        runmat_execution::ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "native-object-data-fixture".into(),
            relative_path: "main.m".into(),
            content_revision: Digest::sha256(text),
            display_name: "main.m".into(),
            text: text.into(),
        },
    ])
    .unwrap();
    let program = runmat_native_codegen::aot::AotProgramManifest::from_assembly(
        &assembly,
        Digest::sha256(&native_ir),
        None,
        source_map,
        runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
        Vec::new(),
    )
    .unwrap();
    let data = vec![
        NativeObjectData {
            symbol: "runmat_aot_program".into(),
            bytes: program.canonical_bytes().unwrap(),
            alignment: 1,
        },
        NativeObjectData {
            symbol: "runmat_aot_native_ir".into(),
            bytes: native_ir,
            alignment: 8,
        },
        NativeObjectData {
            symbol: "runmat_aot_resume_points".into(),
            bytes: b"resume-points".to_vec(),
            alignment: 8,
        },
        NativeObjectData {
            symbol: "runmat_aot_native_interfaces".into(),
            bytes: b"native-interfaces".to_vec(),
            alignment: 8,
        },
        NativeObjectData {
            symbol: "runmat_aot_mex_artifacts".into(),
            bytes: b"mex-artifacts".to_vec(),
            alignment: 8,
        },
        NativeObjectData {
            symbol: "runmat_aot_java_artifacts".into(),
            bytes: b"java-artifacts".to_vec(),
            alignment: 8,
        },
        NativeObjectData {
            symbol: "runmat_aot_python_artifacts".into(),
            bytes: b"python-artifacts".to_vec(),
            alignment: 8,
        },
    ];
    let object =
        emit_relocatable_object_with_data(&assembly, NativeOptimization::Speed, data).unwrap();
    object.validate().unwrap();
    assert_eq!(
        object
            .manifest
            .data
            .iter()
            .map(|item| item.symbol.as_str())
            .collect::<Vec<_>>(),
        [
            "runmat_aot_java_artifacts",
            "runmat_aot_mex_artifacts",
            "runmat_aot_native_interfaces",
            "runmat_aot_native_ir",
            "runmat_aot_program",
            "runmat_aot_python_artifacts",
            "runmat_aot_resume_points"
        ]
    );

    let duplicate = vec![
        NativeObjectData {
            symbol: "runmat_aot_program".into(),
            bytes: vec![1],
            alignment: 1,
        },
        NativeObjectData {
            symbol: "runmat_aot_program".into(),
            bytes: vec![2],
            alignment: 1,
        },
    ];
    assert_eq!(
        emit_relocatable_object_with_data(&assembly, NativeOptimization::None, duplicate)
            .unwrap_err()
            .code,
        "native.object.data_identity"
    );
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
        entrypoint: runmat_native_codegen::NativeEntrypointBinding::new(
            mir,
            *mir.entrypoints.first().expect("fixture entrypoint"),
            manifest.identity.entrypoint_function.unwrap(),
        )?,
        binding_names,
        target: NativeTarget::current(),
    })
}

#[test]
fn native_ir_binds_requirements_and_sites_to_admitted_analysis_contracts() {
    let mir = function(vec![
        assignment(1),
        MirStmt {
            kind: MirStmtKind::WorkspaceEffect {
                effect: WorkspaceEffect::CreatesBinding,
                bindings: vec![MirLocalId(0)],
            },
            span: Span { start: 2, end: 3 },
        },
    ]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let admitted_manifest = manifest(&analysis);
    let mut duplicated = analysis.clone();
    duplicated
        .operation_contracts
        .push(duplicated.operation_contracts[0].clone());
    let duplicated_manifest = manifest(&duplicated);
    assert_eq!(
        lower_with(&mir, &duplicated, &duplicated_manifest)
            .unwrap_err()
            .code,
        "native.lowering.operation_contract_coverage"
    );
    let original = lower_with(&mir, &analysis, &admitted_manifest).expect("valid Native IR");

    let mut omitted = original.clone();
    omitted.functions[0].operation_contracts.remove(0);
    assert_eq!(
        omitted.verify().unwrap_err().code,
        "native.ir.operation_contract"
    );

    let mut mutated = original.clone();
    mutated.functions[0].blocks[0].instructions[0]
        .effects
        .0
        .insert(runmat_types::EffectKind::Unknown);
    assert_eq!(
        mutated.verify().unwrap_err().code,
        "native.ir.operation_contract"
    );

    let mut substituted = original.clone();
    substituted.functions[0].blocks[0].instructions[0]
        .site
        .ordinal = 1;
    assert_eq!(
        substituted.verify().unwrap_err().code,
        "native.ir.operation_contract"
    );

    let mut self_consistent_forgery = original.clone();
    // Forge an already effectful site so its canonical safepoint and epoch
    // shape remain intact. Adding effects to a pure Use is structurally invalid
    // before the independent analysis-projection check can be exercised.
    let effectful = self_consistent_forgery.functions[0]
        .operation_contracts
        .iter()
        .position(|contract| !contract.effects.0.is_empty())
        .expect("fixture contains an admitted workspace-effect site");
    self_consistent_forgery.functions[0].operation_contracts[effectful]
        .effects
        .0
        .insert(runmat_types::EffectKind::Unknown);
    let forged_site = self_consistent_forgery.functions[0].operation_contracts[effectful]
        .site
        .clone();
    self_consistent_forgery.functions[0].blocks[0]
        .instructions
        .iter_mut()
        .find(|instruction| instruction.site == forged_site)
        .expect("operation instruction")
        .effects
        .0
        .insert(runmat_types::EffectKind::Unknown);
    self_consistent_forgery.analysis_projection = Digest::sha256(b"self-issued projection");
    self_consistent_forgery
        .verify()
        .expect("internally matching contracts are structurally valid");
    assert_eq!(
        verify_against_analysis(&self_consistent_forgery, &admitted_manifest, &analysis)
            .unwrap_err()
            .code,
        "native.ir.analysis_projection"
    );

    let mut foreign_analysis = original;
    foreign_analysis.analysis_component = Digest::sha256(b"self-issued-analysis");
    assert_eq!(
        verify_against_manifest(&foreign_analysis, &admitted_manifest)
            .unwrap_err()
            .code,
        "native.ir.manifest_binding"
    );
}

#[test]
fn native_class_declarations_reject_forged_parent_default_and_constructor_owner() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
    let original = lower_with(&mir, &analysis, &manifest).expect("valid classless Native IR");
    let identity = runmat_types::ClassIdentity::new("AotClass").expect("typed class");
    let mut forged = original.clone();
    forged
        .class_declarations
        .push(runmat_native_codegen::NativeClassDeclaration {
            class: identity.clone(),
            parent: Some(identity.clone()),
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: Vec::new(),
            events: Vec::new(),
            enumerations: Vec::new(),
            constructor: None,
        });
    assert_eq!(
        forged.verify().unwrap_err().code,
        "native.ir.super_constructor"
    );
    forged.class_declarations[0].parent = None;
    forged.class_declarations[0]
        .properties
        .push(runmat_native_codegen::NativeClassProperty {
            name: runmat_types::MemberName::from("count"),
            attributes: runmat_types::PropertyAttributes::default(),
            has_default: false,
            default_literal: Some(runmat_native_codegen::NativeClassDefaultLiteral::Num(1.0)),
        });
    assert_eq!(
        forged.verify().unwrap_err().code,
        "native.ir.super_constructor"
    );
    forged.class_declarations[0].properties.clear();
    forged.class_declarations[0].constructor =
        Some(runmat_native_codegen::NativeConstructorMethod {
            function: original.functions[0].id,
            access: runmat_types::MemberAccess::Public,
        });
    assert_eq!(
        forged.verify().unwrap_err().code,
        "native.ir.super_constructor"
    );
    forged.class_declarations[0].constructor = None;
    forged
        .verify()
        .expect("well-formed extra class is structurally valid");
    assert_eq!(
        verify_against_mir(&forged, &mir, None).unwrap_err().code,
        "native.ir.mir_class_declarations"
    );
    let parent = runmat_types::ClassIdentity::new("AotParent").expect("parent");
    forged.class_declarations[0].parent = Some(parent.clone());
    forged
        .verify()
        .expect("unused open companion is structurally valid in generic native");
    forged
        .class_declarations
        .push(runmat_native_codegen::NativeClassDeclaration {
            class: parent,
            parent: Some(identity),
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods: Vec::new(),
            events: Vec::new(),
            enumerations: Vec::new(),
            constructor: None,
        });
    assert_eq!(
        forged.verify().unwrap_err().code,
        "native.ir.super_constructor"
    );
}

#[test]
fn generic_native_retains_unused_super_call_without_closing_external_parent() {
    let mut mir = function(vec![assignment(1)]);
    let class = runmat_types::ClassIdentity::new("UnusedCompanion").expect("class");
    let parent = runmat_types::ClassIdentity::new("AbsentCompanionBase").expect("parent");
    let constructor = FunctionId(1);
    let method = runmat_types::MethodName::from("UnusedCompanion");
    let mut body = mir.bodies[&FunctionId(0)].clone();
    body.function = constructor;
    body.blocks[0].statements[0].kind = MirStmtKind::Assign {
        place: MirPlace::Local(MirLocalId(0)),
        value: MirRvalue::Call(runmat_mir::MirCall {
            callee: runmat_mir::MirCallee::SuperConstructor {
                current_class: class.clone(),
                super_class: parent.clone(),
            },
            args: Vec::new(),
            arg_spans: Vec::new(),
            syntax: runmat_hir::CallSyntax::Plain,
            requested_outputs: runmat_hir::RequestedOutputCount::One,
            fallback_policy: runmat_hir::CallableFallbackPolicy::None,
            workspace_first_name: None,
            bare_identifier: false,
            async_behavior: runmat_mir::AsyncBehaviorFact::NeverSuspends,
            effects: runmat_builtins::BuiltinEffects::none(),
            workspace_effect: None,
            environment_effect: None,
            purity: runmat_builtins::BuiltinPurity::Pure,
            semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
        }),
    };
    mir.bodies.insert(constructor, body);
    let mut metadata = mir.functions[&FunctionId(0)].clone();
    metadata.name = FunctionName("UnusedCompanion".into());
    metadata.enclosing_class = Some(runmat_types::ClassId(0));
    metadata.class_method_owner = Some(runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: method.clone(),
        is_static: false,
        role: runmat_types::ClassMethodRole::Constructor,
    });
    metadata.kind = FunctionKind::ClassMethod {
        is_static: false,
        role: runmat_types::ClassMethodRole::Constructor,
    };
    metadata.source_role = runmat_types::FunctionSourceRole::Method;
    mir.functions.insert(constructor, metadata);
    mir.classes.push(runmat_types::ClassDeclaration {
        id: runmat_types::ClassId(0),
        source: mir.functions[&constructor].source,
        name: runmat_types::QualifiedName(vec![runmat_types::SymbolName("UnusedCompanion".into())]),
        inheritance: runmat_types::InheritanceDeclaration {
            declared_super_class: Some("AbsentCompanionBase".into()),
            ..Default::default()
        },
        kind: runmat_types::ClassKind::Value,
        is_sealed: false,
        is_abstract: false,
        declared_attributes: Vec::new(),
        properties: Vec::new(),
        methods: vec![runmat_types::MethodDeclaration {
            role: runmat_types::ClassMethodRole::Constructor,
            function: constructor,
            name: method,
            is_static: false,
            attributes: Default::default(),
            declared_attributes: Vec::new(),
            span: Span { start: 0, end: 8 },
        }],
        events: Vec::new(),
        enumerations: Vec::new(),
        span: Span { start: 0, end: 8 },
    });
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut executable = manifest(&analysis);
    for function in &analysis.functions {
        executable
            .capabilities
            .0
            .extend(function.capabilities.0.iter().cloned());
    }
    let assembly = lower_with(&mir, &analysis, &executable)
        .expect("unused class does not close an external parent in generic native");
    assert_eq!(assembly.functions.len(), 2);
    assert!(assembly.functions.iter().any(|function| {
        function.class_method_owner.as_ref().is_some_and(|owner| {
            owner.declaring_class == class
                && owner.role == runmat_types::ClassMethodRole::Constructor
        }) && function.super_constructor_calls() == vec![(class.clone(), parent.clone())]
    }));
}

#[test]
fn native_lowering_rejects_compiler_one_with_current_mir_and_analysis() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut executable = manifest(&analysis);
    let current = &executable.identity.program;
    executable.identity.program = ProgramRevision::new(
        *current.graph_digest(),
        *current.source_digest(),
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            1,
            *current.runtime_fingerprint(),
            *current.catalog_fingerprint(),
            current.compatibility_mode().as_str(),
        )
        .unwrap(),
    )
    .unwrap();
    let error = lower_with(&mir, &analysis, &executable).unwrap_err();
    assert_eq!(error.code, "native.lowering.compiler_compatibility");
    assert!(error
        .message
        .contains("compiler schema actual 1 expected 2"));
}

#[test]
fn bound_locals_require_and_retain_canonical_semantic_names() {
    let mut mir = function(vec![assignment(1)]);
    let binding = BindingId(4);
    mir.bodies.get_mut(&FunctionId(0)).unwrap().locals[0].binding = Some(binding);
    mir.bodies.get_mut(&FunctionId(0)).unwrap().locals[0].kind = MirLocalKind::Binding;
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);

    let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
    assert_eq!(error.code, "native.lowering.binding_name");

    let names = std::collections::BTreeMap::from([(binding, "answer".to_string())]);
    let assembly = lower_with_bindings(&mir, &analysis, &manifest, Some(&names)).unwrap();
    let local = &assembly.functions[0].locals[0];
    assert_eq!(local.id, runmat_native_codegen::NativeLocalId(0));
    assert_eq!(local.binding, Some(binding));
    assert_eq!(local.name.as_deref(), Some("answer"));
    assert_eq!(local.kind, runmat_native_codegen::NativeLocalKind::Binding);
    assert_eq!(
        assembly.schema_version,
        runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION
    );
    assembly.verify().unwrap();
    verify_against_mir(&assembly, &mir, Some(&names)).unwrap();

    let mut forged_contract = assembly.clone();
    forged_contract.functions[0].spawn_safety = runmat_types::SpawnSafetyFact::NotSpawnSafe {
        reason: runmat_types::SpawnSafetyReason::UnknownDynamicCapture,
    };
    assert_eq!(
        verify_against_mir(&forged_contract, &mir, Some(&names))
            .unwrap_err()
            .code,
        "native.ir.mir_spawn_contract"
    );

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
fn selected_entrypoint_preserves_the_complete_mir_entrypoint_inventory() {
    let mut mir = function(vec![assignment(1)]);
    let mut second_body = mir.bodies[&FunctionId(0)].clone();
    second_body.function = FunctionId(1);
    let mut second_metadata = mir.functions[&FunctionId(0)].clone();
    second_metadata.name = FunctionName("second".into());
    mir.bodies.insert(FunctionId(1), second_body);
    mir.functions.insert(FunctionId(1), second_metadata);
    mir.entrypoints.push(FunctionId(1));
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);

    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assert_eq!(assembly.entrypoints, vec![FunctionId(0), FunctionId(1)]);
    assert_eq!(assembly.selected_entrypoint, FunctionId(0));
    assert_eq!(
        assembly.selected_portable_entrypoint,
        manifest.identity.entrypoint_function.unwrap()
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
    let manifest = manifest(&analysis);
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
fn selector_mask_limit_is_rejected_before_native_execution() {
    let span = Span { start: 0, end: 8 };
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(0)),
            value: MirRvalue::Index {
                base: MirOperand::Local(MirLocalId(0)),
                indexing: runmat_mir::MirIndexing {
                    argument_spans: None,
                    kind: runmat_types::IndexKind::Paren,
                    plan: runmat_mir::MirIndexPlan::Slice,
                    components: vec![runmat_mir::MirIndexComponent::Colon; 33],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
        },
        span,
    }]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let native_manifest = manifest(&analysis);
    assert_eq!(
        lower_with(&mir, &analysis, &native_manifest)
            .unwrap_err()
            .code,
        "native.capability.selector_dimension_limit"
    );

    let sequence = runmat_mir::MirSequenceLocalId(0);
    let mir = function(vec![
        MirStmt {
            kind: MirStmtKind::CaptureSequence {
                destination: sequence,
                source: runmat_mir::MirExpansionSource::CellContents {
                    base: MirOperand::Constant(MirConstant::EmptyArray),
                    indexing: runmat_mir::MirIndexing {
                        argument_spans: None,
                        kind: runmat_types::IndexKind::Brace,
                        plan: runmat_mir::MirIndexPlan::Cell,
                        components: vec![runmat_mir::MirIndexComponent::Colon; 33],
                        result_context: runmat_types::IndexResultContext::ReadCommaList,
                        cell_expand_all: true,
                    },
                },
            },
            span,
        },
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Aggregate {
                    kind: runmat_mir::MirAggregateKind::Cell,
                    row_lengths: vec![1],
                    elements: vec![runmat_mir::MirAggregateElement::CapturedSequence(sequence)],
                },
            },
            span,
        },
    ]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let native_manifest = manifest(&analysis);
    assert_eq!(
        lower_with(&mir, &analysis, &native_manifest)
            .unwrap_err()
            .code,
        "native.capability.selector_dimension_limit"
    );
}

#[test]
fn expression_continuation_geometry_counts_constant_statement_selectors_only_when_needed() {
    let span = Span { start: 0, end: 8 };
    let pure = lower(&function(vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(0)),
            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Bool(true))),
        },
        span,
    }]));
    assert_eq!(
        pure.functions[0]
            .expression_continuation_geometry
            .region_capacity,
        0
    );
    assert_eq!(
        pure.functions[0]
            .expression_continuation_geometry
            .retained_input_capacity,
        0
    );
    pure.verify().unwrap();

    let prepared = runmat_mir::MirPreparedExpressionRegion::from_lowered(
        Vec::new(),
        MirOperand::Constant(MirConstant::Bool(true)),
    )
    .unwrap();
    let mut components =
        vec![
            runmat_mir::MirIndexComponent::Expr(MirOperand::Constant(MirConstant::Bool(true)));
            20
        ];
    components.push(runmat_mir::MirIndexComponent::PreparedExpr(prepared));
    let sequence = runmat_mir::MirSequenceLocalId(0);
    let statement = MirStmt {
        kind: MirStmtKind::CaptureSequence {
            destination: sequence,
            source: runmat_mir::MirExpansionSource::CellContents {
                base: MirOperand::Constant(MirConstant::EmptyArray),
                indexing: runmat_mir::MirIndexing {
                    argument_spans: None,
                    kind: runmat_types::IndexKind::Brace,
                    plan: runmat_mir::MirIndexPlan::Cell,
                    components,
                    result_context: runmat_types::IndexResultContext::ReadCommaList,
                    cell_expand_all: false,
                },
            },
        },
        span,
    };
    let mut native = lower(&function(vec![
        statement,
        MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Aggregate {
                    kind: runmat_mir::MirAggregateKind::Cell,
                    row_lengths: vec![1],
                    elements: vec![runmat_mir::MirAggregateElement::CapturedSequence(sequence)],
                },
            },
            span,
        },
    ]));
    let geometry = native.functions[0].expression_continuation_geometry;
    assert!(geometry.region_capacity > 0);
    assert!(geometry.retained_input_capacity >= 23);
    native.verify().unwrap();
    native.functions[0]
        .expression_continuation_geometry
        .retained_input_capacity -= 1;
    assert_eq!(
        native.verify().unwrap_err().code,
        "native.ir.expression_continuation_geometry"
    );
}

#[test]
fn expression_continuation_geometry_counts_nested_dotted_argument_cursors() {
    let span = Span { start: 0, end: 8 };
    let call_argument =
        runmat_mir::MirDottedInvokeArgument::Call(runmat_mir::MirCallArgumentRegion {
            steps: Vec::new(),
            argument: runmat_mir::MirCallArg::Single(MirOperand::Constant(MirConstant::Bool(true))),
            span,
        });
    let prepared = runmat_mir::MirPreparedExpressionRegion::from_lowered(
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(1)),
                value: MirRvalue::SubscriptChain(runmat_mir::MirSubscriptChain {
                    root: MirOperand::Constant(MirConstant::Bool(true)),
                    steps: vec![runmat_mir::MirSubscriptStep::DottedInvoke {
                        member: runmat_types::MemberName("method".into()),
                        arguments: vec![call_argument; 3],
                        semantic_fallback: None,
                    }],
                    sequence_use: SequenceUse::RequireSingle,
                    context: runmat_types::ObjectIndexingContext::Expression,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(1)),
    )
    .unwrap();
    let mut mir = function(vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(0)),
            value: MirRvalue::Index {
                base: MirOperand::Constant(MirConstant::EmptyArray),
                indexing: runmat_mir::MirIndexing {
                    argument_spans: None,
                    kind: runmat_types::IndexKind::Paren,
                    plan: runmat_mir::MirIndexPlan::Scalar,
                    components: vec![runmat_mir::MirIndexComponent::PreparedExpr(prepared)],
                    result_context: runmat_types::IndexResultContext::ReadSingle,
                    cell_expand_all: false,
                },
            },
        },
        span,
    }]);
    mir.bodies
        .get_mut(&FunctionId(0))
        .unwrap()
        .locals
        .push(MirLocal {
            id: MirLocalId(1),
            binding: None,
            kind: MirLocalKind::Temporary,
            span,
        });
    let native = lower(&mir);
    let geometry = native.functions[0].expression_continuation_geometry;
    // The outer PreparedExpr and all three zero-step dotted Call arguments
    // have distinct continuation entries, despite sharing one chain step.
    assert!(geometry.region_capacity >= 4);
    native.verify().unwrap();
}

#[test]
fn legacy_binding_places_are_rejected_before_native_execution() {
    let span = Span { start: 0, end: 8 };
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Binding(BindingId(99)),
            value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("1".into()))),
        },
        span,
    }]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
    assert_eq!(
        lower_with(&mir, &analysis, &manifest).unwrap_err().code,
        "native.ir.legacy_binding_place"
    );
}

#[test]
fn unscoped_end_is_rejected_before_native_lowering() {
    let span = Span { start: 0, end: 3 };
    let mir = function(vec![MirStmt {
        kind: MirStmtKind::Assign {
            place: MirPlace::Local(MirLocalId(0)),
            value: MirRvalue::End,
        },
        span,
    }]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
    let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
    assert_eq!(error.code, "native.lowering.expression_region");
    assert!(error.message.contains("not enclosed"), "{error:?}");
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
    let liveness = analyze_liveness(
        &assembly,
        runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
    )
    .unwrap();
    assert!(liveness.get().function(FunctionId(0)).is_some());
    assert_liveness_cfg_and_retained_admission();
}

// One canonical analysis family covers loop/diamond transfer, escaped return
// values and the distinction between retained reads and fresh admission.
fn assert_liveness_cfg_and_retained_admission() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use runmat_native_codegen::{NativeLivenessAdmission, NativeLivenessPlan};
    use runmat_runtime::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let span = Span { start: 0, end: 8 };
    let mut mir = function(vec![]);
    let block = |id, statements, kind| BasicBlock {
        id: BasicBlockId(id),
        statements,
        terminator: MirTerminator { kind, span },
    };
    mir.bodies.get_mut(&FunctionId(0)).unwrap().blocks = vec![
        block(
            0,
            vec![assignment(1)],
            MirTerminatorKind::Branch {
                cond: MirOperand::Local(MirLocalId(0)),
                then_block: BasicBlockId(1),
                else_block: BasicBlockId(2),
            },
        ),
        block(
            1,
            vec![assignment(2)],
            MirTerminatorKind::Goto(BasicBlockId(3)),
        ),
        block(
            2,
            vec![assignment(3)],
            MirTerminatorKind::Goto(BasicBlockId(3)),
        ),
        block(
            3,
            vec![],
            MirTerminatorKind::Branch {
                cond: MirOperand::Local(MirLocalId(0)),
                then_block: BasicBlockId(1),
                else_block: BasicBlockId(4),
            },
        ),
        block(
            4,
            vec![],
            MirTerminatorKind::Return(vec![MirOperand::Local(MirLocalId(0))]),
        ),
    ];
    let assembly = lower(&mir);
    let plan = NativeLivenessPlan::for_assembly(&assembly).unwrap();
    let ledger = ManagedHostEffectLedger::new(1_048_576);
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let admission = NativeLivenessAdmission::capture(&runtime).unwrap();
    let liveness = analyze_liveness(&assembly, admission.clone()).unwrap();
    let charged = ledger.snapshot().live_claimed_bytes;
    assert!(charged >= plan.retained_bytes as u64);
    assert!(
        ledger.snapshot().peak_claimed_bytes >= (plan.retained_bytes + plan.workspace_bytes) as u64
    );
    let function = &assembly.functions[0];
    for block in function
        .blocks
        .iter()
        .filter(|block| block.id.0 == 3 || block.id.0 == 4)
    {
        let parameter = block
            .parameters
            .iter()
            .find(|parameter| parameter.local.0 == 0)
            .unwrap();
        assert!(liveness
            .get()
            .live_on_entry(function.id, block.id, parameter.value));
    }
    for predecessor in function
        .blocks
        .iter()
        .filter(|block| block.id.0 == 1 || block.id.0 == 2)
    {
        let runmat_native_codegen::NativeTerminatorKind::Goto { edge } =
            &predecessor.terminator.kind
        else {
            panic!("diamond predecessor")
        };
        for argument in &edge.arguments {
            if let runmat_native_codegen::NativeEdgeArgument::Value(value) = argument {
                for instruction in &predecessor.instructions {
                    assert!(
                        !liveness
                            .get()
                            .retired_after(function.id, &instruction.site)
                            .contains(value),
                        "diamond edge value retired early"
                    );
                }
            }
        }
    }
    let returned = function
        .blocks
        .iter()
        .find(|block| block.id.0 == 4)
        .unwrap();
    let runmat_native_codegen::NativeTerminatorKind::Return { values } = &returned.terminator.kind
    else {
        panic!("return block")
    };
    for value in values {
        for instruction in &returned.instructions {
            assert!(!liveness
                .get()
                .retired_after(function.id, &instruction.site)
                .contains(value));
        }
    }
    let alias = liveness.clone();
    cancellation.cancel(CancellationReason::User);
    let fresh_runtime = runtime.with_cancellation_token(CancellationSource::new().token());
    assert!(NativeLivenessAdmission::capture(&fresh_runtime).is_ok());
    // Immutable cache metadata remains readable across invocation generations.
    assert!(alias.get().function(function.id).is_some());
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    let failure = analyze_liveness(&assembly, admission)
        .err()
        .expect("stale owner cannot allocate");
    assert_eq!(
        failure.allocation,
        Some(runmat_runtime::context::host_allocation::HostAllocationError::Cancelled)
    );
    drop(liveness);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let tiny = ManagedHostEffectLedger::new(1);
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(tiny.clone()))),
    );
    assert!(analyze_liveness(
        &assembly,
        NativeLivenessAdmission::capture(&runtime).unwrap()
    )
    .is_err());
    assert_eq!(tiny.snapshot().live_claimed_bytes, 0);
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
                1 | 2 => {
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

    log.0.clear();
    resume.block = 2;
    exit = NativeExit::completed(0);
    // SAFETY: the same validated ABI records remain live, and block 2 is a
    // verified resume target in this function.
    let resumed_status = unsafe { entry(&mut call, &mut exit) };
    assert_eq!(resumed_status, NativeHostStatus::OK);
    call.validate_exit(&exit).unwrap();
    assert!(!log.0.iter().any(|site| site.block == 0));
    assert!(!log.0.iter().any(|site| site.block == 1));
    assert!(log.0.iter().any(|site| site.block == 2));

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
        "native.ir.operation_contract_coverage"
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
    let expected = match &mut wrong_abi.target {
        NativeTarget::Native { probe, .. } => {
            probe.abi_encoded_version += 1;
            "native.target.abi"
        }
        NativeTarget::PortableWasm32 { schema_version } => {
            *schema_version += 1;
            "native.target.wasm_schema"
        }
        NativeTarget::UnavailableHost => panic!("lowered fixture must have an admissible target"),
    };
    assert_eq!(wrong_abi.verify().unwrap_err().code, expected);
}

#[test]
fn native_cache_identity_binds_target_and_abi() {
    let assembly = lower(&function(vec![assignment(1)]));
    assert_ne!(assembly.native_cache_key, assembly.executable_cache_key);
    let mut tampered = assembly.clone();
    tampered.native_cache_key = Digest::sha256(b"wrong-native-key");
    assert_eq!(tampered.verify().unwrap_err().code, "native.ir.cache_key");
    let mut wrong_selected = assembly.clone();
    wrong_selected.selected_entrypoint =
        FunctionId(wrong_selected.selected_entrypoint.0.saturating_add(1));
    assert_eq!(
        wrong_selected.verify().unwrap_err().code,
        "native.ir.cache_key",
        "the native cache key must bind the selected compiled entrypoint"
    );

    let mut cross_target = assembly.target;
    let expected = match &mut cross_target {
        NativeTarget::Native { target, .. } => {
            target.target.architecture = match target.target.architecture {
                runmat_execution::host::NativeArchitecture::X86_64 => {
                    runmat_execution::host::NativeArchitecture::Aarch64
                }
                runmat_execution::host::NativeArchitecture::Aarch64 => {
                    runmat_execution::host::NativeArchitecture::X86_64
                }
            };
            "native.target.contract"
        }
        NativeTarget::PortableWasm32 { schema_version } => {
            *schema_version += 1;
            "native.target.wasm_schema"
        }
        NativeTarget::UnavailableHost => panic!("lowered fixture must have an admissible target"),
    };
    assert_eq!(cross_target.validate().unwrap_err().code, expected);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn selected_baseline_objects_and_portable_wasm_have_distinct_admission_domains() {
    use runmat_execution::host::{
        NativeArchitecture, NativeCpuBaseline, NativeEnvironment, NativeOperatingSystem,
    };
    let host_isa_rejection =
        |target: &NativeTarget| match runmat_native_codegen::cranelift::compilation_isa(
            target,
            runmat_native_codegen::cranelift::IsaCompilationPurpose::ExactHostInProcessJit,
            cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        ) {
            Err(error) => error.code,
            Ok(_) => panic!("inadmissible target obtained a host-tuned ISA"),
        };
    let original = lower(&function(vec![assignment(1)]));
    original.target.require_host_executable().unwrap();
    let tuned = runmat_native_codegen::cranelift::compilation_isa(
        &original.target,
        runmat_native_codegen::cranelift::IsaCompilationPurpose::ExactHostInProcessJit,
        cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
    )
    .unwrap();
    let host = cranelift_native::builder()
        .unwrap()
        .finish(cranelift_codegen::settings::Flags::new(
            cranelift_codegen::settings::builder(),
        ))
        .unwrap();
    assert_eq!(tuned.triple(), host.triple());
    assert_eq!(tuned.isa_flags().iter().map(|flag| (flag.name, flag.value_string())).collect::<Vec<_>>(),
        host.isa_flags().iter().map(|flag| (flag.name, flag.value_string())).collect::<Vec<_>>(),
        "process-local JIT preserves canonical host feature detection rather than forcing portable baseline flags");
    for (architecture, baseline) in [
        (NativeArchitecture::X86_64, NativeCpuBaseline::X86_64V1),
        (NativeArchitecture::Aarch64, NativeCpuBaseline::ArmV8A),
    ] {
        // Synthetic records exercise encoding/lowering, not target-built probe
        // provenance, deployed ABI compatibility, or foreign execution.
        let mut probe = runmat_runtime::native::native_abi_probe();
        probe.architecture = architecture;
        probe.operating_system = NativeOperatingSystem::Linux;
        probe.environment = NativeEnvironment::Gnu;
        let mut assembly = original.clone();
        assembly.target = NativeTarget::from_probe(probe, baseline).unwrap();
        assembly.native_cache_key = assembly
            .target
            .cache_key(
                &assembly.executable_cache_key,
                assembly.selected_entrypoint,
                &assembly.selected_entrypoint_body,
            )
            .unwrap();
        assembly.verify().unwrap();
        let isa = runmat_native_codegen::cranelift::compilation_isa(
            &assembly.target,
            runmat_native_codegen::cranelift::IsaCompilationPurpose::SelectedBaselineObject,
            cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder()),
        )
        .unwrap();
        assert!(isa
            .isa_flags()
            .iter()
            .all(|flag| flag.as_bool() != Some(true)));
        assert_eq!(
            isa.triple().to_string(),
            assembly.target.native_parts().unwrap().0.triple().unwrap()
        );
        let object = emit_relocatable_object(&assembly, NativeOptimization::Speed).unwrap();
        assert!(!object.bytes.is_empty());
        assert_eq!(object.manifest.target, assembly.target);
        assert_eq!(
            runmat_native_codegen::NativeAssembly::from_canonical_bytes(
                &assembly.canonical_bytes().unwrap()
            )
            .unwrap(),
            assembly
        );
        if assembly.target != original.target {
            assert_eq!(
                assembly.target.require_host_executable().unwrap_err().code,
                "native.target.host_execution"
            );
            assert_eq!(
                host_isa_rejection(&assembly.target),
                "native.target.host_execution",
                "foreign target must reject at the ISA owner before host feature detection"
            );
        }
    }
    let mut portable = original;
    let mut shifted_probe = runmat_runtime::native::native_abi_probe();
    shifted_probe.offsets.swap(0, 1);
    let shifted = NativeTarget::from_probe(
        shifted_probe,
        portable.target.native_parts().unwrap().0.cpu_baseline,
    )
    .unwrap();
    shifted.validate().unwrap();
    assert_eq!(
        shifted.require_host_executable().unwrap_err().code,
        "native.target.host_execution",
        "self-consistent changed layout metadata is not the actual host ABI"
    );
    portable.target = NativeTarget::PortableWasm32 {
        schema_version: runmat_native_codegen::PORTABLE_WASM_LOWERING_SCHEMA_VERSION,
    };
    portable.native_cache_key = portable
        .target
        .cache_key(
            &portable.executable_cache_key,
            portable.selected_entrypoint,
            &portable.selected_entrypoint_body,
        )
        .unwrap();
    portable.verify().unwrap();
    assert_eq!(
        runmat_native_codegen::NativeAssembly::from_canonical_bytes(
            &portable.canonical_bytes().unwrap()
        )
        .unwrap(),
        portable
    );
    assert_eq!(
        portable.target.require_host_executable().unwrap_err().code,
        "native.target.native_required"
    );
    assert_eq!(
        emit_relocatable_object(&portable, NativeOptimization::Speed)
            .unwrap_err()
            .code,
        "native.target.native_required"
    );
    let old = serde_json::json!({"architecture": "x86_64", "operating_system": "linux", "pointer_width": 64, "abi": runmat_native_codegen::abi::current_native_abi_binding()});
    assert!(
        serde_json::from_value::<NativeTarget>(old).is_err(),
        "untagged prior target shape must not acquire a domain by default"
    );
    assert_eq!(
        NativeTarget::UnavailableHost.validate().unwrap_err().code,
        "native.target.host_unavailable"
    );
    assert_eq!(
        host_isa_rejection(&NativeTarget::UnavailableHost),
        "native.target.host_unavailable"
    );
}

#[test]
fn executable_manifest_binding_rejects_a_coordinated_cache_key_substitution() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
    let mut assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assembly.executable_cache_key = Digest::sha256(b"substituted-executable");
    assembly.native_cache_key = assembly
        .target
        .cache_key(
            &assembly.executable_cache_key,
            assembly.selected_entrypoint,
            &assembly.selected_entrypoint_body,
        )
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
    assembly.functions[0].operation_contracts.drain(0..2);
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
    let mut manifest = manifest(&analysis);
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
        value: MirOperand::Constant(MirConstant::Number("1".into())),
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
        let manifest = manifest(&analysis);
        let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
        assert_eq!(error.code, "native.capability.distributed_core_pending");
        assert_eq!(
            error.construct.unwrap().native_lowering_class(),
            runmat_mir::NativeLoweringClass::CapabilityRejection
        );
    }
}

fn executor_builtin_call(name: &str) -> MirRvalue {
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical builtin");
    let semantics = entry.legacy_semantics();
    MirRvalue::Call(runmat_mir::MirCall {
        callee: runmat_mir::MirCallee::Static(runmat_types::CallableIdentity::Builtin(
            runmat_types::BuiltinId(entry.identity.name.into()),
        )),
        args: Vec::new(),
        arg_spans: Vec::new(),
        syntax: runmat_types::CallSyntax::Plain,
        requested_outputs: runmat_types::RequestedOutputCount::One,
        fallback_policy: runmat_types::CallableFallbackPolicy::None,
        workspace_first_name: None,
        bare_identifier: false,
        async_behavior: match semantics.async_behavior {
            runmat_builtins::BuiltinAsyncBehavior::NeverSuspends => {
                runmat_mir::AsyncBehaviorFact::NeverSuspends
            }
            runmat_builtins::BuiltinAsyncBehavior::MaySuspend
            | runmat_builtins::BuiltinAsyncBehavior::MaySuspendLocally => {
                runmat_mir::AsyncBehaviorFact::MaySuspend
            }
            runmat_builtins::BuiltinAsyncBehavior::RequiresAsyncRuntime => {
                runmat_mir::AsyncBehaviorFact::RequiresAsyncRuntime
            }
        },
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    })
}

#[test]
fn executor_only_builtin_contracts_reject_before_native_publication() {
    for name in ["parfeval", "parfevalOnAll", "fetchOutputs", "fetchNext"] {
        let value = executor_builtin_call(name);
        let mir = function(vec![MirStmt {
            kind: MirStmtKind::Expr(value.clone()),
            span: Span { start: 0, end: 1 },
        }]);
        let analysis = runmat_mir::analysis::analyze_assembly(&mir);
        let manifest = manifest(&analysis);
        let error = lower_with(&mir, &analysis, &manifest).unwrap_err();
        assert_eq!(error.code, "native.capability.executor_intrinsic");
        assert_eq!(
            error.construct,
            Some(runmat_mir::MirConstructKind::ExecutorIntrinsicCall)
        );
        // Even copied metadata cannot turn a runtime-resolved/user target
        // into the canonical builtin operation by matching its spelling.
        let MirRvalue::Call(mut shadowed) = value else {
            unreachable!()
        };
        for callee in [
            runmat_mir::MirCallee::Static(runmat_types::CallableIdentity::DynamicName(
                runmat_types::SymbolName(name.into()),
            )),
            runmat_mir::MirCallee::Static(runmat_types::CallableIdentity::BoundFunction(
                runmat_types::FunctionId(1),
            )),
        ] {
            shadowed.callee = callee;
            assert!(shadowed.executor_intrinsic().is_none());
            assert_eq!(
                runmat_mir::rvalue_construct_kind(&MirRvalue::Call(shadowed.clone())),
                runmat_mir::MirConstructKind::Call
            );
        }
    }
    for name in ["parpool", "gcp"] {
        let value = executor_builtin_call(name);
        assert_eq!(
            runmat_mir::rvalue_construct_kind(&value),
            runmat_mir::MirConstructKind::Call
        );
        let mir = function(vec![MirStmt {
            kind: MirStmtKind::Expr(value),
            span: Span { start: 0, end: 1 },
        }]);
        let analysis = runmat_mir::analysis::analyze_assembly(&mir);
        let mut executable = manifest(&analysis);
        for function in &analysis.functions {
            executable
                .capabilities
                .0
                .extend(function.capabilities.0.iter().cloned());
        }
        lower_with(&mir, &analysis, &executable)
            .expect("shared runtime pool calls remain native-supported");
    }
}

#[test]
fn executor_capability_rejection_visits_short_circuit_assignment_targets() {
    let span = Span { start: 0, end: 1 };
    let MirRvalue::Call(mut call) = executor_builtin_call("fetchOutputs") else {
        unreachable!()
    };
    call.args
        .push(runmat_mir::MirCallArg::Single(MirOperand::Local(
            MirLocalId(1),
        )));
    call.arg_spans.push(span);
    let selector = runmat_mir::MirExpressionRegion::from_lowered(
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::End,
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(2)),
                    value: MirRvalue::Call(call),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(2)),
    )
    .unwrap();
    let value = MirRvalue::ShortCircuit {
        left: MirOperand::Constant(MirConstant::Bool(true)),
        op: runmat_mir::MirShortCircuitOp::And,
        right_temps: vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Index(
                    Box::new(MirPlace::Local(MirLocalId(0))),
                    runmat_mir::MirIndexing {
                        argument_spans: None,
                        kind: runmat_types::IndexKind::Paren,
                        plan: runmat_mir::MirIndexPlan::Slice,
                        components: vec![runmat_mir::MirIndexComponent::ContextualExpr(selector)],
                        result_context: runmat_types::IndexResultContext::AssignmentTarget,
                        cell_expand_all: false,
                    },
                ),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("1".into()))),
            },
            span,
        }],
        right: MirOperand::Constant(MirConstant::Bool(true)),
    };
    let mut mir = function(vec![MirStmt {
        kind: MirStmtKind::Expr(value),
        span,
    }]);
    for id in 1..=2 {
        mir.bodies
            .get_mut(&FunctionId(0))
            .unwrap()
            .locals
            .push(MirLocal {
                id: MirLocalId(id),
                binding: None,
                kind: MirLocalKind::Temporary,
                span,
            });
    }
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let error = lower_with(&mir, &analysis, &manifest(&analysis)).unwrap_err();
    assert_eq!(error.code, "native.capability.executor_intrinsic");
}

#[test]
fn capability_rejection_cannot_hide_inside_short_circuit_payloads() {
    let nested = MirStmt {
        kind: MirStmtKind::Expr(MirRvalue::Distributed(
            runmat_mir::parallel::MirDistributedOp::LocalPart {
                value: MirOperand::Constant(MirConstant::Number("1".into())),
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
    let manifest = manifest(&analysis);
    assert_eq!(
        lower_with(&mir, &analysis, &manifest).unwrap_err().code,
        "native.capability.distributed_core_pending"
    );
}

#[test]
fn capability_rejection_cannot_hide_inside_sequence_capture_regions() {
    let span = Span { start: 0, end: 1 };
    let region = runmat_mir::MirExpressionRegion::from_lowered(
        vec![
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(1)),
                    value: MirRvalue::End,
                },
                span,
            },
            MirStmt {
                kind: MirStmtKind::Assign {
                    place: MirPlace::Local(MirLocalId(2)),
                    value: MirRvalue::Distributed(
                        runmat_mir::parallel::MirDistributedOp::LocalPart {
                            value: MirOperand::Local(MirLocalId(1)),
                        },
                    ),
                },
                span,
            },
        ],
        MirOperand::Local(MirLocalId(2)),
    )
    .unwrap();
    let capture = MirStmt {
        kind: MirStmtKind::CaptureSequence {
            destination: runmat_mir::MirSequenceLocalId(0),
            source: runmat_mir::MirExpansionSource::CellContents {
                base: MirOperand::Constant(MirConstant::EmptyArray),
                indexing: runmat_mir::MirIndexing {
                    argument_spans: None,
                    kind: runmat_hir::IndexKind::Brace,
                    plan: runmat_mir::MirIndexPlan::Slice,
                    components: vec![runmat_mir::MirIndexComponent::ContextualExpr(region)],
                    result_context: runmat_types::IndexResultContext::ReadCommaList,
                    cell_expand_all: true,
                },
            },
        },
        span,
    };
    let mut mir = function(vec![capture]);
    let body = mir.bodies.get_mut(&FunctionId(0)).unwrap();
    for id in 1..=2 {
        body.locals.push(MirLocal {
            id: MirLocalId(id),
            binding: None,
            kind: MirLocalKind::Temporary,
            span,
        });
    }
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let manifest = manifest(&analysis);
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
    // Sequence production and sequence assignment are distinct canonical
    // constructs; neither may disappear into ordinary expression/assignment
    // metadata.
    assert!(all.contains(&runmat_mir::MirConstructKind::CaptureSequence));
    assert!(all.contains(&runmat_mir::MirConstructKind::SequenceAssign));
    assert!(all.contains(&runmat_mir::MirConstructKind::WorkspaceFirstStaticProperty));
    assert_eq!(all.len(), 62);
    assert!(all.contains(&runmat_mir::MirConstructKind::PrepareFixedAssignment));
    assert!(all.contains(&runmat_mir::MirConstructKind::PreparedAssign));
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
        let mut manifest = manifest(&analysis);
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
                outputs: Vec::new(),
                effects: Default::default(),
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
                    transferability:
                        runmat_types::ParallelTransferability::RequiresRuntimeValidation,
                }],
                maximum_workers: Some(LabCount(4)),
                effects: Default::default(),
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
    assert!(analysis.functions.iter().any(|function| {
        function
            .capabilities
            .0
            .contains(&CapabilityRequirement::ParallelRuntime)
    }));
    let mut manifest = manifest(&analysis);
    manifest
        .capabilities
        .0
        .insert(CapabilityRequirement::ParallelRuntime);
    let assembly = lower_with(&mir, &analysis, &manifest).unwrap();
    assert!(assembly.functions[0]
        .capabilities
        .0
        .contains(&CapabilityRequirement::ParallelRuntime));
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

    let mut weakened = assembly;
    weakened.functions[0].capabilities.0.clear();
    assert!(weakened.verify().is_err());
}

#[test]
fn foreign_requirements_are_preserved_without_backend_inference() {
    let mir = function(vec![assignment(1)]);
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let mut manifest = manifest(&analysis);
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
    assert_eq!(
        assembly.target,
        NativeTarget::PortableWasm32 {
            schema_version: runmat_native_codegen::PORTABLE_WASM_LOWERING_SCHEMA_VERSION
        }
    );
    assert_eq!(
        assembly,
        serde_json::from_slice(&serde_json::to_vec(&assembly).unwrap()).unwrap()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn current_native_ir_uses_the_same_canonical_admission_on_native_and_wasm() {
    let assembly = lower(&function(vec![assignment(1)]));
    let bytes = assembly.canonical_bytes().unwrap();
    assert_eq!(
        runmat_native_codegen::NativeAssembly::from_canonical_bytes(&bytes).unwrap(),
        assembly
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_native_ir_five_is_rejected_before_target_or_instruction_admission() {
    let error = runmat_native_codegen::NativeAssembly::from_canonical_bytes(include_bytes!(
        "fixtures/native-ir-5.json"
    ))
    .unwrap_err();
    assert_eq!(error.code, "native.ir.schema_version");
    assert!(error.message.contains("version 5"));
    assert!(error.message.contains(&format!(
        "expected {}",
        runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION
    )));
}
