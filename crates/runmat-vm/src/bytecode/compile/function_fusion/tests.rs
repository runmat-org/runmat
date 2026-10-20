use super::Instr;
use crate::bytecode::FunctionRegistry;

#[test]
fn indexed_windows_preserve_first_candidate_and_pc_gaps() {
    let spans = [
        runmat_hir::Span { start: 10, end: 11 },
        runmat_hir::Span { start: 12, end: 13 },
        runmat_hir::Span { start: 20, end: 21 },
        runmat_hir::Span { start: 10, end: 11 },
        runmat_hir::Span { start: 25, end: 26 },
        runmat_hir::Span { start: 20, end: 21 },
    ];
    let instructions = [
        Instr::Add,
        Instr::StoreVar(0),
        Instr::Mul,
        Instr::Sub,
        Instr::Add,
        Instr::Mul,
    ];
    let candidates = [
        crate::bytecode::FusionCandidateGroup {
            id: 0,
            signal_count: 4,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 0,
            stmt_end: 4,
            source_span: runmat_hir::Span { start: 10, end: 21 },
        },
        crate::bytecode::FusionCandidateGroup {
            id: 1,
            signal_count: 2,
            function: runmat_hir::FunctionId(0),
            block: runmat_mir::BasicBlockId(0),
            stmt_start: 4,
            stmt_end: 6,
            source_span: runmat_hir::Span { start: 10, end: 26 },
        },
    ];
    let windows =
        super::derive_semantic_fusion_instruction_windows(&instructions, &spans, &candidates);
    assert_eq!(
        windows
            .iter()
            .map(|window| (window.span.start, window.span.end, window.kind))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, crate::bytecode::FusionInstructionKind::Elementwise),
            (2, 3, crate::bytecode::FusionInstructionKind::Matmul),
            (5, 5, crate::bytecode::FusionInstructionKind::Matmul),
            (4, 4, crate::bytecode::FusionInstructionKind::Elementwise),
        ]
    );
}

fn compile_function(source: &str) -> (runmat_hir::FunctionId, FunctionRegistry) {
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let function = hir.assembly.functions[0].id;
    let registry = FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
    );
    (function, registry)
}

#[test]
fn general_elementwise_function_keeps_one_chain_through_portable_admission() {
    let (function, registry) =
        compile_function("function y=f(x)\ny=exp(-x).*sin(4*x)+tanh(x).*tanh(x);\nend\n");
    fn assert_chain(registry: &FunctionRegistry, function: runmat_hir::FunctionId) {
        let body = registry.get(function).unwrap();
        body.validate_fusion_scaffold().unwrap();
        let executable = body.execution_bytecode(registry);
        let layout = executable
            .layout
            .as_ref()
            .expect("selected function layout");
        assert!(
            layout.entrypoints.is_empty(),
            "no borrowed entrypoint layout"
        );
        assert!(layout.functions.contains_key(&function));
        let signals = body
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(pc, instruction)| {
                matches!(instruction, Instr::CallBuiltinMulti(name, 1, 1)
                if name == "exp" || name == "sin" || name == "tanh")
                .then_some(pc)
            })
            .collect::<Vec<_>>();
        assert_eq!(signals.len(), 4);
        let (_, groups) = executable.reconciled_runtime_fusion_graph().unwrap();
        assert!(
            groups.iter().any(|group| {
                group.kind == runmat_accelerate::FusionKind::ElementwiseChain
                    && group.nodes.len() >= 8
                    && signals
                        .iter()
                        .all(|pc| group.span.start <= *pc && *pc <= group.span.end)
            }),
            "one original function chain, without injected argument facts"
        );
        // MIR intrinsically gives these primitives no local effects even with
        // unknown function inputs. The new allowance belongs only to calls.
        let projection = body.fusion_stage_projection.as_ref().unwrap();
        for statement in &projection.statements {
            if body.instructions[statement.pc.start..statement.pc.end]
                .iter()
                .any(|instruction| matches!(instruction, Instr::Neg | Instr::Mul))
            {
                assert!(statement.contract.effects.0.is_empty());
            }
        }
    }
    assert_chain(&registry, function);
    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry).unwrap();
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded).unwrap();
    assert_chain(&decoded, decoded.resolve_name("f").unwrap());

    let body = registry.get(function).unwrap();
    let statement = body
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .statements
        .iter()
        .find(|statement| {
            body.instructions[statement.pc.start..statement.pc.end]
                .iter()
                .any(|instruction| {
                    matches!(instruction,
                Instr::CallBuiltinMulti(name, 1, 1) if name == "exp")
                })
        })
        .unwrap();
    let compatible = crate::bytecode::fusion_effects::elementwise_statement_effects_are_compatible;
    assert!(compatible(&body.instructions, statement));
    for effect in [
        runmat_types::EffectKind::Unknown,
        runmat_types::EffectKind::HostCallback,
        runmat_types::EffectKind::WorkspaceWrite,
    ] {
        let mut forbidden = statement.clone();
        forbidden.contract.effects.0.insert(effect);
        assert!(!compatible(&body.instructions, &forbidden));
    }
    // Neither an intrinsic nor an undeclared call can borrow exp's effects.
    let pc = (statement.pc.start..statement.pc.end)
        .find(|pc| {
            matches!(
                &body.instructions[*pc], Instr::CallBuiltinMulti(name, 1, 1) if name == "exp"
            )
        })
        .unwrap();
    let mut effectful = statement.clone();
    effectful
        .contract
        .effects
        .0
        .insert(runmat_types::EffectKind::MayThrow);
    let mut replaced = body.instructions.clone();
    replaced[pc] = Instr::Neg;
    assert!(!compatible(&replaced, &effectful));
    replaced[pc] = Instr::CallBuiltinMulti("disp".into(), 1, 1);
    assert!(!compatible(&replaced, &effectful));
}

#[test]
fn effectful_assertion_chain_does_not_poison_neighboring_function_groups() {
    let source = "function y=f(x)\na=x+x;\nb=a+a;\nassert(max(abs(x-[1 2 3]))<1e-10);\nc=x+x;\nd=c+c;\ny=b+d;\nend\n";
    let (function, registry) = compile_function(source);
    let body = registry.get(function).unwrap();
    let scaffold = body
        .fusion_scaffold
        .as_ref()
        .expect("neighboring pure groups");
    assert!(scaffold.candidates.len() >= 2, "retain both pure runs");
    body.validate_fusion_scaffold()
        .expect("compiler hint certificate");
    crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("compiler-produced ProgramV2 must encode");

    let mut malformed = body.clone();
    malformed.fusion_scaffold.as_mut().unwrap().candidates[0].signal_count += 1;
    let malformed_registry = FunctionRegistry::new([(function, malformed)].into_iter().collect());
    assert!(
        crate::bytecode::encode_interpreter_program_v2(&malformed_registry).is_err(),
        "external metadata still fails the strict ProgramV2 validator"
    );
}

#[test]
fn selected_terminal_reduction_keeps_its_function_certificate() {
    let (function, registry) = compile_function("function y=f(x)\ny=min(x+x);\nend\n");
    let body = registry.get(function).unwrap();
    assert!(body.fusion_scaffold.is_some(), "certified reduction group");
    body.validate_fusion_scaffold()
        .expect("terminal reduction certificate");
    crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("terminal reduction ProgramV2");
}

#[test]
fn uncertified_empty_control_splits_without_losing_pure_neighbors() {
    let source =
        "function y=f(x)\na=x+x;\nb=a+a;\nt=min(x+x,[],2);\nc=x+x;\nd=c+c;\ny=b+d+t;\nend\n";
    let (function, registry) = compile_function(source);
    let mut body = registry.get(function).unwrap().clone();
    assert!(
        body.fusion_scaffold.as_ref().is_some_and(|scaffold| {
            scaffold.candidates.len() >= 2
                && scaffold
                    .candidates
                    .iter()
                    .any(|candidate| candidate.stmt_end - candidate.stmt_start >= 4)
        }),
        "source has a compiler-certified reduction bridge and pure neighbors"
    );
    body.fusion_scaffold.as_mut().unwrap().private_layout = None;
    let mir = {
        let ast = runmat_parser::parse(source).unwrap();
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
        runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap()
    };
    super::certify_compiled_function_scaffold(&mut body, &mir).unwrap();
    let scaffold = body
        .fusion_scaffold
        .as_ref()
        .expect("retained pure neighbors");
    let control = body
        .fusion_stage_projection
        .as_ref()
        .unwrap()
        .statements
        .iter()
        .find(|statement| {
            matches!(
                body.instructions.get(statement.pc.start..statement.pc.end),
                Some([Instr::CreateMatrix(0, 0), Instr::StoreVar(_)])
            )
        })
        .expect("source empty control")
        .contract
        .point;
    assert!(scaffold.candidates.iter().any(|candidate| {
        candidate.block.0 == control.block as usize
            && candidate.stmt_end <= control.position as usize
            && candidate.signal_count >= 2
    }));
    assert!(scaffold.candidates.iter().any(|candidate| {
        candidate.block.0 == control.block as usize
            && candidate.stmt_start > control.position as usize
            && candidate.signal_count >= 2
    }));
    body.validate_fusion_scaffold().unwrap();
}

#[test]
fn function_local_semantic_windows_survive_executable_materialization() {
    let source = "function y = f()\npersistent p\nt = p .* 2;\np = p + t + 1;\ny = p;\nend\n";
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    assert!(hir.assembly.entrypoints.is_empty());
    let function = hir.assembly.functions[0].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
    );
    let body = registry.get(function).unwrap();
    let scaffold = body
        .fusion_scaffold
        .as_ref()
        .expect("function-owned MIR candidates");
    assert!(
        scaffold.private_layout.is_some(),
        "compiler-owned function slots"
    );
    assert!(scaffold.valid_for(function, &body.instructions, &body.instr_spans));
    body.validate_fusion_scaffold()
        .expect("selected statement admission");
    let layout = crate::layout::derive_layout(&hir.assembly, &mir).unwrap();
    let executable = crate::Bytecode::for_function(body, registry.clone(), layout.clone());
    assert_eq!(
        executable.fusion_metadata.mir_fusion_candidate_group_count,
        scaffold.candidates.len()
    );
    assert_eq!(executable.function_private_layout, scaffold.private_layout);
    assert!(executable
        .fusion_metadata
        .instruction_windows
        .iter()
        .all(|window| {
            body.instructions[window.span.start..=window.span.end]
                .iter()
                .all(|instruction| {
                    !matches!(
                        instruction,
                        crate::Instr::StoreVar(_) | crate::Instr::StoreLocal(_)
                    )
                })
        }));

    let mut remapped = body.clone();
    let destination = runmat_hir::FunctionId(function.0 + 10);
    let source = remapped.source_id;
    crate::bytecode::remap_session_function(
        &mut remapped,
        destination,
        source,
        &[(function, destination)].into_iter().collect(),
    )
    .expect("session-owned remap");
    assert!(remapped
        .fusion_scaffold
        .as_ref()
        .is_some_and(|scaffold| scaffold.valid_for(
            destination,
            &remapped.instructions,
            &remapped.instr_spans
        )));
    remapped
        .validate_fusion_scaffold()
        .expect("remapped statement admission");
    let mut legacy = body.clone();
    legacy.fusion_scaffold = None;
    assert_ne!(
        crate::bytecode::semantic_definition_fingerprint(body).unwrap(),
        crate::bytecode::semantic_definition_fingerprint(&legacy).unwrap()
    );

    let mut malformed = body.clone();
    malformed.fusion_scaffold.as_mut().unwrap().candidates[0].function =
        runmat_hir::FunctionId(function.0 + 1);
    let malformed_registry =
        crate::bytecode::FunctionRegistry::new([(function, malformed)].into_iter().collect());
    let rejected = crate::Bytecode::for_function(
        malformed_registry.get(function).unwrap(),
        malformed_registry.clone(),
        layout,
    );
    assert!(rejected.runtime_fusion_groups().is_empty());
}
