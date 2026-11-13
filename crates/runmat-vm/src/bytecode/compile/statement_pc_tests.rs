use crate::bytecode::{Bytecode, FunctionRegistry, Instr, StatementPcRange};
use runmat_hir::FunctionId;

fn compile_source(source: &str) -> (Bytecode, runmat_mir::MirAssembly, FunctionId) {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let entrypoint = &hir.assembly.entrypoints[0];
    let bytecode = super::compile(&hir.assembly, &mir, entrypoint.id).expect("compile bytecode");
    (bytecode, mir, entrypoint.target)
}

fn assert_exact_statement_projection(
    bytecode: &Bytecode,
    mir: &runmat_mir::MirAssembly,
    function: FunctionId,
) -> Vec<StatementPcRange> {
    let ranges = bytecode.statement_pc_ranges().expect("valid PC projection");
    let body = mir.bodies.get(&function).expect("selected MIR body");
    assert_eq!(
        ranges.len(),
        body.blocks
            .iter()
            .map(|block| block.statements.len())
            .sum::<usize>(),
        "one range per emitted MIR statement, including empty emission"
    );
    for block in &body.blocks {
        for (position, statement) in block.statements.iter().enumerate() {
            let range = ranges
                .iter()
                .find(|range| {
                    range.statement.function.0 as usize == function.0
                        && range.statement.block as usize == block.id.0
                        && range.statement.position as usize == position
                })
                .expect("exact function/block/statement owner");
            assert!(range.start <= range.end && range.end <= bytecode.instructions.len());
            for pc in range.start..range.end {
                let span = bytecode.instr_spans[pc];
                assert!(
                    statement.span.start <= span.start && span.end <= statement.span.end,
                    "instruction {pc} escapes owning MIR statement span"
                );
                assert!(
                    !matches!(
                        bytecode.instructions[pc],
                        Instr::Jump(_)
                            | Instr::JumpIfFalse(_)
                            | Instr::StochasticEvolution { .. }
                            | Instr::EnterTry { .. }
                            | Instr::LeaveTry(_)
                            | Instr::Return
                            | Instr::ReturnValue
                    ),
                    "terminator/control boundary entered statement {position}"
                );
            }
            let mut depth = 0usize;
            let mut known_stack = true;
            for instruction in &bytecode.instructions[range.start..range.end] {
                let Some(effect) = instruction.stack_effect() else {
                    known_stack = false;
                    break;
                };
                assert!(depth >= effect.pops, "statement underflows its own stack");
                depth = depth - effect.pops + effect.pushes;
            }
            if known_stack {
                assert_eq!(depth, 0, "statement leaks a value into its successor");
            }
            if range.start < range.end {
                let catch = bytecode.catch_scopes[range.start];
                assert!(
                    bytecode.catch_scopes[range.start..range.end]
                        .iter()
                        .all(|scope| *scope == catch),
                    "statement crosses lexical catch scope"
                );
            }
        }
    }
    ranges
}

fn store_range<'a>(
    bytecode: &Bytecode,
    ranges: &'a [StatementPcRange],
    name: &str,
) -> &'a StatementPcRange {
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .expect("named output slot");
    let pc = bytecode
        .instructions
        .iter()
        .position(|instruction| matches!(instruction, Instr::StoreVar(index) if *index == slot))
        .expect("output store");
    ranges
        .iter()
        .find(|range| range.start <= pc && pc < range.end)
        .expect("store belongs to exactly one MIR statement")
}

#[test]
fn actual_vecdim_image_assignments_have_distinct_exact_store_ranges() {
    let source = r#"
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+single(1e-6)));
        out = single(((imgs-mu)./sigma).^single(1.8));
    "#;
    let (bytecode, mir, function) = compile_source(source);
    let ranges = assert_exact_statement_projection(&bytecode, &mir, function);
    let mu = store_range(&bytecode, &ranges, "mu");
    let sigma = store_range(&bytecode, &ranges, "sigma");
    let out = store_range(&bytecode, &ranges, "out");
    assert!(mu.end <= sigma.start && sigma.end <= out.start);
    assert_eq!(mu.statement.function, sigma.statement.function);
    assert_eq!(sigma.statement.function, out.statement.function);
    // Metadata alone cannot authorize a multi-store specialized route.
    #[cfg(feature = "native-accel")]
    assert!(bytecode
        .runtime_fusion_groups()
        .iter()
        .all(|group| group.kind != runmat_accelerate::FusionKind::ImageNormalize));
}

#[test]
fn generic_stores_and_branch_catch_blocks_keep_separate_authority() {
    for source in [
        "a=1; mu=a+1; sigma=mu.*2; out=sigma-3;",
        "a=1; if a; mu=a+1; else; mu=a-1; end; try; sigma=mu.*2; catch e; sigma=0; end; out=sigma-3;",
    ] {
        let (bytecode, mir, function) = compile_source(source);
        let ranges = assert_exact_statement_projection(&bytecode, &mir, function);
        assert!(!ranges.is_empty());
        if source.contains("catch") {
            assert!(
                ranges
                    .iter()
                    .map(|range| range.statement.block)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 1
            );
            assert!(bytecode.instructions.iter().any(|instr| matches!(instr, Instr::EnterTry { .. })));
        }
    }
}

#[test]
fn function_and_empty_body_projection_stay_function_local() {
    let source = "a=helper(1); function y=helper(x); y=x+1; end; function noop(); end";
    let (bytecode, mir, entry_function) = compile_source(source);
    assert_exact_statement_projection(&bytecode, &mir, entry_function);
    let registry = FunctionRegistry::new(bytecode.bound_functions.clone());
    let mut saw_helper = false;
    let mut saw_empty = false;
    for function in bytecode.bound_functions.values() {
        let executable = function.execution_bytecode(&registry);
        assert_eq!(executable.resume_points, function.resume_points);
        let ranges = assert_exact_statement_projection(&executable, &mir, function.function);
        assert!(ranges
            .iter()
            .all(|range| { range.statement.function.0 as usize == function.function.0 }));
        if function.display_name == "helper" {
            saw_helper = true;
            assert!(!ranges.is_empty());
        } else if function.display_name == "noop" {
            saw_empty = true;
            assert!(ranges.is_empty());
        }
    }
    assert!(saw_helper && saw_empty);
}

#[test]
fn serialized_script_retains_the_exact_resume_authority() {
    let (bytecode, _, _) = compile_source("a=1; b=a+2;");
    let encoded = crate::bytecode::encode_interpreter_script_v2(&bytecode).unwrap();
    let decoded = crate::bytecode::decode_interpreter_script_v2(&encoded).unwrap();
    assert_eq!(decoded.resume_points, bytecode.resume_points);
    assert_eq!(
        decoded.statement_pc_ranges(),
        bytecode.statement_pc_ranges()
    );
}

#[test]
fn serialized_function_retains_the_exact_resume_authority() {
    let (bytecode, _, _) = compile_source("a=helper(1); function y=helper(x); y=x+1; end");
    let registry = FunctionRegistry::new(bytecode.bound_functions);
    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry).unwrap();
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded).unwrap();
    for (function_id, function) in &registry.functions {
        let restored = decoded
            .functions
            .get(function_id)
            .expect("restored function");
        assert_eq!(restored.resume_points, function.resume_points);
        assert_eq!(
            restored.statement_pc_ranges(),
            function.statement_pc_ranges()
        );
    }
}

#[test]
fn session_rebind_projection_tracks_issued_program_point_owner() {
    let (bytecode, _, _) = compile_source("a=helper(1); function y=helper(x); y=x+1; end");
    let mut function = bytecode
        .bound_functions
        .values()
        .find(|body| body.display_name == "helper")
        .expect("compiled helper")
        .clone();
    let portable_origin = function.portable_function;
    function.rebind_identity(FunctionId(77));
    assert_eq!(function.portable_function, portable_origin);
    let ranges = function.statement_pc_ranges().expect("rebound projection");
    assert!(!ranges.is_empty());
    assert!(ranges
        .iter()
        .all(|range| range.statement.function == portable_origin));

    let registry = FunctionRegistry::new(std::collections::HashMap::from([(
        FunctionId(77),
        function,
    )]));
    let artifact = crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("rebound artifact with matching current owner");
    let decoded = crate::bytecode::decode_interpreter_program_v2(&artifact).unwrap();
    let restored = decoded.get(FunctionId(77)).expect("restored rebound owner");
    assert_eq!(restored.portable_function, portable_origin);
    assert_eq!(restored.statement_pc_ranges().unwrap(), ranges);
}

#[test]
fn closed_unit_remap_keeps_portable_point_owner_distinct_from_runtime_function() {
    let (bytecode, _, _) = compile_source("a=helper(1); function y=helper(x); y=x+1; end");
    let mut function = bytecode
        .bound_functions
        .values()
        .find(|body| body.display_name == "helper")
        .expect("compiled helper")
        .clone();
    let source_function = function.function;
    let portable_origin = function.portable_function;
    crate::bytecode::program::function_remap::remap_closed_function(
        &mut function,
        &std::collections::HashMap::from([(source_function, FunctionId(88))]),
    )
    .expect("closed projection");
    assert_eq!(function.function, FunctionId(88));
    assert_eq!(function.portable_function, portable_origin);
    let ranges = function
        .statement_pc_ranges()
        .expect("closed-unit projection");
    assert!(!ranges.is_empty());
    assert!(ranges
        .iter()
        .all(|range| range.statement.function == portable_origin));

    let registry = FunctionRegistry::new(std::collections::HashMap::from([(
        FunctionId(88),
        function,
    )]));
    let artifact = crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("closed-unit projection artifact");
    let decoded = crate::bytecode::decode_interpreter_program_v2(&artifact).unwrap();
    let restored = decoded.get(FunctionId(88)).expect("restored closed unit");
    assert_eq!(restored.statement_pc_ranges().unwrap(), ranges);
}
