use runmat_mir::{
    analysis::analyze_assembly, lowering::lower_assembly, MirConstructKind, MirOperationPhase,
};
use runmat_types::EffectKind;

fn lower(source: &str) -> runmat_mir::MirAssembly {
    let ast = runmat_parser::parse(source).expect("parse operation-contract fixture");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
        .expect("lower operation-contract HIR");
    lower_assembly(&hir.assembly).expect("lower operation-contract MIR")
}

fn analyze(source: &str) -> runmat_mir::analysis::AnalysisStore {
    analyze_assembly(&lower(source))
}

fn analyze_unreachable(source: &str) -> runmat_mir::analysis::AnalysisStore {
    let mut mir = lower(source);
    for body in mir.bodies.values_mut() {
        // Source lowering removes statements after an unconditional return.
        // Retain real lowered operations in structurally present blocks, but
        // give the analysis an entry which cannot reach them.
        let mut entry = body.blocks[0].clone();
        entry.id =
            runmat_mir::BasicBlockId(body.blocks.iter().map(|block| block.id.0).max().unwrap() + 1);
        entry.statements.clear();
        entry.terminator.kind = runmat_mir::MirTerminatorKind::Return(Vec::new());
        body.blocks.insert(0, entry);
    }
    analyze_assembly(&mir)
}

#[test]
fn repeated_and_branched_calls_each_retain_complete_local_requirements() {
    let analysis = analyze(
        "function y = f(x)\n a = rand(); b = rand();\n if x; c = rand(); else; c = rand(); end\n y = a + b + c;\n end",
    );
    let random_calls = analysis
        .operation_contracts
        .iter()
        .filter(|contract| {
            contract.site.phase == MirOperationPhase::Rvalue
                && contract.site.construct == MirConstructKind::Call
                && contract.effects.0.contains(&EffectKind::Randomness)
        })
        .count();
    assert_eq!(random_calls, 4);
}

#[test]
fn unresolved_dynamic_call_keeps_unknown_operation_requirement() {
    let analysis = analyze("function y = f()\n y = unresolvedCallback();\n end");
    let dynamic = analysis
        .operation_contracts
        .iter()
        .find(|contract| contract.site.construct == MirConstructKind::Call)
        .expect("dynamic call contract");
    assert!(dynamic.effects.0.contains(&EffectKind::Unknown));
}

#[test]
fn unresolved_call_keeps_its_declared_host_callback_requirement() {
    let ast = runmat_parser::parse("function y = f()\n y = unresolvedCallback();\n end")
        .expect("parse declared-effect fixture");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
        .expect("lower declared-effect HIR");
    let mut mir = lower_assembly(&hir.assembly).expect("lower declared-effect MIR");
    let call = mir
        .bodies
        .values_mut()
        .flat_map(|body| &mut body.blocks)
        .flat_map(|block| &mut block.statements)
        .find_map(|statement| match &mut statement.kind {
            runmat_mir::MirStmtKind::Assign {
                value: runmat_mir::MirRvalue::Call(call),
                ..
            } => Some(call),
            _ => None,
        })
        .expect("dynamic call");
    call.effects = runmat_builtins::BuiltinEffects::none().with_host_callback();
    let analysis = analyze_assembly(&mir);
    let contract = analysis
        .operation_contracts
        .iter()
        .find(|contract| contract.site.construct == MirConstructKind::Call)
        .expect("dynamic call contract");
    assert!(contract.effects.0.contains(&EffectKind::HostCallback));
    assert!(contract.effects.0.contains(&EffectKind::Unknown));
}

#[test]
fn embedded_and_terminator_values_keep_their_local_contracts() {
    let embedded = analyze("function y = f()\n y = true && rand();\n end");
    let short_circuit = embedded
        .operation_contracts
        .iter()
        .find(|contract| contract.site.construct == MirConstructKind::ShortCircuit)
        .expect("short-circuit operation contract");
    assert!(short_circuit.effects.0.contains(&EffectKind::Randomness));

    let terminator = analyze("function y = f()\n y = 0; for x = rand(); y = x; end\n end");
    assert!(terminator.operation_contracts.iter().any(|contract| {
        contract.site.phase == MirOperationPhase::TerminatorRvalue
            && contract.site.construct == MirConstructKind::Call
            && contract.effects.0.contains(&EffectKind::Randomness)
    }));

    let assignment = analyze("function A = f()\n A = zeros(1, 3); A(randi(end)) = 2;\n end");
    assert!(assignment.operation_contracts.iter().any(|contract| {
        contract.site.phase == MirOperationPhase::Statement
            && contract.effects.0.contains(&EffectKind::Randomness)
    }));
}

#[test]
fn unreachable_embedded_calls_keep_complete_conservative_requirements() {
    let analysis = analyze_unreachable("function y = f()\n y = true && rand();\n end");
    let short_circuit = analysis
        .operation_contracts
        .iter()
        .find(|contract| contract.site.construct == MirConstructKind::ShortCircuit)
        .expect("unreachable short-circuit contract");
    assert!(short_circuit.effects.0.contains(&EffectKind::Randomness));

    let assignment =
        analyze_unreachable("function A = f()\n A = zeros(1, 3); A(randi(end)) = 2;\n end");
    assert!(assignment.operation_contracts.iter().any(|contract| {
        contract.site.phase == MirOperationPhase::Statement
            && contract.effects.0.contains(&EffectKind::Randomness)
    }));
}
