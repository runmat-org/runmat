use super::*;
use crate::analysis::regions::analyze_liveness;

fn analyze(source: &str) -> (crate::MirAssembly, crate::analysis::AnalysisStore) {
    let ast = runmat_parser::parse(source).expect("parse source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let analysis = crate::analysis::analyze_assembly(&mir);
    (mir, analysis)
}

#[test]
fn indexed_array_shape_survives_source_calls_and_structural_transforms() {
    use crate::{MirCallee, MirPlace, MirRvalue, MirStmtKind};
    use runmat_types::{ProgramPointId, RegionValueId, ShapeFact};

    for (source, expected) in [
        (
            "mask = false(2,1,3); mask(1,1,2) = true; rotated = permute(mask,[3 1 2]); orig = ipermute(rotated,[3 1 2]); a = rotated(2,1,1); b = orig(1,1,2);",
            vec![
                ("false", vec![2, 1, 3]),
                ("indexed write", vec![2, 1, 3]),
                ("permute", vec![3, 2, 1]),
                ("ipermute", vec![2, 1, 3]),
            ],
        ),
        (
            "T = zeros(1,1,5); T(:,:,3) = 7; vec = squeeze(T); a = vec(3);",
            vec![
                ("zeros", vec![1, 1, 5]),
                ("indexed write", vec![1, 1, 5]),
                ("squeeze", vec![5, 1]),
            ],
        ),
    ] {
        let (mir, analysis) = analyze(source);
        let mut observed = Vec::new();
        for (function, body) in &mir.bodies {
            let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
            for block in &body.blocks {
                for (position, statement) in block.statements.iter().enumerate() {
                    let (label, local) = match &statement.kind {
                        MirStmtKind::Assign {
                            place: MirPlace::Local(local),
                            value: MirRvalue::Call(call),
                        } => {
                            let MirCallee::Static(identity) = &call.callee else {
                                continue;
                            };
                            let Some(name) = identity.display_name() else {
                                continue;
                            };
                            (name, *local)
                        }
                        MirStmtKind::Assign {
                            place: MirPlace::Index(base, _),
                            ..
                        } => {
                            let MirPlace::Local(local) = base.as_ref() else {
                                continue;
                            };
                            ("indexed write".to_owned(), *local)
                        }
                        _ => continue,
                    };
                    let point = ProgramPointId {
                        function,
                        block: u32::try_from(block.id.0).unwrap(),
                        position: u32::try_from(position + 1).unwrap(),
                    };
                    let shape = analysis
                        .facts_at(point)
                        .expect("post-statement facts")
                        .local(RegionValueId {
                            function,
                            local: u32::try_from(local.0).unwrap(),
                        })
                        .and_then(|local| local.fact.as_ref())
                        .expect("assigned array fact")
                        .shape
                        .clone();
                    observed.push((label, shape));
                }
            }
        }
        for (name, dimensions) in expected {
            let shape = ShapeFact::from(dimensions.into_iter().map(Some).collect::<Vec<_>>());
            assert!(
                observed
                    .iter()
                    .any(|(label, found)| label == name && found.is_proven_equivalent(&shape)),
                "{name} must retain {shape:?}; observed {observed:?}; diagnostics {:#?}",
                analysis.diagnostics
            );
        }
        assert!(
            analysis
                .diagnostics
                .iter()
                .all(|item| item.code != "RM-TYPE-INDEX-BOUNDS"),
            "valid source indices cannot exceed a falsely narrowed bound: {:#?}",
            analysis.diagnostics
        );
    }
}

#[test]
fn same_pass_stage_contracts_preserve_statement_uses_defs_and_live_after() {
    let (mir, analysis) = analyze("a=1; b=a+1; c=b.*2;");
    let (function, body) = mir.bodies.iter().next().expect("selected body");
    let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
    let statement_count: usize = body.blocks.iter().map(|block| block.statements.len()).sum();
    let contracts: Vec<_> = analysis
        .fusion_stage_contracts
        .iter()
        .filter(|contract| contract.point.function == function)
        .collect();
    assert_eq!(contracts.len(), statement_count);
    assert!(contracts
        .windows(2)
        .all(|pair| pair[0].point < pair[1].point));

    let mut found_live_handoff = false;
    for block in &body.blocks {
        for (position, statement) in block.statements.iter().enumerate() {
            let point = ProgramPointId {
                function,
                block: u32::try_from(block.id.0).unwrap(),
                position: u32::try_from(position).unwrap(),
            };
            let contract = analysis.fusion_stage_at(point).expect("same-pass contract");
            let (uses, defs) = statement_uses_defs(statement);
            assert_eq!(contract.uses, typed_locals(function, &uses).unwrap());
            assert_eq!(contract.defs, typed_locals(function, &defs).unwrap());
            assert_ne!(contract.status, FusionStageFactStatus::MissingFacts);
            found_live_handoff |= contract
                .defs
                .iter()
                .any(|value| contract.live_after.contains(value));
        }
    }
    assert!(
        found_live_handoff,
        "b remains live for the following expression"
    );
}

#[test]
fn stage_effects_are_local_even_when_body_flow_remembers_an_earlier_effect() {
    let (_, analysis) = analyze("disp(1); a=1; b=a+2;");
    let effectful = analysis
        .fusion_stage_contracts
        .iter()
        .find(|contract| contract.effects.0.contains(&EffectKind::UserInterface))
        .expect("disp has a cataloged user-interface effect");
    let later_pure = analysis
        .fusion_stage_contracts
        .iter()
        .find(|contract| {
            contract.point.function == effectful.point.function
                && contract.point.block == effectful.point.block
                && contract.point.position > effectful.point.position
                && contract.effects.0.is_empty()
                && contract.status == FusionStageFactStatus::Inferred
        })
        .expect("a later assignment retains its own pure contract");
    assert!(analysis
        .facts_at(later_pure.point)
        .expect("pre-statement body flow")
        .effects
        .0
        .contains(&EffectKind::UserInterface));
}

#[test]
fn unresolved_call_and_missing_flow_point_are_never_inferred_pure() {
    let (mir, analysis) = analyze("x=1; out=unresolved_callback(x);");
    assert!(analysis.fusion_stage_contracts.iter().any(|contract| {
        contract.status == FusionStageFactStatus::UnknownEffects
            && contract.effects.0.contains(&EffectKind::Unknown)
    }));

    let (function, body) = mir.bodies.iter().next().unwrap();
    let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
    let liveness = analyze_liveness(body);
    let missing = certify_body(
        body,
        function,
        &[],
        &BTreeMap::new(),
        &liveness,
        FunctionConvergence::Exact,
    );
    assert_eq!(
        missing.len(),
        body.blocks
            .iter()
            .map(|block| block.statements.len())
            .sum::<usize>()
    );
    assert!(missing.iter().all(|contract| {
        contract.status == FusionStageFactStatus::MissingFacts
            && contract.effects.0.contains(&EffectKind::Unknown)
    }));
}

#[test]
fn unknown_parameter_and_branch_join_cannot_certify_inferred_inputs() {
    let (_, parameter) = analyze("function y=f(x)\n y=x+1;\nend");
    assert!(parameter
        .fusion_stage_contracts
        .iter()
        .any(|contract| contract.status == FusionStageFactStatus::UnknownInputs));

    let (_, branch) = analyze("function y=f(flag)\n if flag\n  a=1;\n end\n y=a+1;\nend");
    assert!(branch.fusion_stage_contracts.iter().any(|contract| {
        contract.status == FusionStageFactStatus::UnknownInputs
            && contract.uses.iter().any(|value| {
                branch.facts_at(contract.point).is_some_and(|facts| {
                    facts
                        .local(*value)
                        .is_some_and(|local| local.assignment != AssignmentFact::DefinitelyAssigned)
                })
            })
    }));
}

#[test]
fn incomplete_or_widened_function_analysis_is_not_an_executable_purity_proof() {
    let (mir, analysis) = analyze("a=1; b=a+1;");
    let (function, body) = mir.bodies.iter().next().unwrap();
    let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
    let points: Vec<_> = analysis
        .program_points
        .iter()
        .filter(|facts| facts.point.function == function)
        .cloned()
        .collect();
    let liveness = analyze_liveness(body);
    for (convergence, expected) in [
        (
            FunctionConvergence::Widened,
            FusionStageFactStatus::WidenedFacts,
        ),
        (
            FunctionConvergence::DynamicRecursion,
            FusionStageFactStatus::Nonconverged,
        ),
    ] {
        let contracts = certify_body(
            body,
            function,
            &points,
            &BTreeMap::new(),
            &liveness,
            convergence,
        );
        assert!(!contracts.is_empty());
        assert!(contracts.iter().all(|contract| {
            contract.status == expected && contract.effects.0.contains(&EffectKind::Unknown)
        }));
    }
}

#[test]
fn stage_contracts_roundtrip_with_the_analysis_schema() {
    let (_, analysis) = analyze("a=1; b=a+1; c=b.*2;");
    assert_eq!(
        analysis.revision.schema_version,
        crate::analysis::ANALYSIS_STORE_SCHEMA_VERSION
    );
    let bytes = serde_json::to_vec(&analysis).unwrap();
    let decoded: crate::analysis::AnalysisStore = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        decoded.fusion_stage_contracts,
        analysis.fusion_stage_contracts
    );
    decoded.revision.validate_current().unwrap();
}
