use super::*;

fn lower(source: &str) -> MirAssembly {
    let ast = runmat_parser::parse(source).unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    crate::lowering::lower_assembly(&hir.assembly).unwrap()
}

fn facts(source: &str) -> FunctionAnalysis {
    let mir = lower(source);
    let (function, _) = mir
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "pulse")
        .unwrap();
    analyze_assembly(&mir)
        .function_for_local(*function)
        .unwrap()
        .clone()
}

#[test]
fn original_layout_and_acoustic_programs_have_typed_local_suspension_not_erased_effects() {
    for source in [
        "function y=pulse(t)\narguments\nt(:,1) double\nend\nscale=[1,11,21,31,2,12,22,32,3,13,23,33];\ny=reshape(t*scale,numel(t),4,3);\nend",
        "function y=pulse(f)\narguments\nf(:,1) double\nend\ny=complex(1+2*f,0.5-f);\nend",
        "function y=pulse(t,p,d)\narguments\nt(:,1) double\np(1,1) double\nd(1,1) double\nend\ninside=t<=d;\nphase=min(t./d,1);\ny=p.*sin(pi.*phase).*inside;\nend",
    ] {
        let function = facts(source);
        assert_eq!(function.convergence, FunctionConvergence::Exact);
        assert!(function.cooperative_suspension, "{function:#?}");
        assert!(function.effects.0.contains(&runmat_types::EffectKind::MaySuspend));
        assert!(!function.effects.is_side_effect_free(), "compiler reordering policy is unchanged");
        let mut old = serde_json::to_value(&function).unwrap();
        old.as_object_mut().unwrap().remove("cooperative_suspension");
        let old: FunctionAnalysis = serde_json::from_value(old).unwrap();
        assert!(!old.cooperative_suspension, "old flattened effects confer no new permission");
    }
}

#[test]
fn ambient_unknown_and_possible_object_dispatch_cannot_inherit_builtin_locality() {
    for body in [
        "y=t+toc();",
        "pause(0); y=t;",
        "y=opaque(t);",
        "y=rand(size(t));",
    ] {
        let function = facts(&format!(
            "function y=pulse(t)\narguments\nt(:,1) double\nend\n{body}\nend"
        ));
        assert!(!function.cooperative_suspension, "{body}: {function:#?}");
    }
    // A resolved builtin identity alone cannot rule out an overload on an
    // unconstrained parameter. No numeric ABI/capture facts are fabricated.
    assert!(!facts("function y=pulse(t)\ny=reshape(t,numel(t),1);\nend").cooperative_suspension);
    for body in [
        "unused = value && true;",
        "unused = true && value;",
        "unused = [value{:}];",
    ] {
        assert!(
            !facts(&format!(
                "function y=pulse(value)\n{body}\ny=complex(1,2);\nend"
            ))
            .cooperative_suspension,
            "unconstrained short-circuit/sequence operands may be opaque objects: {body}"
        );
    }
    let mut shadowed =
        lower("function y=pulse(t)\narguments\nt(:,1) double\nend\ny=complex(t,t);\nend");
    let mut candidates = 0;
    for body in shadowed.bodies.values_mut() {
        for statement in body
            .blocks
            .iter_mut()
            .flat_map(|block| block.statements.iter_mut())
        {
            let value = match &mut statement.kind {
                crate::MirStmtKind::Assign { value, .. }
                | crate::MirStmtKind::PreparedAssign { value, .. }
                | crate::MirStmtKind::MultiAssign { value, .. }
                | crate::MirStmtKind::SequenceAssign { value, .. }
                | crate::MirStmtKind::Expr(value) => value,
                _ => continue,
            };
            if let crate::MirRvalue::Call(call) = value {
                if let crate::MirCallee::Static(CallableIdentity::Builtin(id)) = &call.callee {
                    assert!(
                        call.workspace_first_name.is_none(),
                        "original closed scope must remain closed"
                    );
                    // Add, never clear, a typed workspace candidate while
                    // retaining the exact builtin target and numeric facts.
                    call.workspace_first_name = Some(runmat_types::SymbolName(id.0.clone()));
                    candidates += 1;
                }
            }
        }
    }
    assert!(candidates > 0);
    let (function, _) = shadowed
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "pulse")
        .unwrap();
    assert!(
        !analyze_assembly(&shadowed)
            .function_for_local(*function)
            .unwrap()
            .cooperative_suspension
    );
    let lhs = lower("function y=pulse(t)\narguments\nt(:,1) double\nend\ny=complex(t,t);\ny(selector(end))=0;\nend\nfunction i=selector(n)\narguments\nn(1,1) double\nend\ni=n+0*cos(n);\nend");
    let (helper, _) = lhs
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "selector")
        .unwrap();
    let (caller, _) = lhs
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "pulse")
        .unwrap();
    let mut statement_owned_call = false;
    for statement in lhs.bodies[caller]
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
    {
        statement.kind.visit_statement_expression_steps_dyn(&mut |steps| {
            statement_owned_call |= steps.iter().any(|step| matches!(step,
                crate::MirExpressionStep::Let { value: crate::MirRvalue::Call(call), .. }
                    if matches!(call.callee, crate::MirCallee::Static(CallableIdentity::BoundFunction(id)) if id == *helper)));
        });
    }
    assert!(
        statement_owned_call,
        "the negative must actually exercise contextual LHS preparation"
    );
    let analysis = analyze_assembly(&lhs);
    assert!(
        !analysis
            .function_for_local(*helper)
            .unwrap()
            .cooperative_suspension,
        "ordinary MaySuspend on cos is not an approved local-suspension origin"
    );
    assert!(
        !analysis
            .function_for_local(*caller)
            .unwrap()
            .cooperative_suspension,
        "a statement-phase selector cannot hide a noncertified callee behind caller MaySuspend"
    );
}
