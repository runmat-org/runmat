use super::*;

fn diagnostics(source: &str) -> Vec<MirDiagnostic> {
    let parsed = runmat_parser::parse(source).expect("parse workspace declaration");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("lower workspace declaration HIR");
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower workspace declaration MIR");
    crate::analysis::analyze_assembly(&mir).diagnostics
}

#[test]
fn global_and_persistent_declarations_assign_only_their_exact_bindings() {
    for source in [
        "function out = declared_global()\n global item;\n out = item;\n end",
        "function out = declared_persistent()\n persistent item;\n out = item;\n end",
    ] {
        let found = diagnostics(source);
        assert!(
            found.iter().all(|item| item.code != "RM-MIR0001"),
            "declared storage is initialized by its declaration: {found:#?}"
        );
    }

    let mut state = [InitFact::Unassigned, InitFact::Unassigned];
    mark_declared_storage(
        &runmat_hir::WorkspaceEffect::MutatesGlobal,
        &[MirLocalId(0)],
        &mut state,
    );
    assert_eq!(state, [InitFact::DefinitelyAssigned, InitFact::Unassigned]);
    mark_declared_storage(
        &runmat_hir::WorkspaceEffect::ReadsWorkspace,
        &[MirLocalId(1)],
        &mut state,
    );
    assert_eq!(state, [InitFact::DefinitelyAssigned, InitFact::Unassigned]);
}

#[test]
fn ordinary_local_still_requires_assignment() {
    let found = diagnostics("function out = ordinary()\n seen = out;\n out = 1;\n end");
    assert!(found.iter().any(|item| item.code == "RM-MIR0001"));
}

#[test]
fn successful_spmd_separates_client_exports_from_worker_initialization() {
    for header in ["spmd", "spmd(2)", "spmd(1, 2)"] {
        let source = format!("{header}; exported = 7; end; observed = exported;");
        let found = diagnostics(&source);
        assert!(
            found
                .iter()
                .all(|item| !matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
            "successful SPMD executes its nonempty gang: {source}: {found:#?}"
        );

        let source = format!(
            "{header}; if spmdIndex() == 1; conditional = 7; end; end; observed = conditional;"
        );
        let found = diagnostics(&source);
        assert!(
            found
                .iter()
                .all(|item| !matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
            "conditional entries still have a client Composite: {source}: {found:#?}"
        );
        let source = format!(
            "{header}; if spmdIndex() == 1; conditional = 7; end; observed = conditional; end;"
        );
        let found = diagnostics(&source);
        assert!(
            found
                .iter()
                .any(|item| matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
            "SPMD does not prove a conditional worker assignment: {source}: {found:#?}"
        );
    }
    let found = diagnostics(
        "spmd(2); exported = uint64([1, 2]); end; first = exported{1}; second = exported{2};",
    );
    assert!(
        found.is_empty(),
        "client sees Composite entries: {found:#?}"
    );
    let found = diagnostics("spmd(2); exported = 7; end; outside = exported{3};");
    assert!(found
        .iter()
        .any(|item| item.code == "RM-TYPE-COMPOSITE-INDEX"));
}

#[test]
fn admitted_incoming_workspace_is_initialized_without_seeding_new_locals() {
    let variables = std::collections::HashMap::from([("committed".to_string(), 0usize)]);
    let parsed = runmat_parser::parse("seen = committed; committed = seen + 1;")
        .expect("parse workspace continuation");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::new(&variables))
        .expect("lower workspace continuation HIR");
    let binding = hir
        .assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "committed")
        .expect("incoming binding");
    assert_eq!(binding.role, runmat_hir::BindingRole::IncomingWorkspace);
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower workspace continuation MIR");
    assert!(mir
        .bodies
        .values()
        .any(|body| body.locals.iter().any(|local| {
            local.binding == Some(binding.id)
                && local.kind == crate::MirLocalKind::IncomingWorkspace
        })));
    let found = crate::analysis::analyze_assembly(&mir).diagnostics;
    assert!(
        found.iter().all(|item| item.code != "RM-MIR0001"),
        "incoming workspace value must be assigned on entry: {found:#?}"
    );
    assert!(hir
        .assembly
        .bindings
        .iter()
        .any(|binding| binding.name.0 == "seen" && binding.role == runmat_hir::BindingRole::Local));
}

#[test]
fn dynamic_workspace_reads_defer_presence_to_runtime_without_proving_assignment() {
    use runmat_types::{ProgramFunctionId, RegionValueId};

    for source in [
        "run('worker'); after_run = generated + 1;",
        "eval('created = 41;'); observed = created;",
        "A = [9 8]; picked = A(end + eval('created = 1; created - 1')); observed = created;",
    ] {
        let parsed = runmat_parser::parse(source).expect("parse dynamic workspace source");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower dynamic workspace HIR");
        let binding = hir
            .assembly
            .bindings
            .iter()
            .find(|binding| binding.role == runmat_hir::BindingRole::ExternalWorkspace)
            .expect("dynamic effect exposes an external workspace read");
        let mir =
            crate::lowering::lower_assembly(&hir.assembly).expect("lower dynamic workspace MIR");
        let (function, local) = mir
            .bodies
            .iter()
            .find_map(|(function, body)| {
                body.locals.iter().find_map(|local| {
                    (local.binding == Some(binding.id)).then_some((*function, local))
                })
            })
            .expect("external read has a MIR slot");
        assert_eq!(local.kind, crate::MirLocalKind::ExternalWorkspace);

        let analysis = crate::analysis::analyze_assembly(&mir);
        assert!(
            analysis
                .diagnostics
                .iter()
                .all(|item| !matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
            "dynamic read is checked by the executor, not diagnosed as a local: {source}: {:#?}",
            analysis.diagnostics
        );
        let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
        let local = RegionValueId {
            function,
            local: u32::try_from(local.id.0).unwrap(),
        };
        assert!(analysis.program_points.iter().any(|point| {
            point.point.function == function
                && point.local(local).is_some_and(|fact| {
                    fact.assignment == crate::analysis::AssignmentFact::MaybeAssigned
                        && fact.fact.is_none()
                })
        }));
    }
}

#[test]
fn current_workspace_effect_rechecks_existing_bindings_without_proving_a_value() {
    use runmat_types::{ProgramFunctionId, RegionValueId};

    for source in [
        "function out = current_output()\n eval('out = 1;');\n end",
        "function out = conditional_local()\n if false; created = 0; end\n eval('created = 41;');\n out = created;\n end",
        "function out = run_into_output()\n run('worker');\n out = generated;\n end",
        "function out = indexed_effect()\n A = [9 8];\n picked = A(end + eval('out = 1; out - 1'));\n end",
        "function out = later_index_read()\n A = [9 8];\n picked = A(end + eval('out = 1; out - 1'), out);\n end",
        "function out = argument_effect()\n used = plus(eval('out = 1'), out);\n end",
        "function out = later_expansion_argument()\n C = {7};\n used = plus(C{end + eval('out = 1; out - 1')}, out);\n end",
        "function out = cell_after_effect()\n C = {7};\n eval('C = {9};');\n out = C{1};\n end",
        "function out = short_circuit_effect(flag)\n flag && eval('out = 1');\n end",
    ] {
        let parsed = runmat_parser::parse(source).expect("parse current-workspace mutation");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower current-workspace mutation");
        let mir = crate::lowering::lower_assembly(&hir.assembly)
            .expect("lower current-workspace mutation MIR");
        let analysis = crate::analysis::analyze_assembly(&mir);
        assert!(
            analysis
                .diagnostics
                .iter()
                .all(|item| !matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002" | "RM-TYPE-BRACE-INDEX")),
            "current-workspace presence is checked at execution: {source}: {:#?}",
            analysis.diagnostics
        );
        let (function, output) = mir
            .bodies
            .iter()
            .find_map(|(function, body)| {
                body.locals
                    .iter()
                    .find(|local| local.kind == crate::MirLocalKind::Output)
                    .map(|local| (*function, local.id))
            })
            .expect("function output local");
        let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
        let output = RegionValueId {
            function,
            local: u32::try_from(output.0).unwrap(),
        };
        assert!(analysis.program_points.iter().any(|point| {
            point.point.function == function
                && point.local(output).is_some_and(|fact| {
                    fact.assignment == crate::analysis::AssignmentFact::MaybeAssigned
                        && fact.fact.is_none()
                })
        }), "opaque effect does not prove the output value: {source}");
    }
}

#[test]
fn other_workspace_effect_and_call_arguments_do_not_initialize_current_locals() {
    for source in [
        "function out = base_only()\n evalin('base', 'out = 1;');\n end",
        "function out = base_index()\n A = [9 8];\n picked = A(end + evalin('base', 'out = 1; 0'));\n end",
        "function out = read_before_eval()\n eval(out);\n end",
        "function out = earlier_index_read()\n A = [9 8];\n picked = A(out, end + eval('out = 1; out - 1'));\n end",
        "function out = earlier_expansion_argument()\n C = {7};\n used = plus(out, C{end + eval('out = 1; out - 1')});\n end",
        "function out = earlier_binary_operand()\n used = out + eval('out = 1; 2');\n end",
        "function out = earlier_array_element()\n used = [out, eval('out = 1; 2')];\n end",
        "function out = no_effect()\n if false; created = 1; end\n out = created;\n end",
    ] {
        let found = diagnostics(source);
        assert!(
            found
                .iter()
                .any(|item| matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
            "unassigned current-scope read must stay diagnosed: {source}: {found:#?}"
        );
    }
}

#[test]
fn nested_current_effect_invalidates_preexisting_output_fact() {
    use runmat_types::{ProgramFunctionId, RegionValueId};

    let source = "function out = stale_after_nested_effect()\n out = 7;\n A = [9 8];\n picked = A(end + eval('out = 1; out - 1'), out);\n end";
    let parsed = runmat_parser::parse(source).expect("parse ordered nested effect");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("lower ordered nested effect HIR");
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower ordered nested effect MIR");
    let (function, output) = mir
        .bodies
        .iter()
        .find_map(|(function, body)| {
            body.locals
                .iter()
                .find(|local| local.kind == crate::MirLocalKind::Output)
                .map(|local| (*function, local.id))
        })
        .expect("function output local");
    let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
    let output = RegionValueId {
        function,
        local: u32::try_from(output.0).unwrap(),
    };
    let analysis = crate::analysis::analyze_assembly(&mir);
    assert!(analysis
        .diagnostics
        .iter()
        .all(|item| !matches!(item.code.as_str(), "RM-MIR0001" | "RM-MIR0002")));
    let final_point = analysis
        .program_points
        .iter()
        .rfind(|point| point.point.function == function)
        .expect("final function point");
    let fact = final_point.local(output).expect("output fact");
    assert_eq!(
        fact.assignment,
        crate::analysis::AssignmentFact::MaybeAssigned
    );
    assert!(
        fact.fact.is_none(),
        "opaque eval invalidates the old value proof"
    );
}

#[test]
fn declared_storage_does_not_infer_a_new_cell_from_brace_output() {
    use runmat_types::{ProgramFunctionId, ProgramPointId, RegionValueId, ValueKindFact};

    for declaration in ["persistent", "global"] {
        let source = format!(
            "function out = declared_cell_target()\n {declaration} item;\n [item{{2}}, other] = deal(9, 4);\n out = item;\n end"
        );
        let parsed = runmat_parser::parse(&source).expect("parse declared brace target");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower declared brace target HIR");
        let mir = crate::lowering::lower_assembly(&hir.assembly)
            .expect("lower declared brace target MIR");
        let analysis = crate::analysis::analyze_assembly(&mir);
        let (function, block, position, local) = mir
            .bodies
            .iter()
            .find_map(|(function, body)| {
                body.blocks.iter().find_map(|block| {
                    block
                        .statements
                        .iter()
                        .enumerate()
                        .find_map(|(position, statement)| {
                            let crate::MirStmtKind::MultiAssign { targets, .. } = &statement.kind
                            else {
                                return None;
                            };
                            targets.targets.iter().find_map(|target| {
                                let crate::MirOutputTarget::Sequence(
                                    crate::MirSequenceTarget::CellContents {
                                        base: crate::MirPlace::Local(local),
                                        ..
                                    },
                                ) = target
                                else {
                                    return None;
                                };
                                Some((function, block, position, local))
                            })
                        })
                })
            })
            .expect("source lowered to a prepared cell destination");
        let function = ProgramFunctionId(u32::try_from(function.0).unwrap());
        let point = ProgramPointId {
            function,
            block: u32::try_from(block.id.0).unwrap(),
            position: u32::try_from(position + 1).unwrap(),
        };
        let local = analysis
            .facts_at(point)
            .expect("post-assignment source fact")
            .local(RegionValueId {
                function,
                local: u32::try_from(local.0).unwrap(),
            })
            .expect("declared root fact");
        assert_eq!(
            local.assignment,
            crate::analysis::AssignmentFact::DefinitelyAssigned
        );
        assert!(
            !matches!(
                local.fact.as_ref().map(|fact| &fact.kind),
                Some(ValueKindFact::Cell(_))
            ),
            "{declaration} declaration must not become a proven Cell from a missing payload"
        );
    }
}

#[test]
fn current_eval_invalidates_a_visible_lexical_capture() {
    use runmat_types::{ProgramFunctionId, ProgramPointId, RegionValueId};

    let source = "function out = outer()\n shared = 7;\n out = inner();\n function result = inner()\n before = shared;\n eval('shared = 9;');\n result = shared;\n end\n end";
    let parsed = runmat_parser::parse(source).expect("parse nested lexical eval");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("lower nested lexical eval HIR");
    let mir =
        crate::lowering::lower_assembly(&hir.assembly).expect("lower nested lexical eval MIR");
    let (function_id, _) = mir
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "inner")
        .expect("nested function");
    let body = &mir.bodies[function_id];
    let capture = body
        .locals
        .iter()
        .find(|local| local.kind == crate::MirLocalKind::Capture)
        .expect("outer shared binding is captured");
    assert!(crate::analysis::workspace_effect::current_scope_bindings(body).contains(&capture.id));
    let (block, position) = body
        .blocks
        .iter()
        .find_map(|block| {
            block
                .statements
                .iter()
                .position(|statement| {
                    crate::analysis::workspace_effect::statement_exposes_current_workspace(
                        &statement.kind,
                    )
                })
                .map(|position| (block.id, position))
        })
        .expect("typed current eval statement");
    let analysis = crate::analysis::analyze_assembly(&mir);
    let function = ProgramFunctionId(u32::try_from(function_id.0).unwrap());
    let capture = RegionValueId {
        function,
        local: u32::try_from(capture.id.0).unwrap(),
    };
    let point = |position| ProgramPointId {
        function,
        block: u32::try_from(block.0).unwrap(),
        position: u32::try_from(position).unwrap(),
    };
    let before = analysis
        .facts_at(point(position))
        .and_then(|facts| facts.local(capture))
        .expect("capture fact before eval");
    assert!(
        before.fact.is_some(),
        "the capture had a pre-eval value fact"
    );
    let after = analysis
        .facts_at(point(position + 1))
        .and_then(|facts| facts.local(capture))
        .expect("capture fact after eval");
    assert_eq!(
        after.assignment,
        crate::analysis::AssignmentFact::MaybeAssigned
    );
    assert!(
        after.fact.is_none(),
        "current eval retires the captured value proof"
    );
}
