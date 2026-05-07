use runmat_hir::{lower, EnvironmentEffect, HirCallableRef, LoweringContext};
use runmat_mir::{
    analysis::{
        analyze_assembly, analyze_body, analyze_liveness, analyze_spawn_boundaries,
        diagnose_uninitialized_reads, summarize_body, summarize_spawn_safety, AccelEligibilityFact,
        AnalysisStore, FusibilityFact, InitFact, MirLocalKey, NominalDispatchHook,
        ParallelSafetyFact,
    },
    lowering::lower_assembly,
    AsyncBehaviorFact, CacheProduct, MirAggregateKind, MirBody, MirCallArg, MirLocalKind,
    MirOperand, MirOutputTarget, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
    ProductCacheKey,
};

fn lower_mir(src: &str) -> runmat_mir::MirAssembly {
    let ast = runmat_parser::parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap();
    lower_assembly(&hir.assembly).unwrap()
}

fn analyze_single_body(src: &str) -> (MirBody, AnalysisStore) {
    let mir = lower_mir(src);
    let body = mir.bodies.values().next().unwrap().clone();
    let mut store = AnalysisStore::default();
    analyze_body(&body, &mut store);
    (body, store)
}

fn first_local_of_kind(body: &MirBody, kind: MirLocalKind) -> runmat_mir::MirLocalId {
    body.locals
        .iter()
        .find(|local| local.kind == kind)
        .unwrap()
        .id
}

#[test]
fn simple_function_lowers_to_single_block_with_binding_locals() {
    let mir = lower_mir("function y = f(x); y = x + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 1);
    assert_eq!(body.locals.len(), 4);
    assert_eq!(body.source_map.function, Some(body.function));
    assert!(body.source_map.module.is_some());
    assert!(body.source_map.source_unit.is_some());
    assert_eq!(body.source_map.compatibility_mode, None);
    assert_eq!(body.source_map.enclosing_class, None);
    assert_eq!(body.source_map.locals.len(), body.locals.len());
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Binary(_, _, _),
            ..
        }
    ));
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Return(ref outputs) if outputs.len() == 1
    ));
}

#[test]
fn dataflow_marks_parameters_and_assigned_outputs_definitely_assigned() {
    let (body, store) = analyze_single_body("function y = f(x); y = x; end");
    let param = first_local_of_kind(&body, MirLocalKind::Parameter);
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: param,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
    let output_binding = body
        .locals
        .iter()
        .find(|local| local.id == output)
        .unwrap()
        .binding
        .unwrap();
    assert_eq!(
        store.bindings.get(&output_binding).unwrap().initialized,
        InitFact::DefinitelyAssigned
    );
}

#[test]
fn dataflow_joins_branch_assignment_as_maybe_assigned() {
    let (body, store) = analyze_single_body("function y = f(c); if c; y = 1; end; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::MaybeAssigned
    );
    let output_binding = body
        .locals
        .iter()
        .find(|local| local.id == output)
        .unwrap()
        .binding
        .unwrap();
    assert_eq!(
        store.bindings.get(&output_binding).unwrap().initialized,
        InitFact::MaybeAssigned
    );
}

#[test]
fn analyze_assembly_populates_binding_facts_by_semantic_id() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; end");
    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();
    let output_binding = body
        .locals
        .iter()
        .find(|local| matches!(local.kind, MirLocalKind::Output))
        .unwrap()
        .binding
        .unwrap();

    assert_eq!(
        store.bindings.get(&output_binding).unwrap().initialized,
        InitFact::MaybeAssigned
    );
}

#[test]
fn analyze_body_records_simple_numeric_local_and_binding_facts() {
    let (body, store) = analyze_single_body("function y = f(); y = 1; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
    let output_binding = body
        .locals
        .iter()
        .find(|local| local.id == output)
        .unwrap()
        .binding
        .unwrap();

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .ty,
        runmat_hir::TypeFact::Numeric {
            class: runmat_hir::NumericClass::Double,
            domain: runmat_hir::NumericDomain::Real,
        }
    );
    assert_eq!(
        store.bindings.get(&output_binding).unwrap().ty,
        runmat_hir::TypeFact::Numeric {
            class: runmat_hir::NumericClass::Double,
            domain: runmat_hir::NumericDomain::Real,
        }
    );
}

#[test]
fn analyze_body_records_simple_string_value_flow_fact() {
    let (body, store) = analyze_single_body("function y = f(); y = 'name'; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
    let fact = store
        .mir_locals
        .get(&MirLocalKey {
            function: body.function,
            local: output,
        })
        .unwrap();

    assert_eq!(fact.ty, runmat_hir::TypeFact::CharArray);
    assert_eq!(fact.shape, runmat_hir::ShapeFact::Scalar);
    assert_eq!(
        fact.value_flow,
        runmat_hir::ValueFlowFact::Single(runmat_hir::TypeFact::CharArray)
    );
}

#[test]
fn analyze_body_records_function_handle_value_flow_fact() {
    let mir = lower_mir("function h = f(); h = @target; end\nfunction y = target(x); y = x; end");
    let store = analyze_assembly(&mir);
    let summary = store
        .functions
        .values()
        .find(|summary| !summary.function_handles.is_empty())
        .unwrap();
    let function_fact = store
        .mir_locals
        .values()
        .find(|fact| matches!(fact.ty, runmat_hir::TypeFact::Function(_)))
        .unwrap();

    assert!(matches!(
        function_fact.value_flow,
        runmat_hir::ValueFlowFact::Single(runmat_hir::TypeFact::Function(_))
    ));
    assert!(matches!(
        summary.function_handles[0],
        runmat_hir::FunctionHandleTarget::Function(_)
    ));
}

#[test]
fn analyze_body_records_tensor_and_cell_aggregate_facts() {
    let (tensor_body, tensor_store) = analyze_single_body("function y = f(); y = [1, 2]; end");
    let tensor_output = first_local_of_kind(&tensor_body, MirLocalKind::Output);
    let tensor_fact = &tensor_store.mir_locals[&MirLocalKey {
        function: tensor_body.function,
        local: tensor_output,
    }];

    assert!(matches!(
        tensor_fact.ty,
        runmat_hir::TypeFact::Tensor(runmat_hir::TensorTypeFact {
            element: runmat_hir::TensorElementDomainFact::Numeric {
                class: runmat_hir::NumericClass::Double,
                domain: runmat_hir::NumericDomain::Real,
            },
            ..
        })
    ));
    assert_eq!(
        tensor_fact.shape,
        runmat_hir::ShapeFact::Shaped {
            dims: vec![runmat_hir::DimFact::Known(1), runmat_hir::DimFact::Known(2)]
        }
    );
    assert!(matches!(
        tensor_fact.value_flow,
        runmat_hir::ValueFlowFact::Single(runmat_hir::TypeFact::Tensor(_))
    ));
    assert!(matches!(
        tensor_fact.tensor_element_domain,
        Some(runmat_hir::TensorElementDomainFact::Numeric { .. })
    ));

    let (cell_body, cell_store) = analyze_single_body("function y = f(); y = {1, 2}; end");
    let cell_output = first_local_of_kind(&cell_body, MirLocalKind::Output);
    let cell_fact = &cell_store.mir_locals[&MirLocalKey {
        function: cell_body.function,
        local: cell_output,
    }];

    assert_eq!(cell_fact.ty, runmat_hir::TypeFact::Cell);
    assert_eq!(
        cell_fact.shape,
        runmat_hir::ShapeFact::Shaped {
            dims: vec![runmat_hir::DimFact::Known(1), runmat_hir::DimFact::Known(2)]
        }
    );
    assert_eq!(
        cell_fact.value_flow,
        runmat_hir::ValueFlowFact::Single(runmat_hir::TypeFact::Cell)
    );
}

#[test]
fn analyze_body_records_operator_and_expansion_semantic_facts() {
    let (body, store) = analyze_single_body("function y = f(x); y = x + 1; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
    let fact = &store.mir_locals[&MirLocalKey {
        function: body.function,
        local: output,
    }];

    assert_eq!(fact.operator, Some(runmat_hir::OperatorKind::Add));
    assert_eq!(
        fact.expansion,
        Some(runmat_hir::ExpansionSemantics::ImplicitExpansion)
    );
}

#[test]
fn analyze_body_joins_simple_facts_across_cfg_paths() {
    let (body, store) = analyze_single_body("function y = f(c); y = 1; if c; y = 's'; end; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
    let fact = &store.mir_locals[&MirLocalKey {
        function: body.function,
        local: output,
    }];

    assert_eq!(fact.ty, runmat_hir::TypeFact::Unknown);
    assert_eq!(fact.shape, runmat_hir::ShapeFact::Scalar);
}

#[test]
fn analyze_body_records_future_and_task_async_value_facts() {
    let mir = lower_mir(
        "function y = f(); fut = make(); task = spawn(fut); y = 1; end
async function y = make(); y = 1; end",
    );
    let store = analyze_assembly(&mir);

    let async_values: Vec<_> = store
        .mir_locals
        .values()
        .filter_map(|fact| fact.async_value.as_ref())
        .collect();

    assert!(async_values
        .iter()
        .any(|fact| matches!(fact, runmat_hir::AsyncValueFact::Future(_))));
    assert!(async_values
        .iter()
        .any(|fact| matches!(fact, runmat_hir::AsyncValueFact::TaskHandle(_))));
}

#[test]
fn direct_spawn_of_anonymous_function_uses_function_handle_temp_operand() {
    let mir = lower_mir("function y = f(); task = spawn(@(x) x); y = 1; end");
    let store = analyze_assembly(&mir);
    let body = mir
        .bodies
        .values()
        .find(|body| {
            body.blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|stmt| {
                    matches!(
                        stmt.kind,
                        MirStmtKind::Assign {
                            value: MirRvalue::Spawn(_),
                            ..
                        }
                    )
                })
        })
        .unwrap();

    let mut saw_handle_temp = false;
    let mut spawn_operand = None;
    for stmt in &body.blocks[0].statements {
        match &stmt.kind {
            MirStmtKind::Assign {
                place: MirPlace::Local(local),
                value:
                    MirRvalue::Use(MirOperand::FunctionHandle(
                        runmat_hir::FunctionHandleTarget::Anonymous(_),
                    )),
            } => saw_handle_temp = body.locals[local.0].kind == MirLocalKind::Temporary,
            MirStmtKind::Assign {
                value: MirRvalue::Spawn(operand),
                ..
            } => spawn_operand = Some(operand),
            _ => {}
        }
    }

    assert!(saw_handle_temp);
    assert!(matches!(spawn_operand, Some(MirOperand::Local(_))));

    let handle_expr = body
        .source_map
        .locals
        .iter()
        .find_map(|source| {
            source.expr.filter(|_| {
                body.locals[source.local.0].kind == MirLocalKind::Temporary
                    && matches!(
                        store.mir_locals[&MirLocalKey {
                            function: body.function,
                            local: source.local,
                        }]
                            .ty,
                        runmat_hir::TypeFact::Function(_)
                    )
            })
        })
        .unwrap();
    assert!(matches!(
        store.expressions[&handle_expr].ty,
        runmat_hir::TypeFact::Function(_)
    ));
    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(
            boundary.safety,
            runmat_hir::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_hir::SpawnSafetyReason::UnknownDynamicCapture
            }
        )));
}

#[test]
fn async_function_call_lowers_to_lazy_future_value() {
    let mir = lower_mir(
        "function t = caller(x); t = callee(x); end
async function y = callee(x); y = x; end",
    );
    let caller = mir
        .bodies
        .values()
        .find(|body| {
            body.blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|stmt| {
                    matches!(
                        stmt.kind,
                        MirStmtKind::Assign {
                            value: MirRvalue::Future { .. },
                            ..
                        }
                    )
                })
        })
        .unwrap();

    assert!(caller
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| {
            matches!(
                &stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Future { args, requested_outputs, syntax, .. },
                    ..
                } if args.len() == 1
                    && matches!(requested_outputs, runmat_hir::RequestedOutputCount::One)
                    && matches!(syntax, runmat_hir::CallSyntax::Plain)
            )
        }));
}

#[test]
fn remaining_semantic_expr_and_import_forms_lower_to_mir() {
    let mir = lower_mir("function y = f(x); y = x.field; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Member { .. },
                    ..
                }
            )
        }));
}

#[test]
fn analyze_body_populates_expression_facts_from_source_map() {
    let (body, store) = analyze_single_body("function y = f(x); y = x + 1; end");
    let expr = body
        .source_map
        .statements
        .iter()
        .find_map(|record| record.expr)
        .unwrap();

    assert!(store.expressions.contains_key(&expr));
    assert_eq!(store.expressions[&expr].ty, runmat_hir::TypeFact::Unknown);
}

#[test]
fn dataflow_widens_loop_assignment_as_maybe_assigned() {
    let (body, store) = analyze_single_body("function y = f(c); while c; y = 1; end; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::MaybeAssigned
    );
}

#[test]
fn diagnostics_report_unassigned_local_read() {
    let mir = lower_mir("function y = f(); z = y; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0001"
            && diagnostic.category.as_deref() == Some("definite-assignment")
            && diagnostic.primary.span.start < diagnostic.primary.span.end
    }));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_branch() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; x = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn summary_records_function_outputs_and_store_entry() {
    let mir = lower_mir("function y = f(x); y = x; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(summary.function, body.function);
    assert_eq!(summary.abi.fixed_inputs.len(), 1);
    assert_eq!(summary.abi.fixed_outputs.len(), 1);
    assert!(summary.abi.implicit_nargin.is_some());
    assert!(summary.abi.implicit_nargout.is_some());
    assert_eq!(summary.outputs.len(), 1);
    assert!(store.functions.contains_key(&body.function));
    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::NeverSuspends)
    );
}

#[test]
fn analyze_assembly_projects_output_type_facts_into_summary() {
    let mir = lower_mir("function y = f(); y = 1; end");

    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();
    let summary = store.functions.get(&body.function).unwrap();

    assert_eq!(
        summary.outputs,
        vec![runmat_hir::TypeFact::Numeric {
            class: runmat_hir::NumericClass::Double,
            domain: runmat_hir::NumericDomain::Real,
        }]
    );
    assert_eq!(summary.output_shapes, vec![runmat_hir::ShapeFact::Scalar]);
    assert_eq!(
        summary.output_value_flows,
        vec![runmat_hir::ValueFlowFact::Single(
            runmat_hir::TypeFact::Numeric {
                class: runmat_hir::NumericClass::Double,
                domain: runmat_hir::NumericDomain::Real,
            }
        )]
    );
    assert_eq!(summary.output_async_values, vec![None]);
}

#[test]
fn summary_preserves_variadic_function_abi() {
    let mir = lower_mir("function varargout = f(x, varargin); varargout = varargin; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(summary.abi.fixed_inputs.len(), 2);
    assert_eq!(summary.abi.varargin, Some(summary.abi.fixed_inputs[1]));
    assert_eq!(summary.abi.fixed_outputs.len(), 1);
    assert_eq!(summary.abi.varargout, Some(summary.abi.fixed_outputs[0]));
}

#[test]
fn summary_records_requested_output_sensitive_call_facts() {
    let mir = lower_mir("function y = f(varargin); [a, b] = g(varargin{:}); y = a; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(summary.calls.len(), 1);
    assert!(matches!(
        summary.calls[0].requested_outputs,
        runmat_hir::RequestedOutputCount::Exactly(2)
    ));
    assert_eq!(summary.calls[0].arg_count, 1);
    assert_eq!(summary.calls[0].expansion_arg_count, 1);
    assert!(matches!(
        summary.calls[0].async_behavior,
        AsyncBehaviorFact::MaySuspend
    ));
    assert_eq!(summary.calls[0].dispatch, NominalDispatchHook::Dynamic);
    assert!(summary
        .requested_output_sensitive
        .iter()
        .any(|(requested, outputs)| matches!(
            requested,
            runmat_hir::RequestedOutputCount::Exactly(1)
        ) && outputs.len() == 1));
}

#[test]
fn summary_marks_unresolved_calls_as_unknown_call_barriers() {
    let mir = lower_mir("function y = f(x); y = unresolved_target(x); end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert!(summary.may_call_unknown);
    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::MaySuspend)
    );
    assert!(matches!(summary.fusibility, FusibilityFact::NonFusible(_)));
    assert!(matches!(
        summary.accel_eligibility,
        AccelEligibilityFact::Ineligible(_)
    ));
}

#[test]
fn summary_marks_spawn_as_requiring_async_runtime() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::RequiresAsyncRuntime)
    );
}

#[test]
fn spawn_boundary_analysis_records_spawn_site() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();

    let boundaries = analyze_spawn_boundaries(body);

    assert_eq!(boundaries.len(), 1);
    assert!(matches!(
        boundaries[0].safety,
        runmat_hir::SpawnSafetyFact::RequiresIsolation
    ));
    assert!(matches!(boundaries[0].future, MirOperand::Local(_)));
}

#[test]
fn spawn_safety_summary_is_conservative_for_spawn_sites() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();

    let summary = summarize_spawn_safety(body);

    assert_eq!(summary.function, body.function);
    assert!(matches!(
        summary.safety,
        runmat_hir::SpawnSafetyFact::RequiresIsolation
    ));
}

#[test]
fn analyze_assembly_rejects_spawned_future_with_mutable_lexical_capture() {
    let mir = lower_mir(
        "function y = outer(); acc = 0; async function z = bump(); acc = acc + 1; z = acc; end; fut = bump(); task = spawn(fut); y = acc; end",
    );

    let store = analyze_assembly(&mir);

    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003" && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(
            boundary.safety,
            runmat_hir::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_hir::SpawnSafetyReason::MutableLexicalCapture
            }
        )));
}

#[test]
fn analyze_assembly_rejects_spawned_future_with_lexical_capture_read() {
    let mir = lower_mir(
        "function y = outer(); acc = 1; async function z = read_acc(); z = acc; end; fut = read_acc(); task = spawn(fut); y = acc; end",
    );

    let store = analyze_assembly(&mir);

    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(
            boundary.safety,
            runmat_hir::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_hir::SpawnSafetyReason::MutableLexicalCapture
            }
        )));
}

#[test]
fn spawn_safety_uses_future_target_visible_at_spawn_site() {
    let mir = lower_mir(
        "function y = outer(); cap = 1; async function z = safe(); z = 1; end; async function z = unsafe(); z = cap; end; fut = safe(); task = spawn(fut); fut = unsafe(); y = cap; end",
    );

    let store = analyze_assembly(&mir);

    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(boundary.safety, runmat_hir::SpawnSafetyFact::SpawnSafe)));
    assert!(!store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003" && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn spawn_safety_joins_future_targets_across_cfg_paths() {
    let mir = lower_mir(
        "function y = outer(c); cap = 1; async function z = safe(); z = 1; end; async function z = unsafe(); z = cap; end; if c; fut = safe(); else; fut = unsafe(); end; task = spawn(fut); y = cap; end",
    );

    let store = analyze_assembly(&mir);

    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(
            boundary.safety,
            runmat_hir::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_hir::SpawnSafetyReason::MutableLexicalCapture
            }
        )));
}

#[test]
fn analyze_assembly_rejects_spawned_future_with_unknown_target() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");

    let store = analyze_assembly(&mir);

    assert!(store
        .spawn_boundaries
        .values()
        .flatten()
        .any(|boundary| matches!(
            boundary.safety,
            runmat_hir::SpawnSafetyFact::NotSpawnSafe {
                reason: runmat_hir::SpawnSafetyReason::UnknownDynamicCapture
            }
        )));
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003" && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn dataflow_marks_spawn_as_task_handle_even_when_safety_diagnostic_rejects_target() {
    let mir = lower_mir("function y = f(); task = spawn(@(x) x); y = 1; end");
    let store = analyze_assembly(&mir);

    assert!(store.mir_locals.values().any(|fact| matches!(
        fact.async_value,
        Some(runmat_hir::AsyncValueFact::TaskHandle(_))
    )));
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003" && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn analyze_assembly_populates_store_products_by_function() {
    let mir = lower_mir("async function y = f(g, x); t = spawn(g); await(t); y = x; end");

    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();

    assert!(store.functions.contains_key(&body.function));
    let module = body.source_map.module.unwrap();
    assert!(store.modules.contains_key(&module));
    assert!(store.modules[&module].functions.contains(&body.function));
    assert!(store.modules[&module].source_unit.is_some());
    assert_eq!(store.modules[&module].compatibility_mode, None);
    assert!(store.liveness.contains_key(&body.function));
    assert!(store.spawn_boundaries.contains_key(&body.function));
    assert_eq!(store.spawn_boundaries.get(&body.function).unwrap().len(), 1);
    assert_eq!(
        store
            .liveness
            .get(&body.function)
            .unwrap()
            .live_across_await
            .len(),
        1
    );
}

#[test]
fn analyze_assembly_scopes_mir_local_facts_by_function() {
    let mir = lower_mir("function y = f(x); y = x; end\nfunction z = g(a); z = a; end");

    let store = analyze_assembly(&mir);
    let expected_locals: usize = mir.bodies.values().map(|body| body.locals.len()).sum();

    assert_eq!(store.mir_locals.len(), expected_locals);
    for body in mir.bodies.values() {
        for local in &body.locals {
            assert!(store.mir_locals.contains_key(&MirLocalKey {
                function: body.function,
                local: local.id,
            }));
        }
    }
}

#[test]
fn analyze_assembly_collects_structured_diagnostics() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; z = y; end");

    let store = analyze_assembly(&mir);

    assert!(store
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"
            && diagnostic.category.as_deref() == Some("definite-assignment")
            && diagnostic.help.as_deref()
                == Some("assign this local on every control-flow path before reading it")));
}

#[test]
fn analyze_assembly_collects_semantic_marker_diagnostics() {
    let mut mir = lower_mir("function y = f(x); y = x; end");
    let body = mir.bodies.values_mut().next().unwrap();
    let local = first_local_of_kind(body, MirLocalKind::Parameter);
    body.blocks[0].statements[0].kind = MirStmtKind::Expr(MirRvalue::Call(runmat_mir::MirCall {
        callee: HirCallableRef::Unresolved(runmat_hir::QualifiedName(vec![
            runmat_hir::SymbolName("sink".into()),
        ])),
        args: vec![MirCallArg::Expansion(MirOperand::Local(local))],
        syntax: runmat_hir::CallSyntax::Plain,
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        async_behavior: runmat_mir::AsyncBehaviorFact::MaySuspend,
        effects: runmat_builtins::BuiltinEffects::unknown(),
        purity: runmat_builtins::BuiltinPurity::Impure,
        semantic_kind: runmat_builtins::BuiltinSemanticKind::General,
    }));

    let store = analyze_assembly(&mir);

    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0004" && diagnostic.category.as_deref() == Some("comma-list")
    }));
}

#[test]
fn analysis_store_diagnostics_serialize_with_store() {
    let mir = lower_mir(
        "async function y = f(c, g); if c; y = 1; end; t = spawn(g); await(t); z = y; end",
    );
    let store = analyze_assembly(&mir);

    let json = serde_json::to_string(&store).unwrap();
    let roundtrip: AnalysisStore = serde_json::from_str(&json).unwrap();

    assert!(json.contains("RM-MIR0002"));
    assert!(json.contains("definite-assignment"));
    assert_eq!(roundtrip, store);
    assert!(!roundtrip.modules.is_empty());
    assert!(!roundtrip.expressions.is_empty());
    assert!(!roundtrip.liveness.is_empty());
    assert!(!roundtrip.spawn_boundaries.is_empty());
}

#[test]
fn product_cache_key_serializes_semantic_product_dependencies() {
    let function_path = runmat_hir::DefPath {
        package: runmat_hir::PackageName("pkg".into()),
        module: runmat_hir::QualifiedName(vec![runmat_hir::SymbolName("mod".into())]),
        item: vec![runmat_hir::DefPathSegment::Function(
            runmat_hir::SymbolName("f".into()),
        )],
    };
    let key = ProductCacheKey {
        product: CacheProduct::FunctionSummary(function_path),
        source_hash: "source".into(),
        manifest_hash: "manifest".into(),
        dependency_graph_hash: "deps".into(),
        config_hash: "config".into(),
        compiler_version: "compiler".into(),
    };

    let json = serde_json::to_string(&key).unwrap();
    let roundtrip: ProductCacheKey = serde_json::from_str(&json).unwrap();

    assert_eq!(roundtrip, key);
    assert!(json.contains("FunctionSummary"));
    assert!(json.contains("dependency_graph_hash"));

    let module_key = ProductCacheKey {
        product: CacheProduct::ModuleSummary(runmat_hir::QualifiedName(vec![
            runmat_hir::SymbolName("mod".into()),
        ])),
        source_hash: "source".into(),
        manifest_hash: "manifest".into(),
        dependency_graph_hash: "deps".into(),
        config_hash: "config".into(),
        compiler_version: "compiler".into(),
    };

    let module_json = serde_json::to_string(&module_key).unwrap();
    let module_roundtrip: ProductCacheKey = serde_json::from_str(&module_json).unwrap();

    assert_eq!(module_roundtrip, module_key);
    assert!(module_json.contains("ModuleSummary"));
}

#[test]
fn await_statement_lowers_to_await_terminator_with_resume_block() {
    let mir = lower_mir("async function y = f(g); await(g); y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Await { resume, .. } = body.blocks[0].terminator.kind else {
        panic!("expected await terminator");
    };
    assert_eq!(resume, body.blocks[1].id);
    assert_eq!(body.blocks[1].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn await_terminator_marks_summary_as_requiring_async_runtime() {
    let mir = lower_mir("async function y = f(g); await(g); y = 1; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert_eq!(
        summary.effects.async_behavior,
        Some(AsyncBehaviorFact::RequiresAsyncRuntime)
    );
}

#[test]
fn await_assignment_lowers_to_result_place() {
    let mir = lower_mir("async function y = f(g); y = await(g); end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Await { result, resume, .. } = &body.blocks[0].terminator.kind else {
        panic!("expected await terminator");
    };
    assert!(matches!(result, Some(MirPlace::Local(_))));
    assert_eq!(*resume, body.blocks[1].id);
}

#[test]
fn nested_await_expression_lowers_to_temp_and_resume_block() {
    let mir = lower_mir("async function y = f(g); y = 1 + await(g); end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Await { result, resume, .. } = &body.blocks[0].terminator.kind else {
        panic!("expected await terminator");
    };
    assert!(matches!(result, Some(MirPlace::Local(_))));
    let resume_block = body
        .blocks
        .iter()
        .find(|block| block.id == *resume)
        .unwrap();
    assert!(resume_block.statements.iter().any(|stmt| matches!(
        stmt.kind,
        MirStmtKind::Assign {
            value: MirRvalue::Binary(_, _, _),
            ..
        }
    )));
}

#[test]
fn dataflow_marks_await_assignment_output_definitely_assigned() {
    let (body, store) = analyze_single_body("async function y = f(g); y = await(g); end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);

    assert_eq!(
        store
            .mir_locals
            .get(&MirLocalKey {
                function: body.function,
                local: output,
            })
            .unwrap()
            .initialized,
        InitFact::DefinitelyAssigned
    );
}

#[test]
fn liveness_records_locals_live_across_await() {
    let mir = lower_mir("async function y = f(g, x); await(g); y = x; end");
    let body = mir.bodies.values().next().unwrap();

    let facts = analyze_liveness(body);

    assert_eq!(facts.live_across_await.len(), 1);
    let live = &facts.live_across_await[0].1;
    let param_locals: Vec<_> = body
        .locals
        .iter()
        .filter(|local| matches!(local.kind, MirLocalKind::Parameter))
        .map(|local| local.id)
        .collect();
    assert!(param_locals.iter().any(|local| live.contains(local)));
    assert!(live.len() >= 2);
}

#[test]
fn liveness_does_not_mark_locals_defined_after_await_as_live_across() {
    let mir = lower_mir("async function y = f(g); y = await(g); z = 1; y = z; end");
    let body = mir.bodies.values().next().unwrap();

    let facts = analyze_liveness(body);
    let live = &facts.live_across_await[0].1;
    let z_local = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign {
                place: MirPlace::Local(local),
                value: MirRvalue::Use(MirOperand::Constant(runmat_mir::MirConstant::Number(value))),
            } if value == "1" => Some(*local),
            _ => None,
        })
        .unwrap();

    assert!(!live.contains(&z_local));
}

#[test]
fn global_statement_lowers_to_workspace_effect() {
    let mir = lower_mir("function y = f(); global g; y = 1; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_hir::WorkspaceEffect::MutatesGlobal,
            ..
        }
    ));
    let global_local = match &body.blocks[0].statements[0].kind {
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings[0],
        _ => unreachable!(),
    };
    let summary = summarize_body(body, &mut store);
    let global_binding = body.locals[global_local.0].binding.unwrap();
    assert!(summary.writes_globals.contains(&global_binding));
    assert!(matches!(summary.fusibility, FusibilityFact::NonFusible(_)));
    assert!(matches!(
        summary.parallel_safety,
        ParallelSafetyFact::WritesSharedState
    ));
    assert!(matches!(
        summary.accel_eligibility,
        AccelEligibilityFact::Ineligible(_)
    ));
}

#[test]
fn analyze_assembly_aggregates_module_effect_summaries() {
    let mir = lower_mir("function y = f(); global g; y = unresolved_target(g); end");

    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();
    let module = body.source_map.module.unwrap();
    let summary = store.modules.get(&module).unwrap();

    assert!(summary.functions.contains(&body.function));
    assert!(summary
        .workspace
        .contains(&runmat_hir::WorkspaceEffect::MutatesGlobal));
    assert!(summary.may_call_unknown);
}

#[test]
fn persistent_statement_lowers_to_summary_workspace_effect() {
    let mir = lower_mir("function y = f(); persistent p; y = 1; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    let summary = summarize_body(body, &mut store);

    assert!(summary
        .effects
        .workspace
        .contains(&runmat_hir::WorkspaceEffect::MutatesPersistent));
    let persistent_local = match &body.blocks[0].statements[0].kind {
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings[0],
        _ => unreachable!(),
    };
    let persistent_binding = body.locals[persistent_local.0].binding.unwrap();
    assert!(summary.writes_persistents.contains(&persistent_binding));
}

#[test]
fn source_workspace_and_environment_calls_lower_to_effect_markers() {
    let mir = lower_mir("function y = f(); eval('x = 1'); addpath('/tmp'); y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::WorkspaceEffect {
                    effect: runmat_hir::WorkspaceEffect::DynamicEval,
                    ..
                }
            )
        }));
    assert!(body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::EnvironmentEffect(runmat_hir::EnvironmentEffect::PathMutation)
            )
        }));
}

#[test]
fn summary_records_explicit_environment_effects() {
    let mir = lower_mir("function y = f(x); y = x; end");
    let mut body = mir.bodies.values().next().unwrap().clone();
    let span = body.blocks[0].statements[0].span;
    body.blocks[0].statements.insert(
        0,
        MirStmt {
            kind: MirStmtKind::EnvironmentEffect(EnvironmentEffect::FunctionCacheInvalidation),
            span,
        },
    );
    let mut store = AnalysisStore::default();

    let summary = summarize_body(&body, &mut store);

    assert_eq!(
        summary.effects.environment,
        vec![EnvironmentEffect::FunctionCacheInvalidation]
    );
    assert!(matches!(summary.fusibility, FusibilityFact::NonFusible(_)));
    assert!(matches!(
        summary.parallel_safety,
        ParallelSafetyFact::WritesSharedState
    ));
    assert!(matches!(
        summary.accel_eligibility,
        AccelEligibilityFact::Ineligible(_)
    ));
}

#[test]
fn direct_function_call_preserves_callee_and_requested_outputs() {
    let mir = lower_mir("function y = g(x); y = f(x); end\nfunction z = f(a); z = a; end");
    let stmt = mir
        .bodies
        .values()
        .flat_map(|body| &body.blocks[0].statements)
        .find(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Call(_),
                    ..
                }
            )
        })
        .unwrap();
    let MirStmtKind::Assign {
        value: MirRvalue::Call(call),
        ..
    } = &stmt.kind
    else {
        panic!("expected call assignment");
    };

    assert!(matches!(call.callee, HirCallableRef::Function(_)));
    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::One
    ));
}

#[test]
fn function_summary_propagates_user_callee_effects() {
    let mir =
        lower_mir("function y = g(); y = f(); end\nfunction z = f(); eval('x = 1'); z = 1; end");
    let store = analyze_assembly(&mir);
    let caller = store
        .functions
        .values()
        .find(|summary| {
            summary
                .calls
                .iter()
                .any(|call| matches!(call.callee, HirCallableRef::Function(_)))
        })
        .unwrap();

    assert!(caller
        .effects
        .workspace
        .contains(&runmat_hir::WorkspaceEffect::DynamicEval));
    assert!(matches!(caller.fusibility, FusibilityFact::NonFusible(_)));
    assert!(matches!(
        caller.accel_eligibility,
        AccelEligibilityFact::Ineligible(_)
    ));
}

#[test]
fn function_summary_records_async_future_dependency_edges() {
    let mir = lower_mir(
        "function y = g(t); fut = f(t); y = 1; end\nasync function z = f(t); z = await(t); end",
    );
    let store = analyze_assembly(&mir);
    let caller = store
        .functions
        .values()
        .find(|summary| {
            summary
                .calls
                .iter()
                .any(|call| matches!(call.callee, HirCallableRef::Function(_)))
        })
        .unwrap();

    assert!(matches!(
        caller.effects.async_behavior,
        Some(runmat_mir::AsyncBehaviorFact::RequiresAsyncRuntime)
    ));
}

#[test]
fn mir_call_records_explicit_async_behavior() {
    let mir = lower_mir("function y = f(x); y = unknown_call(x); end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| matches!(
            &stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::Call(call),
                ..
            } if matches!(call.async_behavior, runmat_mir::AsyncBehaviorFact::MaySuspend)
        )));
}

#[test]
fn command_call_lowers_to_zero_output_call_with_string_args() {
    let mir = lower_mir("function y = f()\nformat long\ny = 1\nend");
    let body = mir.bodies.values().next().unwrap();

    let MirStmtKind::Expr(MirRvalue::Call(call)) = &body.blocks[0].statements[0].kind else {
        panic!("expected command call expression");
    };

    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::Zero
    ));
    assert_eq!(call.syntax, runmat_hir::CallSyntax::Command);
    assert_eq!(call.args.len(), 1);
    assert!(matches!(
        call.args[0],
        MirCallArg::Single(MirOperand::Constant(runmat_mir::MirConstant::String(ref value))) if value.0 == "long"
    ));
}

#[test]
fn tensor_literal_lowers_to_mir_aggregate() {
    let mir = lower_mir("function y = make_tensor(x); y = [x, x + 1]; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary)));
    assert!(body
        .source_map
        .locals
        .iter()
        .any(|source| source.binding.is_none() && source.expr.is_some()));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Aggregate {
                kind: MirAggregateKind::Tensor,
                rows: 1,
                cols: 2,
                ref elements,
            },
            ..
        } if elements.len() == 2
    ));
}

#[test]
fn cell_literal_lowers_to_mir_aggregate() {
    let mir = lower_mir("function y = make_cell(x); y = {x, x + 1}; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Aggregate {
                kind: MirAggregateKind::Cell,
                rows: 1,
                cols: 2,
                ref elements,
            },
            ..
        } if elements.len() == 2
    ));
}

#[test]
fn complex_call_arguments_lower_through_temporary_locals() {
    let mir = lower_mir("function y = g(x); y = f(x + 1); end\nfunction z = f(a); z = a; end");
    let body = mir
        .bodies
        .values()
        .find(|body| body.blocks[0].statements.len() == 2)
        .unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary) && local.binding.is_none()));
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            place: MirPlace::Local(_),
            value: MirRvalue::Binary(_, _, _),
        }
    ));

    let MirStmtKind::Assign {
        value: MirRvalue::Call(call),
        ..
    } = &body.blocks[0].statements[1].kind
    else {
        panic!("expected call assignment after temp");
    };
    assert!(matches!(
        call.args[0],
        MirCallArg::Single(MirOperand::Local(_))
    ));
}

#[test]
fn function_argument_expansion_lowers_to_expansion_call_arg() {
    let mir = lower_mir("function y = f(varargin); y = g(varargin{:}); end");
    let body = mir.bodies.values().next().unwrap();

    let stmt = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .find(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Call(_),
                    ..
                }
            )
        })
        .unwrap();
    let MirStmtKind::Assign {
        value: MirRvalue::Call(call),
        ..
    } = &stmt.kind
    else {
        panic!("expected call assignment");
    };

    assert_eq!(call.args.len(), 1);
    assert!(matches!(
        call.args[0],
        MirCallArg::Expansion(MirOperand::Local(_))
    ));
}

#[test]
fn nested_binary_operands_lower_through_temporary_locals() {
    let mir = lower_mir("function y = calc(a, b, c); y = (a + b) * c; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks[0].statements.len(), 2);
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::Assign {
            place: MirPlace::Local(_),
            value: MirRvalue::Binary(_, _, _),
        }
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Binary(MirOperand::Local(_), _, _),
            ..
        }
    ));
}

#[test]
fn multi_assign_preserves_discard_targets_and_requested_outputs() {
    let mir = lower_mir("function idx = pick(x); [~, idx] = max(x); end");
    let body = mir.bodies.values().next().unwrap();

    let MirStmtKind::MultiAssign {
        targets,
        value: MirRvalue::Call(call),
    } = &body.blocks[0].statements[0].kind
    else {
        panic!("expected multi-output call assignment");
    };

    assert_eq!(targets.targets.len(), 2);
    assert!(matches!(targets.targets[0], MirOutputTarget::Discard));
    assert!(matches!(targets.targets[1], MirOutputTarget::Place(_)));
    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::Exactly(2)
    ));
}

#[test]
fn indexed_assignment_lowers_to_index_place() {
    let mir = lower_mir("function y = write_index(x); x(1) = 2; y = x; end");
    let body = mir.bodies.values().next().unwrap();
    let mut store = AnalysisStore::default();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::IndexedAssign
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateArrayByIndex
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::Index(_, _),
            ..
        }
    ));
    let summary = summarize_body(body, &mut store);
    assert!(summary.place_mutations.iter().any(|mutation| {
        mutation.kind == runmat_hir::PlaceMutationKind::IndexedAssign
            && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateArrayByIndex
    }));
}

#[test]
fn indexed_empty_assignment_lowers_to_delete_mutation() {
    let mir = lower_mir("function y = delete_index(x); x(1) = []; y = x; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::Delete
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::ExistingOnly
    ));
}

#[test]
fn index_components_lower_to_mir_operands() {
    let mir = lower_mir("function y = read_index(a, i); y = a(i + 1); end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body
        .locals
        .iter()
        .any(|local| matches!(local.kind, MirLocalKind::Temporary)));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            value: MirRvalue::Index { .. },
            ..
        }
    ));
}

#[test]
fn diagnostics_report_unassigned_index_operand_read() {
    let mir = lower_mir("function y = read_index(a); y = a(y); end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0001"));
}

#[test]
fn member_assignment_lowers_to_member_place() {
    let mir = lower_mir("function s = write_member(s); s.value = 2; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::MemberAssign
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateStructFieldPath
    ));
    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::Member(_, _),
            ..
        }
    ));
}

#[test]
fn dynamic_member_assignment_lowers_to_dynamic_member_place() {
    let mir = lower_mir("function s = write_member(s, name); s.(name) = 2; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::DynamicMember(_, MirOperand::Local(_)),
            ..
        }
    ));
}

#[test]
fn if_statement_lowers_to_branch_blocks_and_merge() {
    let mir = lower_mir("function y = choose(c, x); if c; y = x; else; y = 0; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn if_statement_flows_to_following_statements() {
    let mir = lower_mir("function y = choose(c); if c; y = 1; else; y = 2; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert_eq!(body.blocks[3].statements.len(), 1);
    assert!(matches!(
        body.blocks[3].statements[0].kind,
        MirStmtKind::Assign { .. }
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn elseif_statement_lowers_to_nested_branch_cfg() {
    let mir = lower_mir(
        "function y = choose(c, d); if c; y = 1; elseif d; y = 2; else; y = 3; end; y = y + 1; end",
    );
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(
        body.blocks
            .iter()
            .filter(|block| matches!(block.terminator.kind, MirTerminatorKind::Branch { .. }))
            .count()
            >= 2
    );
    assert!(body.blocks.iter().any(|block| block.statements.len() == 1
        && matches!(block.terminator.kind, MirTerminatorKind::Return(_))));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_elseif_without_else() {
    let mir = lower_mir("function y = choose(c, d); if c; y = 1; elseif d; y = 2; end; z = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn switch_statement_lowers_to_switch_blocks_and_continuation() {
    let mir = lower_mir(
        "function y = choose(x); switch x; case 1; y = 1; otherwise; y = 2; end; y = y + 1; end",
    );
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Switch {
        cases, otherwise, ..
    } = &body.blocks[0].terminator.kind
    else {
        panic!("expected switch terminator");
    };
    assert_eq!(cases.len(), 1);
    assert!(body.blocks.iter().any(|block| block.id == *otherwise));
    assert_eq!(body.blocks[1].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_switch() {
    let mir =
        lower_mir("function y = choose(x); switch x; case 1; y = 1; otherwise; end; z = y; end");
    let body = mir.bodies.values().next().unwrap();

    let diagnostics = diagnose_uninitialized_reads(body);

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn while_loop_lowers_to_branch_body_backedge_and_exit() {
    let mir = lower_mir("function y = spin(c, x); while c; y = x; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 3);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 2);
}

#[test]
fn while_loop_exit_flows_to_following_statements() {
    let mir = lower_mir("function y = after_loop(c); while c; y = 1; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Branch { else_block, .. } = body.blocks[0].terminator.kind else {
        panic!("expected while branch");
    };
    assert_eq!(else_block, body.blocks[2].id);
    assert_eq!(body.blocks[2].statements.len(), 1);
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn for_loop_lowers_to_iteration_terminator_body_backedge_and_exit() {
    let mir = lower_mir("function y = sum_to(n); y = 0; for i = 1:n; y = y + i; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 3);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::For { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn try_catch_lowers_to_try_catch_blocks_and_merge() {
    let mir = lower_mir("function y = guarded(x); try; y = x; catch err; y = 0; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::TryCatch { .. }
    ));
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert_eq!(body.source_map.statements.len(), 3);
}

#[test]
fn try_catch_flows_to_following_statements() {
    let mir =
        lower_mir("function y = guarded(x); try; y = x; catch err; y = 0; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert_eq!(body.blocks[3].statements.len(), 1);
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn break_in_loop_lowers_to_exit_edge() {
    let mir = lower_mir("function y = first(c); while c; break; end; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::Branch { else_block, .. } = body.blocks[0].terminator.kind else {
        panic!("expected while branch");
    };
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == else_block
    ));
}

#[test]
fn continue_in_loop_lowers_to_loop_condition_edge() {
    let mir = lower_mir("function y = again(c); while c; continue; end; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[0].id
    ));
}

#[test]
fn return_in_nested_block_lowers_to_return_terminator() {
    let mir = lower_mir("function y = done(c); if c; return; else; y = 1; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(_)
    ));
}
