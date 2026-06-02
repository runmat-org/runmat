use runmat_hir::{
    lower, CallSyntax, CallableFallbackPolicy, CallableIdentity, EnvironmentEffect,
    LoweringContext, OperatorKind, RequestedOutputCount,
};
use runmat_mir::{
    analysis::{analyze_assembly, AnalysisStore, MirLocalKey},
    lowering::lower_assembly,
    AsyncBehaviorFact, MirAggregateKind, MirBody, MirCallArg, MirCallee, MirConstant, MirIndexPlan,
    MirLocalKind, MirOperand, MirOutputTarget, MirPlace, MirRvalue, MirStmt, MirStmtKind,
    MirTerminatorKind,
};

fn lower_mir(src: &str) -> runmat_mir::MirAssembly {
    let ast = runmat_parser::parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap();
    lower_assembly(&hir.assembly).unwrap()
}

fn analyze_single_body(src: &str) -> (MirBody, AnalysisStore) {
    let mir = lower_mir(src);
    let body = mir.bodies.values().next().unwrap().clone();
    let store = analyze_assembly(&mir);
    (body, store)
}

fn first_local_of_kind(body: &MirBody, kind: MirLocalKind) -> runmat_mir::MirLocalId {
    body.locals
        .iter()
        .find(|local| local.kind == kind)
        .unwrap()
        .id
}

fn first_call(body: &MirBody) -> &runmat_mir::MirCall {
    body.blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign {
                value: MirRvalue::Call(call),
                ..
            }
            | MirStmtKind::MultiAssign {
                value: MirRvalue::Call(call),
                ..
            }
            | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
            _ => None,
        })
        .expect("expected at least one lowered call")
}

fn first_indexing(body: &MirBody) -> &runmat_mir::MirIndexing {
    body.blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign {
                value: MirRvalue::Index { indexing, .. },
                ..
            }
            | MirStmtKind::Expr(MirRvalue::Index { indexing, .. }) => Some(indexing),
            _ => None,
        })
        .expect("expected at least one lowered index expression")
}

#[test]
fn call_arg_spans_preserve_plain_call_argument_text() {
    let source = "alpha = 1; y = f(alpha, alpha + 1);";
    let mir = lower_mir(source);
    let body = mir.bodies.values().next().unwrap();
    let call = first_call(body);

    assert_eq!(call.arg_spans.len(), 2);
    assert_eq!(
        &source[call.arg_spans[0].start..call.arg_spans[0].end],
        "alpha"
    );
    assert_eq!(
        &source[call.arg_spans[1].start..call.arg_spans[1].end],
        "alpha + 1"
    );
}

#[test]
fn call_arg_spans_exclude_feval_handle_argument() {
    let source = "alpha = 1; y = feval(@f, alpha, alpha + 1);";
    let mir = lower_mir(source);
    let body = mir.bodies.values().next().unwrap();
    let call = first_call(body);

    assert_eq!(call.arg_spans.len(), 2);
    assert_eq!(
        &source[call.arg_spans[0].start..call.arg_spans[0].end],
        "alpha"
    );
    assert_eq!(
        &source[call.arg_spans[1].start..call.arg_spans[1].end],
        "alpha + 1"
    );
}

fn patch_entrypoint_call_requested_outputs(
    source: &str,
    requested_outputs: RequestedOutputCount,
) -> runmat_hir::LoweringResult {
    let ast = runmat_parser::parse(source).expect("parse");
    let mut hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
    let entry_target = hir.assembly.entrypoints[0].target;
    let function = hir
        .assembly
        .functions
        .iter_mut()
        .find(|function| function.id == entry_target)
        .expect("entrypoint target function");

    let mut patched = false;
    for stmt in &mut function.body.statements {
        match &mut stmt.kind {
            runmat_hir::HirStmtKind::Assign(_, expr, _)
            | runmat_hir::HirStmtKind::ExprStmt(expr, _)
            | runmat_hir::HirStmtKind::MultiAssign(_, expr, _) => {
                if let runmat_hir::HirExprKind::Call(call) = &mut expr.kind {
                    call.requested_outputs = requested_outputs.clone();
                    patched = true;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(patched, "expected entrypoint to contain a call expression");
    hir
}

fn patch_entrypoint_multi_assign_requested_outputs(
    source: &str,
    target_requested_outputs: RequestedOutputCount,
    call_requested_outputs: RequestedOutputCount,
) -> runmat_hir::LoweringResult {
    let ast = runmat_parser::parse(source).expect("parse");
    let mut hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
    let entry_target = hir.assembly.entrypoints[0].target;
    let function = hir
        .assembly
        .functions
        .iter_mut()
        .find(|function| function.id == entry_target)
        .expect("entrypoint target function");
    let mut patched = false;
    for stmt in &mut function.body.statements {
        let runmat_hir::HirStmtKind::MultiAssign(targets, expr, _) = &mut stmt.kind else {
            continue;
        };
        targets.requested_outputs = target_requested_outputs.clone();
        if let runmat_hir::HirExprKind::Call(call) = &mut expr.kind {
            call.requested_outputs = call_requested_outputs.clone();
            patched = true;
            break;
        }
    }
    assert!(
        patched,
        "expected entrypoint to contain a multi-assign call"
    );
    hir
}

#[test]
fn simple_function_lowers_to_single_block_with_binding_locals() {
    let mir = lower_mir("function y = f(x); y = x + 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 1);
    assert_eq!(body.locals.len(), 4);
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
fn method_syntax_lowers_with_object_dispatch_fallback_policy() {
    let mir = lower_mir("obj = 1; obj.method(1);");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert!(matches!(
        call.syntax,
        CallSyntax::Method | CallSyntax::DottedInvoke
    ));
    assert_eq!(call.fallback_policy, CallableFallbackPolicy::ObjectDispatch);
}

#[test]
fn unresolved_plain_call_lowers_with_runtime_name_resolution_fallback_policy() {
    let mir = lower_mir("y = unresolved_fn(1);");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert!(matches!(
        call.callee,
        MirCallee::Static(CallableIdentity::DynamicName(_))
    ));
    assert_eq!(
        call.fallback_policy,
        CallableFallbackPolicy::RuntimeNameResolution
    );
}

#[test]
fn unresolved_qualified_call_lowers_with_external_boundary_fallback_policy() {
    let mir = lower_mir("y = pkg.remote_inc(1);");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert!(matches!(
        call.callee,
        MirCallee::Static(CallableIdentity::ExternalName(_))
    ));
    assert_eq!(
        call.fallback_policy,
        CallableFallbackPolicy::ExternalBoundary
    );
    assert!(matches!(call.syntax, CallSyntax::Plain));
}

#[test]
fn unresolved_nested_qualified_call_lowers_with_external_boundary_fallback_policy() {
    let mir = lower_mir("y = pkg.sub.remote(1);");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert!(matches!(
        call.callee,
        MirCallee::Static(CallableIdentity::ExternalName(_))
    ));
    assert_eq!(
        call.fallback_policy,
        CallableFallbackPolicy::ExternalBoundary
    );
    assert!(matches!(call.syntax, CallSyntax::Plain));
}

#[test]
fn recursive_function_call_lowers_to_bound_function_identity() {
    let mir = lower_mir("function y = fact(n); y = fact(n - 1); end");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);

    assert!(matches!(
        call.callee,
        MirCallee::Static(CallableIdentity::BoundFunction(id)) if id == body.function
    ));
    assert_eq!(call.syntax, CallSyntax::Plain);
    assert_eq!(call.fallback_policy, CallableFallbackPolicy::None);
}

#[test]
fn lower_assembly_accepts_zero_requested_outputs() {
    let hir = patch_entrypoint_call_requested_outputs("y = sqrt(9);", RequestedOutputCount::Zero);
    let mir = lower_assembly(&hir.assembly).expect("lower should succeed");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert_eq!(call.requested_outputs, RequestedOutputCount::Zero);
}

#[test]
fn lower_assembly_accepts_explicit_multi_requested_outputs() {
    let hir =
        patch_entrypoint_call_requested_outputs("y = sqrt(9);", RequestedOutputCount::Exactly(3));
    let mir = lower_assembly(&hir.assembly).expect("lower should succeed");
    let body = mir.bodies.values().next().expect("body");
    let call = first_call(body);
    assert_eq!(call.requested_outputs, RequestedOutputCount::Exactly(3));
}

#[test]
fn lower_assembly_rejects_multi_assign_target_count_mismatch() {
    let hir = patch_entrypoint_multi_assign_requested_outputs(
        "x = 1; [~, idx] = max(x);",
        RequestedOutputCount::One,
        RequestedOutputCount::One,
    );
    let err = lower_assembly(&hir.assembly).expect_err("lower should fail");
    assert!(err.message.contains("output target count mismatch"));
}

#[test]
fn lower_assembly_rejects_multi_assign_call_target_policy_mismatch() {
    let hir = patch_entrypoint_multi_assign_requested_outputs(
        "x = 1; [~, idx] = max(x);",
        RequestedOutputCount::Exactly(2),
        RequestedOutputCount::One,
    );
    let err = lower_assembly(&hir.assembly).expect_err("lower should fail");
    assert!(err.message.contains("must match multi-assign targets"));
}

#[test]
fn dataflow_marks_parameters_and_assigned_outputs_definitely_assigned() {
    let mir = lower_mir("function y = f(x); y = x; end");
    let body = mir.bodies.values().next().unwrap();
    let store = analyze_assembly(&mir);
    let param = first_local_of_kind(body, MirLocalKind::Parameter);
    let output = first_local_of_kind(body, MirLocalKind::Output);

    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: param,
    }));
    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: output,
    }));
    assert!(!store
        .diagnostics
        .iter()
        .any(|diagnostic| { matches!(diagnostic.code.as_str(), "RM-MIR0001" | "RM-MIR0002") }));
}

#[test]
fn dataflow_joins_branch_assignment_as_maybe_assigned() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; end");
    let body = mir.bodies.values().next().unwrap();
    let store = analyze_assembly(&mir);
    let output = first_local_of_kind(body, MirLocalKind::Output);

    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: output,
    }));
}

#[test]
fn analyze_assembly_populates_output_local_facts_by_semantic_id() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; end");
    let store = analyze_assembly(&mir);
    let body = mir.bodies.values().next().unwrap();
    let output_local = body
        .locals
        .iter()
        .find(|local| matches!(local.kind, MirLocalKind::Output))
        .unwrap()
        .id;

    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: output_local,
    }));
}

#[test]
fn analyze_body_records_simple_numeric_local_and_binding_facts() {
    let (body, store) = analyze_single_body("function y = f(); y = 1; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
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
    let function_fact = store
        .mir_locals
        .values()
        .find(|fact| matches!(fact.ty, runmat_hir::TypeFact::Function(_)))
        .unwrap();

    assert!(matches!(
        function_fact.value_flow,
        runmat_hir::ValueFlowFact::Single(runmat_hir::TypeFact::Function(_))
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
fn analyze_body_records_binary_op_as_unknown_scalar_fact() {
    let (body, store) = analyze_single_body("function y = f(x); y = x + 1; end");
    let output = first_local_of_kind(&body, MirLocalKind::Output);
    let fact = &store.mir_locals[&MirLocalKey {
        function: body.function,
        local: output,
    }];

    assert_eq!(fact.ty, runmat_hir::TypeFact::Unknown);
    assert_eq!(fact.shape, runmat_hir::ShapeFact::Unknown);
    assert_eq!(fact.value_flow, runmat_hir::ValueFlowFact::UnknownList);
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
                    MirRvalue::Use(MirOperand::FunctionHandle(CallableIdentity::AnonymousFunction(_))),
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

    let handle_local = body
        .locals
        .iter()
        .find_map(|local| {
            Some(local.id).filter(|_| {
                local.kind == MirLocalKind::Temporary
                    && matches!(
                        store.mir_locals[&MirLocalKey {
                            function: body.function,
                            local: local.id,
                        }]
                            .ty,
                        runmat_hir::TypeFact::Function(_)
                    )
            })
        })
        .unwrap();
    assert!(matches!(
        store.mir_locals[&MirLocalKey {
            function: body.function,
            local: handle_local,
        }]
            .ty,
        runmat_hir::TypeFact::Function(_)
    ));
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003"
            && diagnostic.message.contains("unknown spawn safety")
            && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
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
fn analyze_body_populates_local_facts_for_parameters() {
    let (body, store) = analyze_single_body("function y = f(x); y = x + 1; end");
    let parameter = first_local_of_kind(&body, MirLocalKind::Parameter);

    assert_eq!(
        store.mir_locals[&MirLocalKey {
            function: body.function,
            local: parameter,
        }]
            .ty,
        runmat_hir::TypeFact::Unknown
    );
}

#[test]
fn dataflow_widens_loop_assignment_as_maybe_assigned() {
    let mir = lower_mir("function y = f(c); while c; y = 1; end; end");
    let body = mir.bodies.values().next().unwrap();
    let store = analyze_assembly(&mir);
    let output = first_local_of_kind(body, MirLocalKind::Output);

    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: output,
    }));
}

#[test]
fn diagnostics_report_unassigned_local_read() {
    let mir = lower_mir("function y = f(); z = y; y = 1; end");
    let diagnostics = analyze_assembly(&mir).diagnostics;

    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0001"
            && diagnostic.category.as_deref() == Some("definite-assignment")
            && diagnostic.primary.span.start < diagnostic.primary.span.end
    }));
}

#[test]
fn diagnostics_report_maybe_assigned_local_read_after_branch() {
    let mir = lower_mir("function y = f(c); if c; y = 1; end; x = y; end");
    let diagnostics = analyze_assembly(&mir).diagnostics;

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0002"));
}

#[test]
fn summary_records_function_outputs_and_store_entry() {
    let mir = lower_mir("function y = f(x); y = x; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.abi.fixed_inputs.len(), 1);
    assert_eq!(body.abi.fixed_outputs.len(), 1);
    assert!(body.abi.implicit_nargin.is_some());
    assert!(body.abi.implicit_nargout.is_some());
}

#[test]
fn summary_preserves_variadic_function_abi() {
    let mir = lower_mir("function varargout = f(x, varargin); varargout = varargin; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.abi.fixed_inputs.len(), 2);
    assert_eq!(body.abi.varargin, Some(body.abi.fixed_inputs[1]));
    assert_eq!(body.abi.fixed_outputs.len(), 1);
    assert_eq!(body.abi.varargout, Some(body.abi.fixed_outputs[0]));
}

#[test]
fn summary_records_requested_output_call_facts() {
    let mir = lower_mir("function y = f(varargin); [a, b] = g(varargin{:}); y = a; end");
    let body = mir.bodies.values().next().unwrap();
    let call = first_call(body);

    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::Exactly(2)
    ));
    assert_eq!(call.args.len(), 1);
    assert_eq!(
        call.args
            .iter()
            .filter(|arg| matches!(arg, runmat_mir::MirCallArg::Expansion { .. }))
            .count(),
        1
    );
    assert!(matches!(call.async_behavior, AsyncBehaviorFact::MaySuspend));
}

#[test]
fn summary_records_unresolved_calls_in_call_facts() {
    let mir = lower_mir("function y = f(x); y = unresolved_target(x); end");
    let body = mir.bodies.values().next().unwrap();
    let call = first_call(body);
    assert!(matches!(
        call.callee,
        MirCallee::Dynamic(_)
            | MirCallee::Static(
                CallableIdentity::DynamicName(_) | CallableIdentity::ExternalName(_)
            )
    ));
}

#[test]
fn summary_records_spawn_operands_without_call_facts() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");
    let body = mir.bodies.values().next().unwrap();
    assert!(body
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .any(|stmt| matches!(
            stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::Spawn(_),
                ..
            }
        )));
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
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003"
            && diagnostic.message.contains("mutates a lexical capture")
            && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn analyze_assembly_rejects_spawned_future_with_lexical_capture_read() {
    let mir = lower_mir(
        "function y = outer(); acc = 1; async function z = read_acc(); z = acc; end; fut = read_acc(); task = spawn(fut); y = acc; end",
    );

    let store = analyze_assembly(&mir);
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003"
            && diagnostic.message.contains("mutates a lexical capture")
            && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn spawn_safety_uses_future_target_visible_at_spawn_site() {
    let mir = lower_mir(
        "function y = outer(); cap = 1; async function z = safe(); z = 1; end; async function z = unsafe(); z = cap; end; fut = safe(); task = spawn(fut); fut = unsafe(); y = cap; end",
    );

    let store = analyze_assembly(&mir);
    assert!(mir
        .bodies
        .values()
        .flat_map(|body| body.blocks.iter())
        .flat_map(|block| block.statements.iter())
        .any(|stmt| matches!(
            stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::Spawn(_),
                ..
            }
        )));
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
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003"
            && diagnostic.message.contains("mutates a lexical capture")
            && diagnostic.category.as_deref() == Some("spawn-safety")
    }));
}

#[test]
fn analyze_assembly_rejects_spawned_future_with_unknown_target() {
    let mir = lower_mir("function y = f(g); y = spawn(g); end");

    let store = analyze_assembly(&mir);
    assert!(store.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-MIR0003"
            && diagnostic.message.contains("unknown spawn safety")
            && diagnostic.category.as_deref() == Some("spawn-safety")
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

    let body = mir.bodies.values().next().unwrap();
    let spawn_count = body
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .filter(|stmt| {
            matches!(
                stmt.kind,
                MirStmtKind::Assign {
                    value: MirRvalue::Spawn(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(spawn_count, 1);
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
        callee: MirCallee::Static(CallableIdentity::ExternalName(runmat_hir::QualifiedName(
            vec![runmat_hir::SymbolName("sink".into())],
        ))),
        args: vec![MirCallArg::Expansion {
            base: MirOperand::Local(local),
            indices: Vec::new(),
            expand_all: true,
        }],
        arg_spans: vec![runmat_hir::Span::default()],
        syntax: runmat_hir::CallSyntax::Plain,
        requested_outputs: runmat_hir::RequestedOutputCount::Zero,
        fallback_policy: runmat_hir::CallableFallbackPolicy::ExternalBoundary,
        async_behavior: runmat_mir::AsyncBehaviorFact::MaySuspend,
        effects: runmat_builtins::BuiltinEffects::unknown(),
        workspace_effect: None,
        environment_effect: None,
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
    assert!(!roundtrip.mir_locals.is_empty());
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
    let mir = lower_mir("async function y = f(g); y = await(g); end");
    let body = mir.bodies.values().next().unwrap();
    let store = analyze_assembly(&mir);
    let output = first_local_of_kind(body, MirLocalKind::Output);

    assert!(store.mir_locals.contains_key(&MirLocalKey {
        function: body.function,
        local: output,
    }));
}

#[test]
fn global_statement_lowers_to_workspace_effect() {
    let mir = lower_mir("function y = f(); global g; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

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
    assert!(body.locals[global_local.0].binding.is_some());
    assert!(body.blocks[0].statements.iter().any(|stmt| matches!(
        stmt.kind,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_hir::WorkspaceEffect::MutatesGlobal,
            ..
        }
    )));
}

#[test]
fn analyze_assembly_preserves_workspace_effect_markers_and_call_facts() {
    let mir = lower_mir("function y = f(); global g; y = unresolved_target(g); end");

    let body = mir.bodies.values().next().unwrap();

    assert!(body.blocks[0].statements.iter().any(|stmt| matches!(
        stmt.kind,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_hir::WorkspaceEffect::MutatesGlobal,
            ..
        }
    )));
    let call = first_call(body);
    assert!(matches!(
        call.callee,
        MirCallee::Dynamic(_)
            | MirCallee::Static(
                CallableIdentity::DynamicName(_) | CallableIdentity::ExternalName(_)
            )
    ));
}

#[test]
fn persistent_statement_lowers_to_summary_workspace_effect() {
    let mir = lower_mir("function y = f(); persistent p; y = 1; end");
    let body = mir.bodies.values().next().unwrap();

    assert!(body.blocks[0].statements.iter().any(|stmt| matches!(
        stmt.kind,
        MirStmtKind::WorkspaceEffect {
            effect: runmat_hir::WorkspaceEffect::MutatesPersistent,
            ..
        }
    )));
    let persistent_local = match &body.blocks[0].statements[0].kind {
        MirStmtKind::WorkspaceEffect { bindings, .. } => bindings[0],
        _ => unreachable!(),
    };
    assert!(body.locals[persistent_local.0].binding.is_some());
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
    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::EnvironmentEffect(EnvironmentEffect::FunctionCacheInvalidation)
    ));
    assert!(!body.blocks[0]
        .statements
        .iter()
        .any(|stmt| matches!(stmt.kind, MirStmtKind::WorkspaceEffect { .. })));
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

    assert!(matches!(
        call.callee,
        MirCallee::Static(CallableIdentity::BoundFunction(_))
    ));
    assert!(matches!(
        call.requested_outputs,
        runmat_hir::RequestedOutputCount::One
    ));
}

#[test]
fn feval_lowers_to_dynamic_mir_callee() {
    let mir = lower_mir("f = @sin; y = feval(f, 0);");
    let body = mir.bodies.values().next().unwrap();
    let call = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign {
                value: MirRvalue::Call(call),
                ..
            } => Some(call),
            _ => None,
        })
        .expect("expected feval call");

    assert!(matches!(
        call.callee,
        MirCallee::Dynamic(MirOperand::Local(_))
    ));
    assert_eq!(call.args.len(), 1);
}

#[test]
fn qualified_static_method_function_handle_lowers_to_imported_identity_operand() {
    let mir = lower_mir("h = @Point.origin; y = feval(h);");
    let body = mir.bodies.values().next().expect("body");
    assert!(body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|stmt| matches!(
            &stmt.kind,
            MirStmtKind::Assign {
                value:
                    MirRvalue::Use(MirOperand::FunctionHandle(
                        CallableIdentity::Imported(path)
                    )),
                ..
            } if path.module.display_name().as_deref() == Some("Point.origin")
                && matches!(path.item.as_slice(), [runmat_hir::DefPathSegment::Function(_)])
        )));
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

    let Some(call) = body.blocks[0]
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
            _ => None,
        })
    else {
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
fn struct_aggregate_literal_lowers_to_mir_struct_literal() {
    let mir = lower_mir("s = struct{a = 1, a = 2};");
    let body = mir.bodies.values().next().expect("body");
    let value = body
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign { value, .. } => Some(value),
            _ => None,
        })
        .expect("assignment");
    let MirRvalue::StructLiteral { fields } = value else {
        panic!("expected MIR struct literal");
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0 .0, "a");
    assert_eq!(fields[1].0 .0, "a");
}

#[test]
fn object_aggregate_literal_lowers_to_mir_object_literal() {
    let mir = lower_mir("p = ?Point{x = 1, y = 2};");
    let body = mir.bodies.values().next().expect("body");
    let value = body
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .find_map(|stmt| match &stmt.kind {
            MirStmtKind::Assign { value, .. } => Some(value),
            _ => None,
        })
        .expect("assignment");
    let MirRvalue::ObjectLiteral { class_name, fields } = value else {
        panic!("expected MIR object literal");
    };
    assert_eq!(
        class_name
            .0
            .iter()
            .map(|segment| segment.0.as_str())
            .collect::<Vec<_>>(),
        vec!["Point"]
    );
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0 .0, "x");
    assert_eq!(fields[1].0 .0, "y");
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
        MirCallArg::Expansion {
            base: MirOperand::Local(_),
            ..
        }
    ));
}

#[test]
fn function_argument_expansion_lowers_end_offset_selectors() {
    let mir = lower_mir("function y = f(C); y = g(C{end-1}, C{end+1}); end");
    let body = mir.bodies.values().next().unwrap();
    let call = first_call(body);

    assert_eq!(call.args.len(), 2);
    assert!(matches!(
        call.args[0],
        MirCallArg::Expansion {
            base: MirOperand::Local(_),
            ref indices,
            expand_all: false,
        } if indices.len() == 1
    ));
    assert!(matches!(
        call.args[1],
        MirCallArg::Expansion {
            base: MirOperand::Local(_),
            ref indices,
            expand_all: false,
        } if indices.len() == 1
    ));

    let statements: Vec<&MirStmt> = body
        .blocks
        .iter()
        .flat_map(|block| block.statements.iter())
        .collect();
    let has_end_seed = statements.iter().any(|stmt| {
        matches!(
            stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::End,
                ..
            }
        )
    });
    let has_end_minus = statements.iter().any(|stmt| {
        matches!(
            stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::Binary(
                    MirOperand::Local(_),
                    OperatorKind::Subtract,
                    MirOperand::Constant(MirConstant::Number(ref value))
                ),
                ..
            } if value == "1"
        )
    });
    let has_end_plus = statements.iter().any(|stmt| {
        matches!(
            stmt.kind,
            MirStmtKind::Assign {
                value: MirRvalue::Binary(
                    MirOperand::Local(_),
                    OperatorKind::Add,
                    MirOperand::Constant(MirConstant::Number(ref value))
                ),
                ..
            } if value == "1"
        )
    });

    assert!(has_end_seed);
    assert!(has_end_minus);
    assert!(has_end_plus);
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
    assert!(matches!(
        targets.requested_outputs,
        runmat_hir::RequestedOutputCount::Exactly(2)
    ));
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
fn paren_index_plan_is_scalar_for_literal_scalar_indices() {
    let mir = lower_mir("function y = f(a); y = a(2, 1); end");
    let body = mir.bodies.values().next().expect("body");
    assert_eq!(first_indexing(body).plan, MirIndexPlan::Scalar);
}

#[test]
fn paren_index_plan_is_slice_for_general_index_operands() {
    let mir = lower_mir("function y = f(a, idx); y = a(idx); end");
    let body = mir.bodies.values().next().expect("body");
    assert_eq!(first_indexing(body).plan, MirIndexPlan::Slice);
}

#[test]
fn paren_index_plan_is_slice_expr_for_range_or_end_selectors() {
    let mir = lower_mir("function y = f(a); y = a(1:end-1); end");
    let body = mir.bodies.values().next().expect("body");
    assert_eq!(first_indexing(body).plan, MirIndexPlan::SliceExpr);
}

#[test]
fn brace_index_plan_is_cell() {
    let mir = lower_mir("function y = f(c); y = c{1}; end");
    let body = mir.bodies.values().next().expect("body");
    assert_eq!(first_indexing(body).plan, MirIndexPlan::Cell);
    assert!(!first_indexing(body).cell_expand_all);
}

#[test]
fn brace_all_colon_sets_cell_expand_all_intent() {
    let mir = lower_mir("function y = f(c); y = c{:}; end");
    let body = mir.bodies.values().next().expect("body");
    let indexing = first_indexing(body);
    assert_eq!(indexing.plan, MirIndexPlan::Cell);
    assert!(indexing.cell_expand_all);
}

#[test]
fn diagnostics_report_unassigned_index_operand_read() {
    let mir = lower_mir("function y = read_index(a); y = a(y); end");
    let diagnostics = analyze_assembly(&mir).diagnostics;

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
fn indexed_member_assignment_lowers_to_index_place_over_member_base() {
    let mir = lower_mir("function s = write_member_index(s); s.a(2) = 9; end");
    let body = mir.bodies.values().next().expect("body");

    assert!(matches!(
        body.blocks[0].statements[0].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::IndexedAssign
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::CreateArrayByIndex
    ));
    assert!(matches!(
        &body.blocks[0].statements[1].kind,
        MirStmtKind::Assign {
            place: MirPlace::Index(base, indexing),
            ..
        } if matches!(&**base, MirPlace::Member(_, _))
            && indexing.plan == MirIndexPlan::Scalar
    ));
}

#[test]
fn indexed_member_delete_lowers_to_index_place_over_member_base() {
    let mir = lower_mir("function s = delete_member_index(s); s.a = {1, 2, 3}; s.a(2) = []; end");
    let body = mir.bodies.values().next().expect("body");

    assert!(matches!(
        body.blocks[0].statements[2].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::Delete
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::ExistingOnly
    ));
    assert!(matches!(
        &body.blocks[0].statements[3].kind,
        MirStmtKind::Assign {
            place: MirPlace::Index(base, indexing),
            ..
        } if matches!(&**base, MirPlace::Member(_, _))
            && indexing.plan == MirIndexPlan::Scalar
    ));
}

#[test]
fn indexed_dynamic_member_delete_lowers_to_index_place_over_dynamic_member_base() {
    let mir = lower_mir(
        "function s = delete_dynamic_member_index(s, f); s.(f) = {1, 2, 3}; s.(f)(2) = []; end",
    );
    let body = mir.bodies.values().next().expect("body");

    assert!(matches!(
        body.blocks[0].statements[2].kind,
        MirStmtKind::PlaceMutation(ref mutation)
            if mutation.kind == runmat_hir::PlaceMutationKind::Delete
                && mutation.creation_policy == runmat_hir::AssignmentCreationPolicy::ExistingOnly
    ));
    assert!(matches!(
        &body.blocks[0].statements[3].kind,
        MirStmtKind::Assign {
            place: MirPlace::Index(base, indexing),
            ..
        } if matches!(&**base, MirPlace::DynamicMember(_, _))
            && indexing.plan == MirIndexPlan::Scalar
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
    let diagnostics = analyze_assembly(&mir).diagnostics;

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
    let diagnostics = analyze_assembly(&mir).diagnostics;

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
fn while_loop_with_prefix_splits_clean_header() {
    let mir = lower_mir("function y = spin(c); y = 0; while c; y = y + 1; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[1].id
    ));
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::Branch { .. }
    ));
    assert!(body.blocks[1].statements.is_empty());
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[1].id
    ));
}

#[test]
fn for_loop_lowers_to_iteration_terminator_body_backedge_and_exit() {
    let mir = lower_mir("function y = sum_to(n); y = 0; for i = 1:n; y = y + i; end; end");
    let body = mir.bodies.values().next().unwrap();

    assert_eq!(body.blocks.len(), 4);
    assert!(matches!(
        body.blocks[0].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[1].id
    ));
    assert_eq!(body.blocks[0].statements.len(), 1);
    assert!(matches!(
        body.blocks[1].terminator.kind,
        MirTerminatorKind::For { .. }
    ));
    assert!(body.blocks[1].statements.is_empty());
    assert!(matches!(
        body.blocks[2].terminator.kind,
        MirTerminatorKind::Goto(target) if target == body.blocks[1].id
    ));
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
}

#[test]
fn for_loop_exit_flows_to_following_statements() {
    let mir =
        lower_mir("function y = after_loop(n); y = 0; for i = 1:n; y = y + i; end; y = y + 1; end");
    let body = mir.bodies.values().next().unwrap();

    let MirTerminatorKind::For { exit_block, .. } = body.blocks[1].terminator.kind else {
        panic!("expected for terminator");
    };
    assert_eq!(exit_block, body.blocks[3].id);
    assert_eq!(body.blocks[3].statements.len(), 1);
    assert!(matches!(
        body.blocks[3].terminator.kind,
        MirTerminatorKind::Return(_)
    ));
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
