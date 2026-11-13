use runmat_hir::{
    lower, CallKind, DefPathSegment, FunctionHandleTarget, FunctionId, FunctionKind, HirAssembly,
    HirCall, HirCallableRef, HirExprKind, HirPlace, HirStmtKind, HirSubscriptStep, IndexComponent,
    IndexKind, IndexResultContext, IndexingSemantics, LoweringContext,
};
use runmat_parser::parse;
use std::collections::{HashMap, HashSet};

fn lower_assembly(src: &str) -> HirAssembly {
    let ast = parse(src).unwrap();
    lower(&ast, &LoweringContext::empty()).unwrap().assembly
}

fn lower_assembly_with_bound_functions(
    src: &str,
    bound_functions: &HashMap<String, FunctionId>,
) -> HirAssembly {
    let ast = parse(src).unwrap();
    lower(
        &ast,
        &LoweringContext::empty().with_bound_functions(bound_functions),
    )
    .unwrap()
    .assembly
}

fn entry_body(assembly: &HirAssembly) -> &[runmat_hir::HirStmt] {
    let entry = assembly.entrypoints[0].target;
    &assembly.functions[entry.0].body.statements
}

fn discarded_parentheses(
    stmt: &runmat_hir::HirStmt,
    binding: runmat_hir::BindingId,
) -> &IndexingSemantics {
    let HirStmtKind::ExprStmt(expr, true) = &stmt.kind else {
        panic!("expected suppressed expression statement: {stmt:?}");
    };
    let HirExprKind::SubscriptChain(chain) = &expr.kind else {
        panic!("expected receiver-selected parentheses: {expr:?}");
    };
    assert!(matches!(chain.root.kind, HirExprKind::Binding(id) if id == binding));
    assert_eq!(chain.sequence_use, runmat_types::SequenceUse::Discard);
    assert_eq!(
        chain.context,
        runmat_types::ObjectIndexingContext::Statement
    );
    let [HirSubscriptStep::Index(indexing)] = chain.steps.as_slice() else {
        panic!("expected one indexing step: {chain:?}");
    };
    assert_eq!(indexing.kind, IndexKind::Paren);
    assert_eq!(indexing.result_context, IndexResultContext::ReadSingle);
    indexing
}

#[test]
fn expr_arithmetic_and_comparisons_lower_to_semantic_shapes() {
    let prog = lower_assembly("1 + 2 * 3; 4 .^ 2; 5 < 6; 7 && 8");
    let body = entry_body(&prog);
    assert_eq!(body.len(), 4);
    for stmt in body {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => {
                assert!(matches!(expr.kind, HirExprKind::Binary(_, _, _)))
            }
            other => panic!("expected expr stmt, got {other:?}"),
        }
    }
}

#[test]
fn matrices_cells_and_indexing_lower_to_semantic_shapes() {
    let prog = lower_assembly("A = [1,2;3,4]; C = {A, 2}; A(1,2); C{1}");
    let body = entry_body(&prog);
    assert_eq!(body.len(), 4);
    assert!(
        matches!(&body[0].kind, HirStmtKind::Assign(_, expr, _) if matches!(expr.kind, HirExprKind::Tensor(_)))
    );
    assert!(
        matches!(&body[1].kind, HirStmtKind::Assign(_, expr, _) if matches!(expr.kind, HirExprKind::Cell(_)))
    );
    let HirStmtKind::Assign(HirPlace::Binding(a), _, _) = &body[0].kind else {
        panic!("expected A assignment");
    };
    let indexing = discarded_parentheses(&body[2], *a);
    assert!(matches!(
        indexing.components.as_slice(),
        [IndexComponent::Expr(row), IndexComponent::Expr(column)]
            if matches!(&row.kind, HirExprKind::Number(value) if value == "1")
                && matches!(&column.kind, HirExprKind::Number(value) if value == "2")
    ));
    assert!(
        matches!(&body[3].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Index(_, _)))
    );
}

#[test]
fn end_colon_and_range_lower_to_semantic_shapes() {
    let prog = lower_assembly("A = [1,2,3,4]; A(2:end); 1:2:5; :");
    let body = entry_body(&prog);
    assert_eq!(body.len(), 4);
    let HirStmtKind::Assign(HirPlace::Binding(a), _, _) = &body[0].kind else {
        panic!("expected A assignment");
    };
    let indexing = discarded_parentheses(&body[1], *a);
    assert!(matches!(
        indexing.components.as_slice(),
        [IndexComponent::Expr(range)]
            if matches!(&range.kind, HirExprKind::Range(start, None, end)
                if matches!(&start.kind, HirExprKind::Number(value) if value == "2")
                    && matches!(end.kind, HirExprKind::End))
    ));
    assert!(
        matches!(&body[2].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Range(_, _, _)))
    );
    assert!(
        matches!(&body[3].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Colon))
    );
}

#[test]
fn control_flow_if_while_for_switch_trycatch_lower_to_semantic_shapes() {
    let src = r#"
x = 1; y = 0; z = 0;
if x
  y=1;
elseif x>0
  y=2;
else
  y=3;
end
while x
  y=4; x = 0;
end
for i=1:3
  y=i;
end
switch y
  case 1
    z=1;
  otherwise
    z=0;
end
try
  a=1;
catch e
  b=2;
end
"#;
    let prog = lower_assembly(src);
    let body = entry_body(&prog);
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::If { .. })));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::While { .. })));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::For { .. })));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::Switch { .. })));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::TryCatch { .. })));
}

#[test]
fn globals_persistents_and_multiassign_lower_to_semantic_shapes() {
    let prog = lower_assembly("global a,b; persistent p; [x,y]=deal(1,2)");
    let body = entry_body(&prog);
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::Global(_))));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::Persistent(_))));
    assert!(body
        .iter()
        .any(|s| matches!(s.kind, HirStmtKind::MultiAssign(_, _, _))));
}

#[test]
fn functions_and_calls_lower_to_semantic_shapes() {
    let src = r#"
function y=f(x)
  if x > 0
    y = 1;
  else
    y = 2;
  end
end
z = f(3)
"#;
    let ast = parse(src).unwrap();
    let result = lower(&ast, &LoweringContext::empty()).unwrap();
    assert!(result
        .assembly
        .functions
        .iter()
        .any(|function| function.name.0 == "f"));
    assert!(result
        .hir_index
        .calls
        .iter()
        .any(|call| matches!(call.kind, CallKind::DirectFunction(_))));
}

#[test]
fn methods_members_handles_and_anon_lower_to_semantic_shapes() {
    let method = lower_assembly("obj = 1; obj.method(1);");
    assert!(entry_body(&method).iter().any(|stmt| {
        matches!(
            &stmt.kind,
            HirStmtKind::ExprStmt(expr, _)
                if matches!(
                    &expr.kind,
                    HirExprKind::SubscriptChain(chain)
                        if matches!(&chain.root.kind, HirExprKind::Binding(_))
                            && matches!(
                                chain.steps.as_slice(),
                                [HirSubscriptStep::DottedInvoke { member, arguments, .. }]
                                    if member.0 == "method"
                                        && arguments.len() == 1
                            )
                )
        )
    }));

    let member = lower_assembly("obj = 1; obj.field;");
    assert!(entry_body(&member).iter().any(|stmt| {
        matches!(&stmt.kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Member { .. }))
    }));

    let handle = lower_assembly("@sin;");
    assert!(
        matches!(&entry_body(&handle)[0].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::FunctionHandle(FunctionHandleTarget::Builtin(_)) | HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(_))))
    );

    let imported_handle = lower_assembly("import MissingCoverageClass.origin; @origin;");
    assert!(entry_body(&imported_handle).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::ExprStmt(expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(name))
                    if name.0 == "origin"
            )
    )));

    let qualified_handle = lower_assembly("@pkg.remote_inc;");
    assert!(entry_body(&qualified_handle).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::ExprStmt(expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(name))
                    if name.0 == "pkg.remote_inc"
            )
    )));

    let qualified_static_method_handle = lower_assembly("@Point.origin;");
    assert!(entry_body(&qualified_static_method_handle)
        .iter()
        .any(|stmt| matches!(
            &stmt.kind,
            HirStmtKind::ExprStmt(expr, _)
                if matches!(
                    &expr.kind,
                    HirExprKind::FunctionHandle(FunctionHandleTarget::Builtin(identity))
                        if identity.0 == "Point.origin"
                )
        )));

    let wildcard_imported_handle = lower_assembly("import MissingCoverageClass.*; @origin;");
    assert!(entry_body(&wildcard_imported_handle)
        .iter()
        .any(|stmt| matches!(
            &stmt.kind,
            HirStmtKind::ExprStmt(expr, _)
                if matches!(
                    &expr.kind,
                    HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(name))
                        if name.0 == "origin"
                )
        )));

    let qualified_unresolved_call = lower_assembly("pkg.remote_inc(1);");
    assert!(entry_body(&qualified_unresolved_call)
        .iter()
        .any(|stmt| matches!(
            &stmt.kind,
            HirStmtKind::ExprStmt(expr, _)
                if matches!(
                    &expr.kind,
                    HirExprKind::Call(HirCall {
                        callee: HirCallableRef::Unresolved(path),
                        ..
                    }) if path.display_name().as_deref() == Some("pkg.remote_inc")
                        && path.0.len() == 2
                )
        )));

    let nested_qualified_unresolved_call = lower_assembly("pkg.sub.remote(1);");
    assert!(entry_body(&nested_qualified_unresolved_call)
        .iter()
        .any(|stmt| matches!(
            &stmt.kind,
            HirStmtKind::ExprStmt(expr, _)
                if matches!(
                    &expr.kind,
                    HirExprKind::Call(HirCall {
                        callee: HirCallableRef::Unresolved(path),
                        ..
                    }) if path.display_name().as_deref() == Some("pkg.sub.remote")
                        && path.0.len() == 3
                )
        )));

    let anon = lower_assembly("@(x) x+1");
    assert!(anon
        .functions
        .iter()
        .any(|function| matches!(function.kind, FunctionKind::Anonymous)));
}

#[test]
fn imported_call_lowers_to_package_function_identity() {
    let src = "import Point.origin; __register_test_classes(); o = origin();";
    let ast = parse(src).unwrap();
    let known_symbols = HashSet::from(["Point.origin".to_string()]);
    let result = lower(
        &ast,
        &LoweringContext::empty().with_known_project_symbols(&known_symbols),
    )
    .unwrap();
    assert!(result.hir_index.calls.iter().any(|call| matches!(
        &call.kind,
        CallKind::PackageFunction(path)
            if path.module.display_name().as_deref() == Some("Point.origin")
                && matches!(path.item.as_slice(), [DefPathSegment::Function(_)])
    )));
}

#[test]
fn an_import_does_not_prove_that_its_target_exists() {
    // Point.origin is a known canonical fixture builtin, so it cannot witness
    // an import whose target has no semantic evidence.
    assert!(!runmat_builtins::builtin_name_is_known(
        "MissingCoverageClass.origin"
    ));
    let src = "import MissingCoverageClass.origin; o = origin(); h = @origin;";
    let ast = parse(src).unwrap();
    let result = lower(&ast, &LoweringContext::empty()).unwrap();
    assert!(result
        .hir_index
        .calls
        .iter()
        .any(|call| matches!(&call.kind, CallKind::Dynamic)
            && matches!(&call.callee, HirCallableRef::Unresolved(name)
                if name.display_name().as_deref() == Some("origin"))));
    assert!(entry_body(&result.assembly).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::Assign(_, expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(name))
                    if name.0 == "origin"
            )
    )));
}

#[test]
fn verified_project_function_handles_use_static_identity() {
    let src = "h = @pkg.remote_inc;";
    let ast = parse(src).unwrap();
    let known_symbols = HashSet::from(["pkg.remote_inc".to_string()]);
    let result = lower(
        &ast,
        &LoweringContext::empty().with_known_project_symbols(&known_symbols),
    )
    .unwrap();
    assert!(entry_body(&result.assembly).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::Assign(_, expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::DefPath(path))
                    if path.module.display_name().as_deref() == Some("pkg.remote_inc")
                        && matches!(path.item.as_slice(), [DefPathSegment::Function(_)])
            )
    )));
}

#[test]
fn wildcard_import_non_builtin_call_lowers_to_dynamic_identity() {
    assert!(!runmat_builtins::builtin_name_is_known(
        "MissingCoverageClass.origin"
    ));
    let src = "import MissingCoverageClass.*; o = origin();";
    let ast = parse(src).unwrap();
    let result = lower(&ast, &LoweringContext::empty()).unwrap();
    assert!(result
        .hir_index
        .calls
        .iter()
        .any(|call| matches!(&call.kind, CallKind::Dynamic)
            && matches!(&call.callee, HirCallableRef::Unresolved(name)
                if name.display_name().as_deref() == Some("origin"))));
}

#[test]
fn function_handle_prefers_semantic_function_identity_from_context() {
    let mut bound_functions = HashMap::new();
    bound_functions.insert("remote_inc".to_string(), FunctionId(4242));
    let assembly = lower_assembly_with_bound_functions("h = @remote_inc;", &bound_functions);
    assert!(entry_body(&assembly).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::Assign(_, expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::ExternalFunction {
                    function: FunctionId(4242),
                    display_name,
                }) if display_name == "remote_inc"
            )
    )));
}

#[test]
fn lvalue_semantic_shapes_cover_paren_brace_and_member() {
    let prog = lower_assembly("A=1; A(1)=2; A{1}=3; s = struct(); s.f = 4;");
    let body = entry_body(&prog);
    assert!(body.iter().any(|s| matches!(&s.kind, HirStmtKind::Assign(HirPlace::Index(_, indexing), _, _) if indexing.kind == IndexKind::Paren)));
    assert!(body.iter().any(|s| matches!(&s.kind, HirStmtKind::Assign(HirPlace::IndexCell(_, indexing), _, _) if indexing.kind == IndexKind::Brace)));
    assert!(body
        .iter()
        .any(|s| matches!(&s.kind, HirStmtKind::Assign(HirPlace::Member(_, _), _, _))));
}
