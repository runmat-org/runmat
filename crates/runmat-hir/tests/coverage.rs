use runmat_hir::{
    lower, CallKind, DefPathSegment, FunctionHandleTarget, FunctionKind, HirAssembly, HirExprKind,
    HirPlace, HirStmtKind, IndexKind, LoweringContext,
};
use runmat_parser::parse;

fn lower_assembly(src: &str) -> HirAssembly {
    let ast = parse(src).unwrap();
    lower(&ast, &LoweringContext::empty()).unwrap().assembly
}

fn entry_body(assembly: &HirAssembly) -> &[runmat_hir::HirStmt] {
    let entry = assembly.entrypoints[0].target;
    &assembly.functions[entry.0].body.statements
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
    assert!(
        matches!(&body[2].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Index(_, _)))
    );
    assert!(
        matches!(&body[3].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Index(_, _)))
    );
}

#[test]
fn end_colon_and_range_lower_to_semantic_shapes() {
    let prog = lower_assembly("A = [1,2,3,4]; A(2:end); 1:2:5; :");
    let body = entry_body(&prog);
    assert_eq!(body.len(), 4);
    assert!(
        matches!(&body[1].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Index(_, _)))
    );
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
        .semantic_index
        .calls
        .iter()
        .any(|call| matches!(call.kind, CallKind::DirectFunction(_))));
}

#[test]
fn methods_members_handles_and_anon_lower_to_semantic_shapes() {
    let method = lower_assembly("obj = 1; obj.method(1);");
    assert!(entry_body(&method).iter().any(|stmt| {
        matches!(&stmt.kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Call(_)))
    }));

    let member = lower_assembly("obj = 1; obj.field;");
    assert!(entry_body(&member).iter().any(|stmt| {
        matches!(&stmt.kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::Member(_, _)))
    }));

    let handle = lower_assembly("@sin;");
    assert!(
        matches!(&entry_body(&handle)[0].kind, HirStmtKind::ExprStmt(expr, _) if matches!(expr.kind, HirExprKind::FunctionHandle(FunctionHandleTarget::Builtin(_)) | HirExprKind::FunctionHandle(FunctionHandleTarget::DynamicName(_))))
    );

    let imported_handle = lower_assembly("import Point.origin; @origin;");
    assert!(entry_body(&imported_handle).iter().any(|stmt| matches!(
        &stmt.kind,
        HirStmtKind::ExprStmt(expr, _)
            if matches!(
                &expr.kind,
                HirExprKind::FunctionHandle(FunctionHandleTarget::DefPath(path))
                    if path.module.display_name().as_deref() == Some("Point.origin")
                        && matches!(path.item.as_slice(), [DefPathSegment::Function(_)])
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
    let result = lower(&ast, &LoweringContext::empty()).unwrap();
    assert!(result.semantic_index.calls.iter().any(|call| matches!(
        &call.kind,
        CallKind::PackageFunction(path)
            if path.module.display_name().as_deref() == Some("Point.origin")
                && matches!(path.item.as_slice(), [DefPathSegment::Function(_)])
    )));
}

#[test]
fn wildcard_import_call_lowers_to_package_function_identity() {
    let src = "import Point.*; __register_test_classes(); o = origin();";
    let ast = parse(src).unwrap();
    let result = lower(&ast, &LoweringContext::empty()).unwrap();
    assert!(result.semantic_index.calls.iter().any(|call| matches!(
        &call.kind,
        CallKind::PackageFunction(path)
            if path.module.display_name().as_deref() == Some("Point.origin")
                && matches!(path.item.as_slice(), [DefPathSegment::Function(_)])
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
