use runmat_hir::{
    lower, BindingId, BindingRole, HirAssembly, HirExprKind, HirPlace, HirStmtKind, LoweringContext,
};
use runmat_parser::parse;

fn lower_assembly(src: &str) -> HirAssembly {
    let ast = parse(src).unwrap();
    lower(&ast, &LoweringContext::empty()).unwrap().assembly
}

fn entry_body(assembly: &HirAssembly) -> &[runmat_hir::HirStmt] {
    let entry = &assembly.entrypoints[0];
    &assembly.functions[entry.target.0].body.statements
}

fn binding_named(assembly: &HirAssembly, name: &str) -> BindingId {
    assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == name)
        .map(|binding| binding.id)
        .unwrap_or_else(|| panic!("missing binding {name}"))
}

#[test]
fn lower_simple_assignments() {
    let assembly = lower_assembly("x=1; y=x+2;");
    let body = entry_body(&assembly);
    assert_eq!(body.len(), 2);
    let x_id = binding_named(&assembly, "x");
    let y_id = binding_named(&assembly, "y");
    match &body[0].kind {
        HirStmtKind::Assign(HirPlace::Binding(id), expr, _) => {
            assert_eq!(*id, x_id);
            assert!(matches!(expr.kind, HirExprKind::Number(_)));
        }
        other => panic!("unexpected stmt: {other:?}"),
    }
    match &body[1].kind {
        HirStmtKind::Assign(HirPlace::Binding(id), expr, _) => {
            assert_eq!(*id, y_id);
            let HirExprKind::Binary(left, _, _) = &expr.kind else {
                panic!("rhs not binary");
            };
            assert!(matches!(left.kind, HirExprKind::Binding(id) if id == x_id));
        }
        other => panic!("unexpected stmt: {other:?}"),
    }
    assert_ne!(x_id, y_id);
}

#[test]
fn error_on_undefined_variable() {
    let ast = parse("y=x").unwrap();
    assert!(lower(&ast, &LoweringContext::empty()).is_err());
}

#[test]
fn function_scope_shadows_outer_variable() {
    let assembly = lower_assembly("x=1; function y=foo(x); y=x+1; end");
    let outer_id = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "x" && matches!(binding.role, BindingRole::Local))
        .unwrap()
        .id;
    let foo = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "foo")
        .unwrap();
    let param_id = foo.params[0];
    assert_ne!(param_id, outer_id);
    let HirStmtKind::Assign(_, expr, _) = &foo.body.statements[0].kind else {
        panic!("expected function assignment");
    };
    let HirExprKind::Binary(left, _, _) = &expr.kind else {
        panic!("expected binary expression");
    };
    assert!(matches!(left.kind, HirExprKind::Binding(id) if id == param_id));
}

#[test]
fn function_output_reuses_param_binding_when_names_match() {
    let assembly = lower_assembly("function x = bump(x); y = x; x = x + 1; end");
    let bump = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "bump")
        .unwrap();
    assert_eq!(bump.params.len(), 1);
    assert_eq!(bump.outputs.len(), 1);
    assert_eq!(bump.params[0], bump.outputs[0]);
    let shared = bump.params[0];

    let HirStmtKind::Assign(_, first_expr, _) = &bump.body.statements[0].kind else {
        panic!("expected first assignment");
    };
    assert!(matches!(first_expr.kind, HirExprKind::Binding(id) if id == shared));

    let HirStmtKind::Assign(HirPlace::Binding(assign_id), second_expr, _) =
        &bump.body.statements[1].kind
    else {
        panic!("expected second assignment");
    };
    assert_eq!(*assign_id, shared);
    let HirExprKind::Binary(left, _, _) = &second_expr.kind else {
        panic!("expected binary update");
    };
    assert!(matches!(left.kind, HirExprKind::Binding(id) if id == shared));
}

#[test]
fn undefined_variable_in_function_errors() {
    let ast = parse("function y=f(); y=z; end").unwrap();
    assert!(lower(&ast, &LoweringContext::empty()).is_err());
}

#[test]
fn reassignment_reuses_binding_identity() {
    let assembly = lower_assembly("x=1; x=[1,2];");
    let body = entry_body(&assembly);
    let x_id = binding_named(&assembly, "x");
    for stmt in body {
        let HirStmtKind::Assign(HirPlace::Binding(id), _, _) = &stmt.kind else {
            panic!("expected assignment");
        };
        assert_eq!(*id, x_id);
    }
}
