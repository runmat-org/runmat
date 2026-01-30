use runmat_hir::{lower, HirExprKind, HirStmt, LoweringContext, Type};
use runmat_parser::parse;

#[test]
fn lower_simple_assignments() {
    let ast = parse("x=1; y=x+2;").unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap().hir;
    assert_eq!(hir.body.len(), 2);
    let (x_id, y_id) = match (&hir.body[0], &hir.body[1]) {
        (HirStmt::Assign(x_id, x_expr, _, _), HirStmt::Assign(y_id, _, _, _)) => {
            assert!(matches!(x_expr.kind, HirExprKind::Number(_)));
            (*x_id, *y_id)
        }
        _ => panic!("unexpected stmt kinds"),
    };
    // second assignment should reference first variable
    if let HirStmt::Assign(_, rhs, _, _) = &hir.body[1] {
        if let HirExprKind::Binary(left, _, _) = &rhs.kind {
            if let HirExprKind::Var(id) = left.kind {
                assert_eq!(id, x_id);
            } else {
                panic!("lhs not var");
            }
        } else {
            panic!("rhs not binary");
        }
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
    let ast = parse("x=1; function y=foo(x); y=x+1; end").unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap().hir;
    // outer assignment defines variable 0
    let outer_id = match &hir.body[0] {
        HirStmt::Assign(id, _, _, _) => *id,
        _ => panic!(),
    };
    if let HirStmt::Function { params, body, .. } = &hir.body[1] {
        let param_id = params[0];
        assert_ne!(param_id, outer_id);
        if let HirStmt::Assign(_, expr, _, _) = &body[0] {
            if let HirExprKind::Binary(left, _, _) = &expr.kind {
                if let HirExprKind::Var(id) = left.kind {
                    assert_eq!(id, param_id);
                } else {
                    panic!();
                }
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    } else {
        panic!();
    }
}

#[test]
fn undefined_variable_in_function_errors() {
    let ast = parse("function y=f(); y=z; end").unwrap();
    assert!(lower(&ast, &LoweringContext::empty()).is_err());
}

#[test]
fn type_inference_propagates_through_assignments() {
    let ast = parse("x=[1,2]; y=x+1;").unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap().hir;
    let (x_id, y_id) = match (&hir.body[0], &hir.body[1]) {
        (HirStmt::Assign(x_id, x_expr, _, _), HirStmt::Assign(y_id, y_expr, _, _)) => {
            assert!(matches!(x_expr.ty, Type::Tensor { .. }));
            assert!(matches!(y_expr.ty, Type::Tensor { .. }));
            (*x_id, *y_id)
        }
        _ => panic!("unexpected statements"),
    };
    assert_ne!(x_id, y_id);
}

#[test]
fn reassignment_updates_variable_type() {
    let ast = parse("x=1; x=[1,2];").unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap().hir;
    if let HirStmt::Assign(id, expr2, _, _) = &hir.body[1] {
        assert!(matches!(expr2.ty, Type::Tensor { .. }));
        // ensure variable id is same as first assignment
        if let HirStmt::Assign(id1, _, _, _) = &hir.body[0] {
            assert_eq!(*id, *id1);
        } else {
            panic!();
        }
    } else {
        panic!();
    }
}
