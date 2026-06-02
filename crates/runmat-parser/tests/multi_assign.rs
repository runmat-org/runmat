use runmat_parser::{Expr, LValue, MultiAssignTarget, Stmt};

mod parse;
use parse::parse;

#[test]
fn multi_assign_parses() {
    let program = parse("[a,b] = f(x)").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::MultiAssign(targets, rhs, suppressed, _) => {
            let expected = vec![
                MultiAssignTarget::LValue(LValue::Var("a".into())),
                MultiAssignTarget::LValue(LValue::Var("b".into())),
            ];
            assert_eq!(targets, &expected);
            assert!(!suppressed);
            match rhs {
                Expr::FuncCall(name, args, _) => {
                    assert_eq!(name, "f");
                    assert_eq!(args.len(), 1);
                    assert!(matches!(args[0], Expr::Ident(ref n, _) if n == "x"));
                }
                _ => panic!("expected function call"),
            }
        }
        _ => panic!("expected multi-assign"),
    }
}

#[test]
fn multi_assign_semicolon_and_newline_behavior() {
    let program = parse("[a,b] = f(x);\n[c, d] = g(y)").unwrap();
    assert_eq!(program.body.len(), 2);

    match &program.body[0] {
        Stmt::MultiAssign(targets, rhs, suppressed, _) => {
            assert_eq!(
                targets,
                &vec![
                    MultiAssignTarget::LValue(LValue::Var("a".into())),
                    MultiAssignTarget::LValue(LValue::Var("b".into()))
                ]
            );
            assert!(*suppressed);
            if let Expr::FuncCall(name, _, _) = rhs {
                assert_eq!(name, "f");
            } else {
                panic!("expected first RHS as function call");
            }
        }
        other => panic!("unexpected first stmt: {other:?}"),
    }

    match &program.body[1] {
        Stmt::MultiAssign(targets, rhs, suppressed, _) => {
            assert_eq!(
                targets,
                &vec![
                    MultiAssignTarget::LValue(LValue::Var("c".into())),
                    MultiAssignTarget::LValue(LValue::Var("d".into()))
                ]
            );
            assert!(!*suppressed);
            if let Expr::FuncCall(name, _, _) = rhs {
                assert_eq!(name, "g");
            } else {
                panic!("expected second RHS as function call");
            }
        }
        other => panic!("unexpected second stmt: {other:?}"),
    }
}

#[test]
fn multi_assign_index_cell_target_parses() {
    let program = parse("[varargout{1:nargout}] = f(x)").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::MultiAssign(targets, rhs, suppressed, _) => {
            assert!(!suppressed);
            assert_eq!(targets.len(), 1);
            match &targets[0] {
                MultiAssignTarget::LValue(LValue::IndexCell(base, indices)) => {
                    assert!(matches!(**base, Expr::Ident(ref n, _) if n == "varargout"));
                    assert_eq!(indices.len(), 1);
                }
                other => panic!("expected cell-index target, got {other:?}"),
            }
            assert!(matches!(rhs, Expr::FuncCall(ref name, _, _) if name == "f"));
        }
        other => panic!("expected multi-assign, got {other:?}"),
    }
}
