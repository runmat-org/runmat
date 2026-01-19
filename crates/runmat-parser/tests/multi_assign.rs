use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn multi_assign_parses() {
    let program = parse("[a,b] = f(x)").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::MultiAssign(names, rhs, suppressed, _) => {
            let expected: Vec<String> = vec!["a".into(), "b".into()];
            assert_eq!(names, &expected);
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
        Stmt::MultiAssign(names, rhs, suppressed, _) => {
            assert_eq!(names, &vec!["a".to_string(), "b".to_string()]);
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
        Stmt::MultiAssign(names, rhs, suppressed, _) => {
            assert_eq!(names, &vec!["c".to_string(), "d".to_string()]);
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
