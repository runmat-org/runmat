use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn multi_assign_parses() {
    let program = parse("[a,b] = f(x)").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::MultiAssign(names, rhs, suppressed) => {
            let expected: Vec<String> = vec!["a".into(), "b".into()];
            assert_eq!(names, &expected);
            assert!(!suppressed);
            match rhs {
                Expr::FuncCall(name, args) => {
                    assert_eq!(name, "f");
                    assert_eq!(args, &vec![Expr::Ident("x".into())]);
                }
                _ => panic!("expected function call"),
            }
        }
        _ => panic!("expected multi-assign"),
    }
}


