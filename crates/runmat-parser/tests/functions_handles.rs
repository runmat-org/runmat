use runmat_parser::{parse_simple as parse, BinOp, Expr, Stmt};

#[test]
fn anonymous_function_and_handle_parse() {
    // @(x) x^2
    let program = parse("@(x) x^2").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::AnonFunc { params, body }, _) => {
            let expected: Vec<String> = vec!["x".into()];
            assert_eq!(params, &expected);
            assert!(matches!(**body, Expr::Binary(_, BinOp::Pow, _)));
        }
        _ => panic!("expected anonymous function expr stmt"),
    }

    // @sin
    let program = parse("@sin").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncHandle(name), _) => assert_eq!(name, "sin"),
        _ => panic!("expected function handle expr stmt"),
    }
}
