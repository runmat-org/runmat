use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn basic_command_syntax_to_func_call() {
    let program = parse("plot x y").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), false) => {
            assert_eq!(name, "plot");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::Ident(ref n) if n == "x"));
            assert!(matches!(args[1], Expr::Ident(ref n) if n == "y"));
        }
        _ => panic!("expected plot x y to parse as function call"),
    }
}

#[test]
fn command_syntax_with_numbers_and_strings() {
    let program = parse("cmd 42 'ok'").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), false) => {
            assert_eq!(name, "cmd");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::Number(ref n) if n == "42"));
            assert!(matches!(args[1], Expr::String(_)));
        }
        _ => panic!("expected command with number and string args"),
    }
}


