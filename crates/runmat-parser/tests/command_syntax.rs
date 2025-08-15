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
fn command_form_with_quoted_and_ellipsis() {
    // echo with quoted arg and ellipsis line continuation
    let program = parse("echo 'hello' ...\n 42").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(_)));
            assert!(matches!(args[1], Expr::Number(_)));
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn command_form_with_end_token_as_arg() {
    // end appears as a bare token in command-form; parser should treat it as an identifier literal
    let program = parse("foo end bar").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), _) => {
            assert_eq!(name, "foo");
            assert_eq!(args.len(), 2);
            // We currently parse end as EndKeyword only inside expression contexts; here it remains Ident("end")
            assert!(matches!(args[0], Expr::Ident(ref s) if s == "end")
                || matches!(args[0], Expr::EndKeyword));
            assert!(matches!(args[1], Expr::Ident(ref s) if s == "bar"));
        }
        _ => panic!("expected command form call"),
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


