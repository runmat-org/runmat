use runmat_parser::{
    parse_simple as parse, parse_with_options, CompatMode, Expr, ParserOptions, Stmt,
};

#[test]
fn basic_command_syntax_to_func_call() {
    let program = parse("plot x y").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "plot");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::Ident(ref n, _) if n == "x"));
            assert!(matches!(args[1], Expr::Ident(ref n, _) if n == "y"));
        }
        _ => panic!("expected plot x y to parse as function call"),
    }
}

#[test]
fn command_form_with_quoted_and_ellipsis() {
    // echo with quoted arg and ellipsis line continuation
    let program = parse("echo 'hello' ...\n 42").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(_, _)));
            assert!(matches!(args[1], Expr::Number(_, _)));
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn command_form_with_end_token_as_arg() {
    // end appears as a bare token in command-form; parser should treat it as an identifier literal
    let program = parse("foo end bar").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "foo");
            assert_eq!(args.len(), 2);
            // We currently parse end as EndKeyword only inside expression contexts; here it remains Ident("end")
            assert!(
                matches!(args[0], Expr::Ident(ref s, _) if s == "end")
                    || matches!(args[0], Expr::EndKeyword(_))
            );
            assert!(matches!(args[1], Expr::Ident(ref s, _) if s == "bar"));
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn command_syntax_with_numbers_and_strings() {
    let program = parse("cmd 42 'ok'").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "cmd");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::Number(ref n, _) if n == "42"));
            assert!(matches!(args[1], Expr::String(_, _)));
        }
        _ => panic!("expected command with number and string args"),
    }
}

#[test]
fn hold_on_rewrites_to_string_arg() {
    let program = parse_with_options("hold on", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "hold");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"on\""));
        }
        _ => panic!("expected hold command to become func call"),
    }
}

#[test]
fn colorbar_without_arg_allowed() {
    let program = parse_with_options("colorbar", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "colorbar");
            assert!(args.is_empty());
        }
        _ => panic!("expected colorbar()"),
    }
}

#[test]
fn invalid_keyword_rejected() {
    let err = parse_with_options("grid maybe", ParserOptions::new(CompatMode::Matlab));
    assert!(err.is_err());
}

#[test]
fn strict_mode_rejects_command_syntax() {
    let err = parse_with_options("hold on", ParserOptions::new(CompatMode::Strict));
    assert!(err.is_err());
}
