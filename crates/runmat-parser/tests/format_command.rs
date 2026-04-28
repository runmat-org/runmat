use runmat_parser::{Expr, Stmt};

mod parse;
use parse::parse;

#[test]
fn format_long_command_syntax_produces_string_arg() {
    let program = parse("format long").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "format");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(&args[0], Expr::String(s, _) if s.trim_matches('"') == "long"),
                "expected string arg \"long\", got {:?}",
                args[0]
            );
        }
        other => panic!("expected ExprStmt(FuncCall), got {other:?}"),
    }
}

#[test]
fn format_short_command_syntax_produces_string_arg() {
    let program = parse("format short").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "format");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(&args[0], Expr::String(s, _) if s.trim_matches('"') == "short"),
                "expected string arg \"short\", got {:?}",
                args[0]
            );
        }
        other => panic!("expected ExprStmt(FuncCall), got {other:?}"),
    }
}

#[test]
fn format_no_args_command_syntax_parses() {
    let program = parse("format").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "format");
            assert_eq!(args.len(), 0);
        }
        other => panic!("expected ExprStmt(FuncCall), got {other:?}"),
    }
}

#[test]
fn format_all_numeric_modes_parse() {
    // "rat" is the MATLAB-canonical keyword; "rational" is accepted as an alias.
    for mode in &[
        "shortE", "longE", "shortG", "longG", "rat", "rational", "hex",
    ] {
        let src = format!("format {mode}");
        let program = parse(&src).unwrap();
        match &program.body[0] {
            Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
                assert_eq!(name, "format");
                assert_eq!(args.len(), 1, "mode '{mode}' should produce 1 arg");
            }
            other => panic!("format {mode}: expected FuncCall, got {other:?}"),
        }
    }
}

#[test]
fn format_rat_is_the_matlab_canonical_keyword() {
    // Verify "format rat" parses (not "format rational").
    let program = parse("format rat").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "format");
            assert!(
                matches!(&args[0], Expr::String(s, _) if s.trim_matches('"') == "rat"),
                "expected string arg \"rat\""
            );
        }
        other => panic!("expected ExprStmt(FuncCall), got {other:?}"),
    }
}

#[test]
fn format_spacing_modes_parse_without_error() {
    // compact/loose are spacing modes; not yet implemented but should not be syntax errors.
    for mode in &["compact", "loose"] {
        let src = format!("format {mode}");
        assert!(
            parse(&src).is_ok(),
            "format {mode} should parse without error"
        );
    }
}

#[test]
fn format_truly_unknown_mode_is_a_parse_error() {
    assert!(
        parse("format bank").is_err(),
        "expected parse error for unknown mode 'bank'"
    );
}

#[test]
fn format_long_newline_pi_is_two_statements_not_two_args() {
    // In MATLAB a newline terminates command-form argument parsing.
    // "format long\npi" must produce two statements, not format("long", pi).
    let program = parse("format long\npi").unwrap();
    assert_eq!(program.body.len(), 2, "expected 2 statements");
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "format");
            assert_eq!(args.len(), 1);
        }
        other => panic!("statement 0: expected format FuncCall, got {other:?}"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::Ident(name, _), _, _) => {
            assert_eq!(name, "pi");
        }
        other => panic!("statement 1: expected pi Ident, got {other:?}"),
    }
}
