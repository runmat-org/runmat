use runmat_parser::{parse_with_options, CompatMode, Expr, ParserOptions, Stmt};

mod parse;
use parse::parse;

#[test]
fn basic_command_syntax_to_func_call() {
    let program = parse("plot x y").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
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
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), _, _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(_, _)));
            assert!(matches!(args[1], Expr::Number(_, _)));
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn command_form_ellipsis_consumes_multiple_trailing_newlines() {
    // `...` followed by more than one newline should still continue the command.
    let program = parse("echo 'hello' ...\n\n 42").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), _, _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(_, _)));
            assert!(matches!(args[1], Expr::Number(_, _)));
        }
        _ => panic!("expected command form call with two args"),
    }
}

#[test]
fn command_form_does_not_continue_on_bare_newline() {
    let program = parse("foo\nbar").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Ident(name, _), _, _) => assert_eq!(name, "foo"),
        other => panic!("statement 0: expected identifier, got {other:?}"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::Ident(name, _), _, _) => assert_eq!(name, "bar"),
        other => panic!("statement 1: expected identifier, got {other:?}"),
    }

    let known_command = parse("grid\non").unwrap();
    assert_eq!(known_command.body.len(), 2);
    match &known_command.body[0] {
        Stmt::ExprStmt(Expr::Ident(name, _), _, _) => assert_eq!(name, "grid"),
        other => panic!("statement 0: expected identifier, got {other:?}"),
    }
}

#[test]
fn command_form_with_end_token_as_arg() {
    // end appears as a bare token in command-form; parser should treat it as an identifier literal
    let program = parse("foo end bar").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), _, _) => {
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
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
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
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "hold");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"on\""));
        }
        _ => panic!("expected hold command to become command call"),
    }
}

#[test]
fn colorbar_without_arg_allowed() {
    let program = parse_with_options("colorbar", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "colorbar");
            assert!(args.is_empty());
        }
        _ => panic!("expected colorbar()"),
    }
}

#[test]
fn drawnow_without_arg_is_command_form() {
    let program = parse_with_options("drawnow", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "drawnow");
            assert!(args.is_empty());
        }
        _ => panic!("expected drawnow command form"),
    }
}

#[test]
fn axis_on_off_rewrite_to_string_args() {
    for src in ["axis on", "axis off"] {
        let program = parse_with_options(src, ParserOptions::new(CompatMode::Matlab)).unwrap();
        match &program.body[0] {
            Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
                assert_eq!(name, "axis");
                assert_eq!(args.len(), 1);
                let expected = src.split_whitespace().nth(1).unwrap();
                assert!(matches!(&args[0], Expr::String(s, _) if s.trim_matches('"') == expected));
            }
            _ => panic!("expected {src} command form"),
        }
    }
}

#[test]
fn warning_stringifies_bare_word_args() {
    let program =
        parse_with_options("warning off all", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "warning");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"off\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"all\""));
        }
        _ => panic!("expected warning command form"),
    }
}

#[test]
fn close_all_stringifies_bare_word_arg() {
    let program = parse_with_options("close all", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "close");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"all\""));
        }
        _ => panic!("expected close all to become close(\"all\")"),
    }
}

#[test]
fn clear_all_stringifies_bare_word_arg() {
    let program = parse_with_options("clear all", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "clear");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"all\""));
        }
        _ => panic!("expected clear all to become clear(\"all\")"),
    }
}

#[test]
fn clear_without_arg_is_command_form() {
    let program = parse_with_options("clear", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "clear");
            assert!(args.is_empty());
        }
        _ => panic!("expected clear command form"),
    }
}

#[test]
fn clc_without_arg_is_command_form() {
    let program = parse_with_options("clc", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "clc");
            assert!(args.is_empty());
        }
        _ => panic!("expected clc command form"),
    }
}

#[test]
fn pause_without_arg_is_command_form() {
    let program = parse_with_options("pause", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "pause");
            assert!(args.is_empty());
        }
        _ => panic!("expected pause command form"),
    }
}

#[test]
fn clearvars_stringifies_bare_word_args() {
    let program =
        parse_with_options("clearvars x y", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "clearvars");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"x\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"y\""));
        }
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "clearvars");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"x\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"y\""));
        }
        _ => panic!("expected clearvars x y to become clearvars(\"x\", \"y\")"),
    }
}

#[test]
fn clearvars_except_stringifies_dash_option_and_names() {
    let program = parse_with_options(
        "clearvars -except x y",
        ParserOptions::new(CompatMode::Matlab),
    )
    .unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "clearvars");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"-except\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"x\""));
            assert!(matches!(args[2], Expr::String(ref s, _) if s == "\"y\""));
        }
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), false, _) => {
            assert_eq!(name, "clearvars");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"-except\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"x\""));
            assert!(matches!(args[2], Expr::String(ref s, _) if s == "\"y\""));
        }
        _ => panic!("expected clearvars -except x y to become a clearvars call"),
    }
}

#[test]
fn print_command_form_stringifies_dash_options_and_dotted_filename() {
    let program = parse_with_options(
        "print -dpng -r300 command_style_plot.png",
        ParserOptions::new(CompatMode::Matlab),
    )
    .unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "print");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"-dpng\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"-r300\""));
            assert!(matches!(args[2], Expr::String(ref s, _) if s == "\"command_style_plot.png\""));
        }
        other => panic!("expected print command form, got {other:?}"),
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
