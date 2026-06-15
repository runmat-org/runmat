use runmat_lexer::tokenize_detailed;
use runmat_parser::{parse_with_options, BinOp, CompatMode, Expr, LValue, ParserOptions, Stmt};

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

    let known_command = parse("hold\non").unwrap();
    assert_eq!(known_command.body.len(), 2);
    match &known_command.body[0] {
        Stmt::ExprStmt(Expr::Ident(name, _), _, _) => assert_eq!(name, "hold"),
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
fn axis_command_modes_rewrite_to_string_args() {
    for src in [
        "axis auto",
        "axis manual",
        "axis tight",
        "axis equal",
        "axis image",
        "axis ij",
        "axis xy",
        "axis on",
        "axis off",
    ] {
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
fn grid_command_forms_rewrite_to_string_args() {
    for src in ["grid on", "grid off", "grid minor"] {
        let program = parse_with_options(src, ParserOptions::new(CompatMode::Matlab)).unwrap();
        match &program.body[0] {
            Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
                assert_eq!(name, "grid");
                assert_eq!(args.len(), 1);
                let expected = src.split_whitespace().nth(1).unwrap();
                assert!(matches!(&args[0], Expr::String(s, _) if s.trim_matches('"') == expected));
            }
            _ => panic!("expected {src} command form"),
        }
    }
}

#[test]
fn grid_without_arg_is_command_form() {
    let program = parse_with_options("grid", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "grid");
            assert!(args.is_empty());
        }
        _ => panic!("expected grid command form"),
    }
}

#[test]
fn grid_zero_arg_command_form_does_not_capture_binary_expressions() {
    for (src, op) in [("grid + 1", BinOp::Add), ("grid - x", BinOp::Sub)] {
        let program = parse_with_options(src, ParserOptions::new(CompatMode::Matlab)).unwrap();
        match &program.body[0] {
            Stmt::ExprStmt(Expr::Binary(lhs, actual_op, _, _), false, _) => {
                assert_eq!(*actual_op, op);
                assert!(matches!(&**lhs, Expr::Ident(name, _) if name == "grid"));
            }
            other => panic!("expected {src} to parse as binary expression, got {other:?}"),
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
fn syms_command_form_accepts_symbolic_function_declarations() {
    let program = parse_with_options(
        "syms Y(X); cond = Y(0) == 0;",
        ParserOptions::new(CompatMode::Matlab),
    )
    .unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), true, _) => {
            assert_eq!(name, "syms");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"Y(X)\""));
        }
        other => panic!("expected syms command call, got {other:?}"),
    }
}

#[test]
fn syms_command_form_accepts_multiple_function_parameters() {
    let program =
        parse_with_options("syms f(x,y) z real", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "syms");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"f(x,y)\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"z\""));
            assert!(matches!(args[2], Expr::String(ref s, _) if s == "\"real\""));
        }
        other => panic!("expected syms command call, got {other:?}"),
    }
}

#[test]
fn syms_command_form_rejects_malformed_function_declarations() {
    for source in ["syms f(x,)", "syms Y("] {
        assert!(
            parse_with_options(source, ParserOptions::new(CompatMode::Matlab)).is_err(),
            "expected parse error for {source}"
        );
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
fn addpath_command_form_stringifies_path_words_and_options() {
    let program = parse_with_options(
        "addpath ./SourceCode:../lib -end -frozen",
        ParserOptions::new(CompatMode::Matlab),
    )
    .unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, "addpath");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(ref s, _) if s == "\"./SourceCode:../lib\""));
            assert!(matches!(args[1], Expr::String(ref s, _) if s == "\"-end\""));
            assert!(matches!(args[2], Expr::String(ref s, _) if s == "\"-frozen\""));
        }
        other => panic!("expected addpath command form, got {other:?}"),
    }
}

#[test]
fn filesystem_path_command_forms_stringify_path_words() {
    let cases: &[(&str, &str, &[&str])] = &[
        ("cd ..", "cd", &[".."]),
        ("cd ./SourceCode", "cd", &["./SourceCode"]),
        (
            "rmpath ./SourceCode:../lib",
            "rmpath",
            &["./SourceCode:../lib"],
        ),
        ("dir *.m", "dir", &["*.m"]),
        ("ls +pkg/@Thing/*.m", "ls", &["+pkg/@Thing/*.m"]),
        ("mkdir ./out/cache", "mkdir", &["./out/cache"]),
        ("rmdir ./out s", "rmdir", &["./out", "s"]),
        (
            "copyfile ./src/file.m ../dst/file.m f",
            "copyfile",
            &["./src/file.m", "../dst/file.m", "f"],
        ),
        (
            "movefile ./src/file.m ../dst/file.m",
            "movefile",
            &["./src/file.m", "../dst/file.m"],
        ),
        ("delete ./tmp/*.mat", "delete", &["./tmp/*.mat"]),
        (
            "run ./SourceCode/path_worker.m",
            "run",
            &["./SourceCode/path_worker.m"],
        ),
        (
            "save ./results/out.mat x -v7.3",
            "save",
            &["./results/out.mat", "x", "-v7.3"],
        ),
        (
            "load ./results/out.mat x",
            "load",
            &["./results/out.mat", "x"],
        ),
        (
            "which -all ./SourceCode/path_worker.m",
            "which",
            &["-all", "./SourceCode/path_worker.m"],
        ),
        (
            "whos -file ./results/out.mat",
            "whos",
            &["-file", "./results/out.mat"],
        ),
        (
            "print -dpng ./plots/figure-1.png",
            "print",
            &["-dpng", "./plots/figure-1.png"],
        ),
    ];

    for (source, expected_name, expected_args) in cases {
        assert_command_string_args(source, expected_name, expected_args);
    }
}

#[test]
fn path_command_form_does_not_steal_trailing_binary_operator() {
    let parsed = parse_with_options("mkdir ./out + 1", ParserOptions::new(CompatMode::Matlab));
    if let Ok(program) = parsed {
        assert!(
            !matches!(
                &program.body[0],
                Stmt::ExprStmt(Expr::CommandCall(name, _, _), false, _) if name == "mkdir"
            ),
            "path command with trailing binary operator parsed as a truncated command call"
        );
    }
}

#[test]
fn zero_arg_filesystem_command_forms_are_allowed() {
    for source in [
        "cd",
        "dir",
        "genpath",
        "getenv",
        "ls",
        "path",
        "print",
        "pwd",
        "save",
        "savepath",
        "tempdir",
        "tempname",
        "uigetfile",
        "who",
        "whos",
    ] {
        let program = parse_with_options(source, ParserOptions::new(CompatMode::Matlab)).unwrap();
        match &program.body[0] {
            Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
                assert_eq!(name, source);
                assert!(args.is_empty(), "{source} should not have parsed arguments");
            }
            other => panic!("expected {source} command form, got {other:?}"),
        }
    }
}

#[test]
fn elementwise_rdivide_expression_is_not_path_command_form() {
    let program = parse_with_options("x ./ y", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Binary(left, BinOp::ElemDiv, right, _), false, _) => {
            assert!(matches!(&**left, Expr::Ident(name, _) if name == "x"));
            assert!(matches!(&**right, Expr::Ident(name, _) if name == "y"));
        }
        other => panic!("expected elementwise rdivide expression, got {other:?}"),
    }
}

#[test]
fn path_command_names_do_not_steal_operator_expressions() {
    let cases = [
        ("path + 1", "path", BinOp::Add, "1"),
        ("path.*x", "path", BinOp::ElemMul, "x"),
        ("dir - x", "dir", BinOp::Sub, "x"),
    ];

    for (source, expected_left, expected_op, expected_right) in cases {
        let program = parse_with_options(source, ParserOptions::new(CompatMode::Matlab))
            .unwrap_or_else(|err| {
                panic!(
                    "{source}: parse failed: {err:?}; tokens: {:?}",
                    tokenize_detailed(source)
                )
            });
        match &program.body[0] {
            Stmt::ExprStmt(Expr::Binary(left, op, right, _), false, _) => {
                assert!(matches!(&**left, Expr::Ident(name, _) if name == expected_left));
                assert_eq!(*op, expected_op);
                match &**right {
                    Expr::Ident(name, _) => assert_eq!(name, expected_right),
                    Expr::Number(number, _) => assert_eq!(number, expected_right),
                    other => panic!("{source}: unexpected right operand {other:?}"),
                }
            }
            other => panic!("{source}: expected binary expression, got {other:?}"),
        }
    }
}

#[test]
fn known_path_command_name_member_assignment_is_not_command_form() {
    let program =
        parse_with_options("path.value = 1", ParserOptions::new(CompatMode::Matlab)).unwrap();
    match &program.body[0] {
        Stmt::AssignLValue(LValue::Member(base, field), value, false, _) => {
            assert!(matches!(&**base, Expr::Ident(name, _) if name == "path"));
            assert_eq!(field, "value");
            assert!(matches!(value, Expr::Number(n, _) if n == "1"));
        }
        other => panic!("expected member assignment, got {other:?}"),
    }
}

fn assert_command_string_args(source: &str, expected_name: &str, expected_args: &[&str]) {
    let program = parse_with_options(source, ParserOptions::new(CompatMode::Matlab))
        .unwrap_or_else(|err| {
            panic!(
                "{source}: parse failed: {err:?}; tokens: {:?}",
                tokenize_detailed(source)
            )
        });
    match &program.body[0] {
        Stmt::ExprStmt(Expr::CommandCall(name, args, _), false, _) => {
            assert_eq!(name, expected_name);
            assert_eq!(args.len(), expected_args.len());
            for (arg, expected) in args.iter().zip(expected_args) {
                assert!(
                    matches!(arg, Expr::String(actual, _) if actual.trim_matches('"') == *expected),
                    "{source}: expected string argument {expected:?}, got {arg:?}"
                );
            }
        }
        other => panic!("expected {source} command form, got {other:?}"),
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
