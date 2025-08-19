use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn switch_and_try_catch() {
    // Newline-separated block to exercise parse_block logic
    let program = parse("switch x\n case 1\n y=2;\n otherwise\n y=3;\n end").unwrap();
    match &program.body[0] {
        Stmt::Switch {
            expr,
            cases,
            otherwise,
        } => {
            assert!(matches!(expr, Expr::Ident(ref n) if n == "x"));
            assert_eq!(cases.len(), 1);
            assert!(otherwise.is_some());
        }
        _ => panic!("expected switch statement"),
    }

    let program = parse("try x; catch e; y; end").unwrap();
    match &program.body[0] {
        Stmt::TryCatch {
            try_body,
            catch_var,
            catch_body,
        } => {
            assert!(!try_body.is_empty());
            assert_eq!(catch_var.as_deref(), Some("e"));
            assert!(!catch_body.is_empty());
        }
        _ => panic!("expected try/catch"),
    }
}
