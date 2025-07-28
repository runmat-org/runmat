use rustmat_parser::{parse, BinOp, Expr, Program, Stmt, UnOp};

#[test]
fn parse_expression() {
    let program = parse("1 + 2 * 3").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Add,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("2".into())),
                    BinOp::Mul,
                    Box::new(Expr::Number("3".into()))
                ))
            ))]
        }
    );
}

#[test]
fn parse_assignment() {
    let program = parse("x = 4 + 5;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::Binary(
                    Box::new(Expr::Number("4".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("5".into()))
                )
            )]
        }
    );
}

#[test]
fn precedence_and_associativity() {
    let program = parse("1 - 2 - 3").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("1".into())),
                    BinOp::Sub,
                    Box::new(Expr::Number("2".into()))
                )),
                BinOp::Sub,
                Box::new(Expr::Number("3".into()))
            ))]
        }
    );
}

#[test]
fn parentheses_override_precedence() {
    let program = parse("1 * (2 + 3)").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Mul,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("2".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("3".into()))
                ))
            ))]
        }
    );
}

#[test]
fn multiple_statements() {
    let program = parse("x = 1; y = 2;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![
                Stmt::Assign("x".into(), Expr::Number("1".into())),
                Stmt::Assign("y".into(), Expr::Number("2".into()))
            ]
        }
    );
}

#[test]
fn trailing_semicolon_is_allowed() {
    let program = parse("1 + 2;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Add,
                Box::new(Expr::Number("2".into()))
            ))]
        }
    );
}

#[test]
fn final_semicolon_not_required() {
    let program = parse("1 + 2").unwrap();
    assert_eq!(program.body.len(), 1);
}

#[test]
fn empty_input_yields_empty_program() {
    let program = parse("").unwrap();
    assert!(program.body.is_empty());
}

#[test]
fn missing_closing_paren_produces_error() {
    assert!(parse("(1 + 2").is_err());
}

#[test]
fn invalid_token_produces_error() {
    assert!(parse("1 + $").is_err());
}

#[test]
fn incomplete_expression_produces_error() {
    assert!(parse("1 +").is_err());
}

#[test]
fn power_is_right_associative() {
    let program = parse("2 ^ 3 ^ 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("2".into())),
                BinOp::Pow,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("3".into())),
                    BinOp::Pow,
                    Box::new(Expr::Number("2".into()))
                ))
            ))]
        }
    );
}

#[test]
fn unary_minus_precedence() {
    let program = parse("-1 + 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Unary(UnOp::Minus, Box::new(Expr::Number("1".into())))),
                BinOp::Add,
                Box::new(Expr::Number("2".into()))
            ))]
        }
    );
}

#[test]
fn unary_minus_with_power() {
    let program = parse("-2 ^ 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Unary(
                UnOp::Minus,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("2".into())),
                    BinOp::Pow,
                    Box::new(Expr::Number("2".into()))
                ))
            ))]
        }
    );
}

#[test]
fn parse_simple_matrix() {
    let program = parse("[1,2;3,4]").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Matrix(vec![
                vec![Expr::Number("1".into()), Expr::Number("2".into())],
                vec![Expr::Number("3".into()), Expr::Number("4".into())],
            ]))]
        }
    );
}

#[test]
fn parse_empty_matrix() {
    let program = parse("[]").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Matrix(vec![]))]
        }
    );
}

#[test]
fn matrix_with_expressions() {
    let program = parse("[1+2,3*4]").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Matrix(vec![vec![
                Expr::Binary(
                    Box::new(Expr::Number("1".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("2".into()))
                ),
                Expr::Binary(
                    Box::new(Expr::Number("3".into())),
                    BinOp::Mul,
                    Box::new(Expr::Number("4".into()))
                ),
            ]]))]
        }
    );
}

#[test]
fn nested_matrix_literal() {
    let program = parse("[1,[2,3]]").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Matrix(vec![vec![
                Expr::Number("1".into()),
                Expr::Matrix(vec![vec![
                    Expr::Number("2".into()),
                    Expr::Number("3".into())
                ]])
            ],]))]
        }
    );
}

#[test]
fn missing_matrix_comma_is_error() {
    assert!(parse("[1 2]").is_err());
}

#[test]
fn missing_closing_bracket_is_error() {
    assert!(parse("[1,2").is_err());
}

#[test]
fn left_division_operator() {
    let program = parse("4 \\ 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("4".into())),
                BinOp::LeftDiv,
                Box::new(Expr::Number("2".into())),
            ))]
        }
    );
}

#[test]
fn elementwise_power_operator() {
    let program = parse("3 .^ 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("3".into())),
                BinOp::Pow,
                Box::new(Expr::Number("2".into())),
            ))]
        }
    );
}
