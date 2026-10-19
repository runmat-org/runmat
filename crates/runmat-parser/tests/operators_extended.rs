use runmat_parser::{BinOp, Expr, Stmt, UnOp};

mod parse;
use parse::parse;

#[test]
fn non_conjugate_transpose_and_dot_plus() {
    let program = parse("A.'; A .+ B").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Unary(UnOp::NonConjugateTranspose, _, _), true, _) => {}
        _ => panic!("expected non-conjugate transpose"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::Binary(_, BinOp::Add, _, _), false, _) => {}
        _ => panic!("expected dot-plus to parse as add"),
    }
}

#[test]
fn numeric_literals_do_not_absorb_adjacent_dot_operators() {
    for (source, expected) in [
        ("1./2", BinOp::ElemDiv),
        ("2.^3", BinOp::ElemPow),
        ("1.*2", BinOp::ElemMul),
        ("1.\\2", BinOp::ElemLeftDiv),
    ] {
        let program = parse(source).expect("adjacent numeric dot operator");
        let [Stmt::ExprStmt(Expr::Binary(left, op, right, _), _, _)] = program.body.as_slice()
        else {
            panic!("expected binary expression for {source}");
        };
        assert_eq!(*op, expected, "{source}");
        assert!(matches!(&**left, Expr::Number(_, _)), "{source}");
        assert!(matches!(&**right, Expr::Number(_, _)), "{source}");
    }
    let program = parse("1.'").expect("numeric nonconjugate transpose");
    assert!(matches!(
        program.body.as_slice(),
        [Stmt::ExprStmt(
            Expr::Unary(UnOp::NonConjugateTranspose, _, _),
            _,
            _
        )]
    ));
}

fn expression_tree(source: &str) -> String {
    fn tree(expr: &Expr) -> String {
        match expr {
            Expr::Number(value, _) | Expr::Ident(value, _) => value.clone(),
            Expr::Unary(op, operand, _) => {
                let operator = match op {
                    UnOp::Plus => "+",
                    UnOp::Minus => "-",
                    UnOp::Not => "~",
                    _ => panic!("unexpected unary operator: {op:?}"),
                };
                format!("({operator} {})", tree(operand))
            }
            Expr::Binary(left, op, right, _) => {
                let operator = match op {
                    BinOp::Pow => "^",
                    BinOp::ElemPow => ".^",
                    BinOp::Mul => "*",
                    BinOp::Add => "+",
                    _ => panic!("unexpected binary operator: {op:?}"),
                };
                format!("({operator} {} {})", tree(left), tree(right))
            }
            _ => panic!("unexpected expression: {expr:?}"),
        }
    }
    let program = parse(source).expect("valid operator expression");
    let [Stmt::ExprStmt(expr, _, _)] = program.body.as_slice() else {
        panic!("expected one expression statement");
    };
    tree(expr)
}

#[test]
fn exponent_operands_accept_prefix_signs() {
    for (source, expected) in [
        ("2^-52", "(^ 2 (- 52))"),
        ("2^+3", "(^ 2 (+ 3))"),
        ("2 .^ -3", "(.^ 2 (- 3))"),
        ("2 .^ +3", "(.^ 2 (+ 3))"),
        ("x.^-3", "(.^ x (- 3))"),
        ("x.^+3", "(.^ x (+ 3))"),
        ("2^-+3", "(^ 2 (- (+ 3)))"),
        ("2^~0", "(^ 2 (~ 0))"),
    ] {
        assert_eq!(expression_tree(source), expected, "{source}");
    }
    assert!(parse("assert(value == 2^-52)").is_ok());
}

#[test]
fn signed_exponents_preserve_power_precedence_and_existing_right_association() {
    for (source, expected) in [
        ("-2^2", "(- (^ 2 2))"),
        ("(-2)^2", "(^ (- 2) 2)"),
        ("2^3^2", "(^ 2 (^ 3 2))"),
        ("2 .^ 3 .^ 2", "(.^ 2 (.^ 3 2))"),
        ("2^-3^2", "(^ 2 (- (^ 3 2)))"),
        ("2 .^ -3 .^ 2", "(.^ 2 (- (.^ 3 2)))"),
        ("2^-3*4", "(* (^ 2 (- 3)) 4)"),
        ("2^-3+4", "(+ (^ 2 (- 3)) 4)"),
    ] {
        assert_eq!(expression_tree(source), expected, "{source}");
    }
}

#[test]
fn signed_exponent_spans_and_incomplete_operands() {
    let program = parse("2^-52; 3").expect("two expressions");
    assert_eq!(program.body.len(), 2);
    let Stmt::ExprStmt(Expr::Binary(_, BinOp::Pow, exponent, span), _, _) = &program.body[0] else {
        panic!("expected power");
    };
    assert_eq!((span.start, span.end), (0, 5));
    assert_eq!((exponent.span().start, exponent.span().end), (2, 5));
    for source in ["2^-", "2 .^ +", "2^", "2 .^ -"] {
        assert!(parse(source).is_err(), "missing exponent in {source}");
    }
}
