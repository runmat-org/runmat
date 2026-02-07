mod parse;

use parse::parse;
use runmat_parser::{Expr, Stmt, UnOp};

#[test]
fn parse_range_with_negative_start_and_step() {
    let program = parse("a = -2:0.02:2;").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::Assign(name, Expr::Range(start, step, end, _), true, _) => {
            assert_eq!(name, "a");
            assert!(matches!(
                **start,
                Expr::Unary(UnOp::Minus, ref inner, _)
                    if matches!(**inner, Expr::Number(ref text, _) if text == "2")
            ));
            assert!(matches!(
                step,
                Some(step) if matches!(**step, Expr::Number(ref text, _) if text == "0.02")
            ));
            assert!(matches!(
                **end,
                Expr::Number(ref text, _) if text == "2"
            ));
        }
        other => panic!("unexpected statement: {other:?}"),
    }
}
