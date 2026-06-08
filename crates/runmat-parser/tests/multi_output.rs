use runmat_parser::{Expr, MultiAssignTarget, Stmt};

mod parse;
use parse::parse;

#[test]
fn multi_output_with_placeholder() {
    let program = parse("[a, ~, c] = f(x, y)").unwrap();
    match &program.body[0] {
        Stmt::MultiAssign(targets, rhs, _, _) => {
            let expected = vec![
                MultiAssignTarget::LValue(runmat_parser::LValue::Var("a".to_string())),
                MultiAssignTarget::Discard,
                MultiAssignTarget::LValue(runmat_parser::LValue::Var("c".to_string())),
            ];
            assert_eq!(targets, &expected);
            match rhs {
                Expr::FuncCall(name, args, _) => {
                    assert_eq!(name, "f");
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("expected function call"),
            }
        }
        _ => panic!("expected multi-assign"),
    }
}
