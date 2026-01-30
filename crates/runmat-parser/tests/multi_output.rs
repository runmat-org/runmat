use runmat_parser::{Expr, Stmt};

mod parse;
use parse::parse;

#[test]
fn multi_output_with_placeholder() {
    let program = parse("[a, ~, c] = f(x, y)").unwrap();
    match &program.body[0] {
        Stmt::MultiAssign(names, rhs, _, _) => {
            let expected: Vec<String> = vec!["a".to_string(), "~".to_string(), "c".to_string()];
            assert_eq!(names, &expected);
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
