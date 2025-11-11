use runmat_parser::{parse, Expr, Stmt};

#[test]
fn single_of_range_parses_as_function_call() {
    let program = parse("x = single(0:B-1);").expect("parse");
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::Assign(name, expr, _) => {
            assert_eq!(name, "x");
            match expr {
                Expr::FuncCall(fname, args) => {
                    assert_eq!(fname, "single");
                    assert_eq!(args.len(), 1);
                    assert!(matches!(&args[0], Expr::Range(_, _, _)));
                }
                other => panic!("expected FuncCall(single,...), got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}


