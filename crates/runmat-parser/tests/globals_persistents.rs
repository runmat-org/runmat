use runmat_parser::Stmt;

mod parse;
use parse::parse;

#[test]
fn global_and_persistent() {
    let program = parse("global a, b; persistent x,y").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::Global(names, _) => {
            let expected: Vec<String> = vec!["a".into(), "b".into()];
            assert_eq!(names, &expected);
        }
        _ => panic!("expected global"),
    }
    match &program.body[1] {
        Stmt::Persistent(names, _) => {
            let expected: Vec<String> = vec!["x".into(), "y".into()];
            assert_eq!(names, &expected);
        }
        _ => panic!("expected persistent"),
    }
}
