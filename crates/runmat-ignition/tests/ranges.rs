use runmat_builtins::Value;
use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn simple_range() {
    let ast = parse("x = 1:4").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.rows(), 1);
        assert_eq!(t.cols(), 4);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn range_with_step() {
    let ast = parse("x = 1:2:5").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.data, vec![1.0, 3.0, 5.0]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn descending_range() {
    // MATLAB: 5:-2:0 -> [5, 3, 1]
    let ast = parse("x = 5:-2:0").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.data, vec![5.0, 3.0, 1.0]);
    } else {
        panic!("expected tensor");
    }
}
