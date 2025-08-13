use runmat_builtins::Value;
use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn basic_matrix_and_slices() {
    // A(:)
    let ast = parse("A=[1,2;3,4]; v=A(:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(v) = &vars[1] {
        assert_eq!(v.rows(), 4);
        assert_eq!(v.cols(), 1);
        assert_eq!(v.data, vec![1.0, 3.0, 2.0, 4.0]);
    } else { panic!("Expected vector from A(:)"); }

    // A(:,2)
    let ast = parse("A=[1,2;3,4]; c=A(:,2)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(c) = &vars[1] {
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.data, vec![2.0, 4.0]);
    } else { panic!("Expected column slice A(:,2)"); }

    // A(2,:)
    let ast = parse("A=[1,2;3,4]; r=A(2,:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(r) = &vars[1] {
        assert_eq!(r.rows(), 1);
        assert_eq!(r.cols(), 2);
        assert_eq!(r.data, vec![3.0, 4.0]);
    } else { panic!("Expected row slice A(2,:)"); }

    // A(:,:)
    let ast = parse("A=[1,2;3,4]; B=A(:,:)").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Tensor(b) = &vars[1] {
        assert_eq!(b.rows(), 2);
        assert_eq!(b.cols(), 2);
        // Column-major storage
        assert_eq!(b.data, vec![1.0, 3.0, 2.0, 4.0]);
    } else { panic!("Expected full slice A(:,:)"); }
}


