use runmat_builtins::Value;
use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn linear_index_and_end_keyword() {
    // v = A(end) and A(1) linear indexing
    let ast = parse("A=[1,2;3,4]; v1=A(1); v2=A(end);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    if let Value::Num(v1) = &vars[1] {
        assert_eq!(*v1, 1.0);
    } else {
        panic!("v1");
    }
    if let Value::Num(v2) = &vars[2] {
        assert_eq!(*v2, 4.0);
    } else {
        panic!("v2");
    }
}

#[test]
fn logical_mask_indexing() {
    // A([1 0 1 0]) over linearized A
    let ast = parse("A=[1,2;3,4]; idx=[1,0,1,0]; v=A(idx);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // MATLAB uses column-major linearization: A(:) = [1;3;2;4], mask [1 0 1 0] selects [1,2]
    if let Value::Tensor(v) = &vars[2] {
        assert_eq!(v.data, vec![1.0, 2.0]);
    } else {
        panic!("v");
    }
}
