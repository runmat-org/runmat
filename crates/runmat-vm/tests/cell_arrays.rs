#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

#[test]
fn create_and_index_cell_2d() {
    let vars = execute_source("C = {1, 2; 3, 4}; a = C{1,2}; b = C{2,1};").unwrap();
    if let Value::Num(a) = &vars[1] {
        assert_eq!(*a, 2.0);
    } else {
        panic!("a should be 2");
    }
    if let Value::Num(b) = &vars[2] {
        assert_eq!(*b, 3.0);
    } else {
        panic!("b should be 3");
    }
}

#[test]
fn linear_cell_index() {
    let vars = execute_source("C = {10, 20, 30}; v = C{3};").unwrap();
    if let Value::Num(v) = &vars[1] {
        assert_eq!(*v, 30.0);
    } else {
        panic!("v should be 30");
    }
}
