#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

#[test]
fn simple_range() {
    let vars = execute_source("x = 1:4").unwrap();
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
    let vars = execute_source("x = 1:2:5").unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.data, vec![1.0, 3.0, 5.0]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn descending_range() {
    // MATLAB: 5:-2:0 -> [5, 3, 1]
    let vars = execute_source("x = 5:-2:0").unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.data, vec![5.0, 3.0, 1.0]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn leading_dot_step_range_executes() {
    let vars = execute_source("x = 0:.1:0.3").unwrap();
    if let Value::Tensor(t) = &vars[0] {
        assert_eq!(t.rows(), 1);
        assert_eq!(t.cols(), 4);
        assert_eq!(t.data, vec![0.0, 0.1, 0.2, 0.3]);
    } else {
        panic!("expected tensor");
    }
}
