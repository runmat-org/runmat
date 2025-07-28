use rustmat_builtins::builtins;
use rustmat_macros::matlab_fn;

mod inner {
    use super::*;

    #[matlab_fn(name = "add")]
    pub fn add(x: i32, y: i32) -> i32 {
        x + y
    }
}

#[matlab_fn(name = "sub")]
pub fn sub(x: i32, y: i32) -> i32 {
    x - y
}

#[test]
fn registers_multiple_functions() {
    let names: Vec<&str> = builtins().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
}
