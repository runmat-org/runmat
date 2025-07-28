use rustmat_builtins::builtins;
use rustmat_macros::matlab_fn;

#[matlab_fn(name = "foo")]
fn foo(x: i32) -> i32 {
    x + 1
}

#[test]
fn works() {
    assert_eq!(foo(1), 2);
    let names: Vec<&str> = builtins().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"foo"));
}
