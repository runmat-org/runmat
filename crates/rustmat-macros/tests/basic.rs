use rustmat_builtins::builtins;
use rustmat_macros::runtime_builtin;

#[runtime_builtin(name = "foo")]
fn foo(x: i32) -> Result<i32, String> {
    Ok(x + 1)
}

#[test]
fn works() {
    assert_eq!(foo(1).unwrap(), 2);
    let names: Vec<&str> = builtins().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"foo"));
}
