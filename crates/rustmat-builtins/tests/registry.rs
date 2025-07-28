use rustmat_builtins::builtins;
use rustmat_macros::matlab_fn;

#[matlab_fn(name = "add")]
fn add(x: i32, y: i32) -> i32 {
    x + y
}

#[matlab_fn(name = "sub")]
/// subtract numbers
fn sub(x: i32, y: i32) -> i32 {
    x - y
}

#[test]
fn contains_registered_functions() {
    let names: Vec<&str> = builtins().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
}

#[test]
fn docs_available() {
    for b in builtins() {
        if b.name == "sub" {
            assert_eq!(b.doc, " subtract numbers");
            return;
        }
    }
    panic!("sub builtin not found");
}
