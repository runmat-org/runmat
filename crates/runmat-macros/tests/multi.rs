use runmat_builtins::builtin_functions;
use runmat_macros::runtime_builtin;

mod inner {
    use super::*;

    #[runtime_builtin(name = "add", builtin_path = "tests::add")]
    pub fn add(a: i32, b: i32) -> Result<i32, String> {
        Ok(a + b)
    }
}

#[runtime_builtin(name = "sub", builtin_path = "tests::sub")]
pub fn sub(a: i32, b: i32) -> Result<i32, String> {
    Ok(a - b)
}

#[test]
fn registers_multiple_functions() {
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
}
