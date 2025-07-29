use rustmat_builtins::builtins;
use rustmat_macros::runtime_builtin;

mod inner {
    use super::*;

    #[runtime_builtin(name = "add")]
    pub fn add(a: i32, b: i32) -> Result<i32, String> {
        Ok(a + b)
    }
}

#[runtime_builtin(name = "sub")]
pub fn sub(a: i32, b: i32) -> Result<i32, String> {
    Ok(a - b)
}

#[test]
fn registers_multiple_functions() {
    let names: Vec<&str> = builtins().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
}
