use rustmat_macros::runtime_builtin;

#[runtime_builtin(name = "oops")]
fn bad(x: i32) -> i32 {
    x
}

fn main() {} 