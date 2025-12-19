use runmat_macros::runtime_builtin;

#[runtime_builtin(name = "oops", builtin_path = "tests::oops")]
fn bad(x: i32) -> i32 {
    x
}

fn main() {} 