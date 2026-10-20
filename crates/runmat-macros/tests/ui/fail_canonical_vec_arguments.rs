use runmat_macros::runtime_builtin;

#[runtime_builtin(name = "retained_fixture", binding_variant = "default")]
fn captured(args: Vec<runmat_value::Value>) -> Result<runmat_value::Value, String> {
    Ok(args.into_iter().next().unwrap())
}

fn main() {}
