use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "nonbool_observation",
    binding_variant = "default",
    call_observation = "true"
)]
fn nonbool_observation() {}

fn main() {}
