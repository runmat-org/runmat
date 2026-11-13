use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "missing_observation",
    binding_variant = "default",
    call_observation = true
)]
fn missing_observation() {}

fn main() {}
