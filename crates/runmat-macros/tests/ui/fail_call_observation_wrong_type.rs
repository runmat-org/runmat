use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "wrong_observation_type",
    binding_variant = "default",
    call_observation = true
)]
fn wrong_observation_type(observation: Option<u32>) {}

fn main() {}
