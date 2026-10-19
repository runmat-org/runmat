use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "wrong_observation_order",
    binding_variant = "default",
    requested_outputs = true,
    call_observation = true
)]
fn wrong_observation_order(
    observation: Option<CallAdmissionSink>,
    requested_outputs: Option<usize>,
) {}

fn main() {}
