use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "noncanonical_sequence",
    output_sequence = true,
    builtin_path = "tests::noncanonical_sequence"
)]
fn noncanonical_sequence() {}

fn main() {}
