use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "bad_sequence_flag",
    binding_variant = "default",
    output_sequence = "true",
    builtin_path = "tests::bad_sequence_flag"
)]
fn bad_sequence_flag() {}

fn main() {}
