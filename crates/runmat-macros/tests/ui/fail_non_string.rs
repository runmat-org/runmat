use runmat_macros::runtime_builtin;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinSignatureDescriptor,
};

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const NOOP_SIGNATURES: [BuiltinSignatureDescriptor; 0] = [];
const NOOP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NOOP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};

#[runtime_builtin(name = 123, descriptor(crate::NOOP_DESCRIPTOR))]
fn foo() {}

fn main() {}
