mod errors;
pub(super) mod integer;
mod signatures;

use crate::*;

pub use errors::*;

pub const COPYFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: signatures::SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: errors::ERRORS,
};
