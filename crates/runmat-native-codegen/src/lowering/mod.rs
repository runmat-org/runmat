mod assembly;
mod context;
mod function;
mod inventory;
mod operation;
mod requirements;
mod terminator;

pub use assembly::{
    lower_executable, verify_against_manifest, verify_against_mir, NativeLoweringInput,
};
