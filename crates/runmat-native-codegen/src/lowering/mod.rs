mod assembly;
mod super_constructor;
mod context;
mod function;
mod inventory;
mod operation;
mod requirements;
mod terminator;

pub use assembly::{
    lower_executable, verify_against_analysis, verify_against_manifest, verify_against_mir,
    NativeEntrypointBinding, NativeLoweringInput,
};
