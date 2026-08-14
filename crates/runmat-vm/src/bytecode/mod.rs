pub mod compile;
pub mod instr;
pub mod program;

/// Portable schema for serialized [`Bytecode`] payloads.
pub const BYTECODE_SCHEMA_VERSION: u16 = 2;
/// Portable schema for serialized [`FunctionRegistry`] payloads.
pub const FUNCTION_REGISTRY_SCHEMA_VERSION: u16 = 2;

pub use compile::{compile, compile_semantic_function_registry};
pub use instr::{EmitLabel, Instr, StackEffect};
pub use program::{
    AsyncMetadata, AwaitSite, Bytecode, FunctionBytecode, FunctionRegistry, SpawnSite,
};
#[cfg(feature = "native-accel")]
pub use program::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
