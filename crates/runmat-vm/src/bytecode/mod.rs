pub mod compile;
pub mod instr;
mod parallel;
pub mod program;
mod region;

/// Portable schema for serialized [`Bytecode`] payloads.
pub const BYTECODE_SCHEMA_VERSION: u16 = 4;
/// Portable schema for serialized [`FunctionRegistry`] payloads.
pub const FUNCTION_REGISTRY_SCHEMA_VERSION: u16 = 4;

pub use compile::{compile, compile_semantic_function_registry};
pub use instr::{
    BytecodeCollectiveOp, BytecodeDistributedOp, BytecodeSpmdHeader, EmitLabel, Instr, StackEffect,
};
pub use parallel::{BytecodeParallelVariable, BytecodeParforRegion, BytecodeSpmdRegion};
pub use program::{
    AsyncMetadata, AwaitSite, Bytecode, FunctionBytecode, FunctionRegistry, SpawnSite,
};
#[cfg(feature = "native-accel")]
pub use program::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
pub use region::{BytecodeRegion, BytecodeRegionBoundary};
