mod artifact;
pub mod compile;
pub mod instr;
mod parallel;
pub mod program;
mod region;
mod sequence_register;
mod subscript;

/// Portable schema for serialized [`Bytecode`] payloads.
pub const BYTECODE_SCHEMA_VERSION: u16 = 7;
/// Portable schema for serialized [`FunctionRegistry`] payloads.
pub const FUNCTION_REGISTRY_SCHEMA_VERSION: u16 = 5;

pub use artifact::{
    decode_interpreter_program_v2, decode_interpreter_script_v2, encode_interpreter_program_v2,
    encode_interpreter_script_v2, InterpreterPayloadError, InterpreterPayloadForm,
    InterpreterRevisionField,
};
pub use compile::{
    compile, compile_semantic_function_registry, compile_semantic_function_registry_with_analysis,
    compile_with_analysis,
};
pub use instr::{
    AggregateElementSpec, BytecodeCodistributedOverload, BytecodeCollectiveOp,
    BytecodeDistributedBuildValidation, BytecodeDistributedOp, BytecodeSpmdHeader, EmitLabel,
    Instr, StackEffect,
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
pub use sequence_register::validate_sequence_register_flow;
pub use subscript::{BytecodeSubscriptSelector, BytecodeSubscriptStep};
