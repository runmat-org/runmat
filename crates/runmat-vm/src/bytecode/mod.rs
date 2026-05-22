pub mod compile;
pub mod instr;
pub mod program;

pub use compile::{compile, compile_semantic_function_registry};
pub use instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
pub use program::{
    AsyncMetadata, AwaitSite, Bytecode, FunctionBytecode, FunctionRegistry, SpawnSite,
};
#[cfg(feature = "native-accel")]
pub use program::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
};
