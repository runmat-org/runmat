pub mod compile;
pub mod instr;
pub mod program;

pub use compile::{compile, compile_semantic_function_registry};
pub use instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
pub use program::{
    Bytecode, SemanticAsyncMetadata, SemanticFunctionBytecode, SemanticFunctionRegistry,
    SemanticSpawnSite,
};
#[cfg(feature = "native-accel")]
pub use program::{SemanticFusionCandidateGroup, SemanticFusionMetadata};
