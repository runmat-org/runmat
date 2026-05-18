pub mod compile;
pub mod instr;
pub mod program;

pub use compile::{compile, compile_semantic_function_registry};
pub use instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
#[cfg(feature = "native-accel")]
pub use program::SemanticFusionMetadata;
pub use program::{Bytecode, SemanticFunctionBytecode, SemanticFunctionRegistry};
