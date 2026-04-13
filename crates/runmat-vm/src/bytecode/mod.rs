pub mod instr;
pub mod program;

pub use instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
pub use program::{Bytecode, CallFrame, ExecutionContext, UserFunction};
