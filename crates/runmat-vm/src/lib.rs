#![allow(clippy::result_large_err)]

pub mod accel;
pub mod bytecode;
pub mod call;
pub mod compiler;
pub mod functions {
    pub use crate::bytecode::program::{Bytecode, CallFrame, ExecutionContext, UserFunction};
}
pub mod indexing;
pub mod instr {
    pub use crate::bytecode::instr::{ArgSpec, EmitLabel, EndExpr, Instr, StackEffect};
}
pub mod interpreter;
pub mod object;
pub mod ops;
pub mod runtime;

pub use bytecode::compile;
pub use bytecode::{
    ArgSpec, Bytecode, CallFrame, EmitLabel, EndExpr, ExecutionContext, Instr, StackEffect,
    UserFunction,
};
pub use compiler::CompileError;
