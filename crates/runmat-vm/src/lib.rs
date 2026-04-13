#![allow(clippy::result_large_err)]

pub mod accel;
pub mod bytecode;
pub mod call;
pub mod compiler;
pub mod indexing;
pub mod interpreter;
pub mod object;
pub mod ops;
pub mod runtime;

pub use bytecode::{
    ArgSpec, Bytecode, CallFrame, EmitLabel, EndExpr, ExecutionContext, Instr, StackEffect,
    UserFunction,
};
