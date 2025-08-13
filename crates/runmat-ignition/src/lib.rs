pub mod instr;
pub mod functions;
pub mod gc_roots;
pub mod compiler;
pub mod bytecode;
pub mod vm;

pub use bytecode::{compile, compile_with_functions};
pub use functions::{Bytecode, ExecutionContext, UserFunction};
pub use vm::{interpret, interpret_with_vars};
pub use instr::Instr;

use runmat_builtins::Value;
use runmat_hir::HirProgram;

pub fn execute(program: &HirProgram) -> Result<Vec<Value>, String> {
    let bc = compile(program)?;
    interpret(&bc)
}
