use crate::compiler::Compiler;
use crate::functions::{Bytecode, UserFunction};
use runmat_hir::HirProgram;
use std::collections::HashMap;

pub fn compile(prog: &HirProgram) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.compile_program(prog)?;
    Ok(Bytecode {
        instructions: c.instructions,
        var_count: c.var_count,
        functions: c.functions,
        var_types: c.var_types,
    })
}

pub fn compile_with_functions(
    prog: &HirProgram,
    existing_functions: &HashMap<String, UserFunction>,
) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.functions = existing_functions.clone();
    c.compile_program(prog)?;
    Ok(Bytecode {
        instructions: c.instructions,
        var_count: c.var_count,
        functions: c.functions,
        var_types: c.var_types,
    })
}
