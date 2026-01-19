#[cfg(feature = "native-accel")]
use crate::accel_graph::build_accel_graph;
use crate::compiler::Compiler;
use crate::functions::{Bytecode, UserFunction};
use runmat_hir::HirProgram;
use std::collections::HashMap;

pub fn compile(prog: &HirProgram) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.compile_program(prog)?;
    #[cfg(feature = "native-accel")]
    let accel_graph = build_accel_graph(&c.instructions, &c.var_types);
    #[cfg(feature = "native-accel")]
    let fusion_groups = accel_graph.detect_fusion_groups();
    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        var_count: c.var_count,
        functions: c.functions,
        var_types: c.var_types,
        var_names: HashMap::new(),
        #[cfg(feature = "native-accel")]
        accel_graph: Some(accel_graph),
        #[cfg(feature = "native-accel")]
        fusion_groups,
    })
}

pub fn compile_with_functions(
    prog: &HirProgram,
    existing_functions: &HashMap<String, UserFunction>,
) -> Result<Bytecode, String> {
    let mut c = Compiler::new(prog);
    c.functions = existing_functions.clone();
    c.compile_program(prog)?;
    #[cfg(feature = "native-accel")]
    let accel_graph = build_accel_graph(&c.instructions, &c.var_types);
    #[cfg(feature = "native-accel")]
    let fusion_groups = accel_graph.detect_fusion_groups();
    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        var_count: c.var_count,
        functions: c.functions,
        var_types: c.var_types,
        var_names: HashMap::new(),
        #[cfg(feature = "native-accel")]
        accel_graph: Some(accel_graph),
        #[cfg(feature = "native-accel")]
        fusion_groups,
    })
}
