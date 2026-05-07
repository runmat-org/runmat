#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
use crate::bytecode::{Bytecode, UserFunction};
use crate::compiler::{CompileError, Compiler};
use crate::layout::derive_layout;
use runmat_hir::{EntrypointId, HirAssembly, LegacyHirProgram as HirProgram};
use runmat_mir::MirAssembly;
use std::collections::HashMap;

pub fn compile(
    hir: &HirAssembly,
    mir: &MirAssembly,
    entrypoint: EntrypointId,
) -> Result<Bytecode, CompileError> {
    let layout = derive_layout(hir, mir)
        .map_err(|err| CompileError::new(format!("failed to derive VM layout: {err:?}")))?;
    let entrypoint_layout = layout.entrypoints.get(&entrypoint).ok_or_else(|| {
        CompileError::new(format!("missing VM layout for entrypoint {entrypoint:?}"))
    })?;
    let function_layout = layout
        .functions
        .get(&entrypoint_layout.target)
        .ok_or_else(|| {
            CompileError::new(format!(
                "missing VM layout for entrypoint target {:?}",
                entrypoint_layout.target
            ))
        })?;

    Ok(Bytecode {
        instructions: Vec::new(),
        instr_spans: Vec::new(),
        call_arg_spans: Vec::new(),
        source_id: None,
        var_count: function_layout.local_count,
        functions: HashMap::new(),
        var_types: Vec::new(),
        var_names: HashMap::new(),
        layout: Some(layout),
        #[cfg(feature = "native-accel")]
        accel_graph: None,
        #[cfg(feature = "native-accel")]
        fusion_groups: Vec::new(),
    })
}

pub fn compile_legacy(
    prog: &HirProgram,
    existing_functions: &HashMap<String, UserFunction>,
) -> Result<Bytecode, CompileError> {
    let mut c = Compiler::new(prog);
    c.functions = existing_functions.clone();
    c.compile_program(prog)?;
    #[cfg(feature = "native-accel")]
    let accel_graph = build_accel_graph(&c.instructions, &c.var_types);
    #[cfg(feature = "native-accel")]
    let mut fusion_groups = accel_graph.detect_fusion_groups();
    #[cfg(feature = "native-accel")]
    annotate_fusion_groups_with_stack_layout(&c.instructions, &accel_graph, &mut fusion_groups);
    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        call_arg_spans: c.call_arg_spans,
        source_id: None,
        var_count: c.var_count,
        functions: c.functions,
        var_types: c.var_types,
        var_names: HashMap::new(),
        layout: None,
        #[cfg(feature = "native-accel")]
        accel_graph: Some(accel_graph),
        #[cfg(feature = "native-accel")]
        fusion_groups,
    })
}
