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
    let mut c = Compiler::new(hir, mir, layout, entrypoint)?;
    c.compile()?;
    let var_names = c
        .layout
        .as_ref()
        .and_then(|layout| layout.entrypoints.get(&entrypoint))
        .map(|entrypoint_layout| {
            entrypoint_layout
                .exports
                .iter()
                .map(|export| (export.slot.0, export.name.clone()))
                .collect()
        })
        .unwrap_or_default();

    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        call_arg_spans: c.call_arg_spans,
        source_id: None,
        var_count: c.var_count,
        functions: HashMap::new(),
        var_types: c.var_types,
        var_names,
        layout: c.layout,
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
    let mut c = Compiler::new_legacy(prog);
    c.functions = existing_functions.clone();
    c.compile_program_legacy(prog)?;
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

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::Instr;
    use futures::executor::block_on;
    use runmat_builtins::Value;
    use runmat_hir::{lower, LoweringContext};
    use runmat_mir::lowering::lower_assembly;

    #[test]
    fn primary_compile_attaches_derived_layout() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        let layout = bytecode.layout.as_ref().expect("layout");
        let entrypoint_layout = &layout.entrypoints[&entrypoint];
        let function_layout = &layout.functions[&entrypoint_layout.target];
        assert_eq!(bytecode.var_count, function_layout.local_count);
        assert_eq!(bytecode.var_types.len(), function_layout.local_count);
    }

    #[test]
    fn primary_compile_lowers_simple_assignment_arithmetic() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(bytecode.instructions.len(), 4);
        assert!(matches!(bytecode.instructions[0], Instr::LoadConst(1.0)));
        assert!(matches!(bytecode.instructions[1], Instr::LoadConst(2.0)));
        assert!(matches!(bytecode.instructions[2], Instr::Add));
        assert!(matches!(bytecode.instructions[3], Instr::StoreVar(_)));
    }

    #[test]
    fn primary_compile_interprets_visible_assignment() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert_eq!(bytecode.var_names[&export.slot.0], "x");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_builtin_assignment() {
        let ast = runmat_parser::parse("x = sqrt(9);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert!(matches!(
            bytecode.instructions.as_slice(),
            [Instr::LoadConst(9.0), Instr::CallBuiltin(name, 1), Instr::StoreVar(_)] if name == "sqrt"
        ));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_matrix_literal_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4];").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let export = &layout.entrypoints[&entrypoint].exports[0];

        assert!(matches!(
            bytecode.instructions.as_slice(),
            [
                Instr::LoadConst(1.0),
                Instr::LoadConst(2.0),
                Instr::LoadConst(3.0),
                Instr::LoadConst(4.0),
                Instr::CreateMatrix(2, 2),
                Instr::StoreVar(_),
            ]
        ));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_matrix_indexing() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; y = x(2, 1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let y_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "y")
            .expect("y export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Index(2))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_simple_indexed_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; x(1, 2) = 9;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreIndex(2))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 9.0, 4.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_cell_indexing() {
        let ast = runmat_parser::parse("c = {1, 2}; x = c{2};").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::IndexCell(1))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_interprets_basic_if_statement() {
        let ast = runmat_parser::parse("if 1; x = 2; else; x = 3; end").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let x_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "x")
            .expect("x export");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::JumpIfFalse(_))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }
}
