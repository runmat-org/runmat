#[cfg(feature = "native-accel")]
use crate::accel::graph::build_accel_graph;
#[cfg(feature = "native-accel")]
use crate::accel::stack_layout::annotate_fusion_groups_with_stack_layout;
use crate::bytecode::program::SemanticFunctionBytecode;
use crate::bytecode::{Bytecode, SemanticFunctionRegistry};
use crate::compiler::{CompileError, Compiler};
use crate::layout::derive_layout;
#[cfg(feature = "native-accel")]
use runmat_builtins::BuiltinSemanticKind;
use runmat_hir::{EntrypointId, FunctionId, HirAssembly};
use runmat_mir::MirAssembly;
use runmat_mir::{MirRvalue, MirStmtKind};
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
    let semantic_functions =
        compile_semantic_functions(hir, mir, c.layout.as_ref().unwrap(), Some(entrypoint))?;
    let semantic_function_registry = SemanticFunctionRegistry::new(semantic_functions.clone());
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
    let entrypoint_target = hir
        .entrypoints
        .iter()
        .find(|candidate| candidate.id == entrypoint)
        .map(|candidate| candidate.target);
    #[cfg(feature = "native-accel")]
    let semantic_fusion_metadata = derive_semantic_fusion_metadata(mir, entrypoint_target);
    #[cfg(feature = "native-accel")]
    let (accel_graph, fusion_groups) =
        if semantic_fusion_metadata.mir_fusion_candidate_group_count == 0 {
            (None, Vec::new())
        } else {
            let accel_graph = build_accel_graph(&c.instructions, &c.var_types);
            let mut fusion_groups = accel_graph.detect_fusion_groups();
            if !fusion_groups.is_empty() {
                annotate_fusion_groups_with_stack_layout(
                    &c.instructions,
                    &accel_graph,
                    &mut fusion_groups,
                );
            }
            (Some(accel_graph), fusion_groups)
        };
    let semantic_async_metadata = derive_semantic_async_metadata(mir);

    Ok(Bytecode {
        instructions: c.instructions,
        instr_spans: c.instr_spans,
        call_arg_spans: c.call_arg_spans,
        source_id: None,
        var_count: c.var_count,
        semantic_functions,
        semantic_function_registry,
        var_types: c.var_types,
        var_names,
        layout: c.layout,
        semantic_async_metadata,
        #[cfg(feature = "native-accel")]
        accel_graph,
        #[cfg(feature = "native-accel")]
        fusion_groups,
        #[cfg(feature = "native-accel")]
        semantic_fusion_metadata,
    })
}

fn derive_semantic_async_metadata(mir: &MirAssembly) -> crate::bytecode::SemanticAsyncMetadata {
    let mut sites = Vec::new();
    let mut function_ids: Vec<_> = mir.bodies.keys().copied().collect();
    function_ids.sort_by_key(|id| id.0);
    for function_id in function_ids {
        let Some(body) = mir.bodies.get(&function_id) else {
            continue;
        };
        for block in &body.blocks {
            for (stmt_index, stmt) in block.statements.iter().enumerate() {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => continue,
                };
                if matches!(value, MirRvalue::Spawn(_)) {
                    sites.push(crate::bytecode::SemanticSpawnSite {
                        function: body.function,
                        block: block.id,
                        stmt_index,
                    });
                }
            }
        }
    }
    crate::bytecode::SemanticAsyncMetadata {
        mir_spawn_site_count: sites.len(),
        mir_spawn_sites: sites,
    }
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_metadata(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> crate::bytecode::SemanticFusionMetadata {
    let (mir_fusion_signal_count, mir_fusion_candidate_groups) =
        derive_semantic_fusion_candidate_groups(mir, entrypoint_target);
    crate::bytecode::SemanticFusionMetadata {
        mir_fusion_signal_count,
        mir_fusion_candidate_group_count: mir_fusion_candidate_groups.len(),
        mir_fusion_candidate_groups,
    }
}

#[cfg(feature = "native-accel")]
fn derive_semantic_fusion_candidate_groups(
    mir: &MirAssembly,
    entrypoint_target: Option<FunctionId>,
) -> (usize, Vec<crate::bytecode::SemanticFusionCandidateGroup>) {
    let mut signal_count = 0usize;
    let mut groups = Vec::new();
    let mut function_ids: Vec<_> = if let Some(function) = entrypoint_target {
        vec![function]
    } else {
        mir.bodies.keys().copied().collect()
    };
    function_ids.sort_by_key(|id| id.0);
    for function_id in function_ids {
        let Some(body) = mir.bodies.get(&function_id) else {
            continue;
        };
        for block in &body.blocks {
            let mut run_len = 0usize;
            for (stmt_index, stmt) in block.statements.iter().enumerate() {
                let value = match &stmt.kind {
                    MirStmtKind::Assign { value, .. }
                    | MirStmtKind::MultiAssign { value, .. }
                    | MirStmtKind::Expr(value) => value,
                    MirStmtKind::PlaceMutation(_)
                    | MirStmtKind::WorkspaceEffect { .. }
                    | MirStmtKind::EnvironmentEffect(_) => {
                        if run_len >= 2 {
                            groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                                id: groups.len(),
                                signal_count: run_len,
                                function: body.function,
                                block: block.id,
                                stmt_start: stmt_index - run_len,
                                stmt_end: stmt_index,
                            });
                        }
                        run_len = 0;
                        continue;
                    }
                };
                if rvalue_has_fusion_signal(value) {
                    signal_count += 1;
                    run_len += 1;
                } else {
                    if run_len >= 2 {
                        groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                            id: groups.len(),
                            signal_count: run_len,
                            function: body.function,
                            block: block.id,
                            stmt_start: stmt_index - run_len,
                            stmt_end: stmt_index,
                        });
                    }
                    run_len = 0;
                }
            }
            if run_len >= 2 {
                groups.push(crate::bytecode::SemanticFusionCandidateGroup {
                    id: groups.len(),
                    signal_count: run_len,
                    function: body.function,
                    block: block.id,
                    stmt_start: block.statements.len() - run_len,
                    stmt_end: block.statements.len(),
                });
            }
        }
    }
    (signal_count, groups)
}

#[cfg(feature = "native-accel")]
fn rvalue_has_fusion_signal(value: &MirRvalue) -> bool {
    match value {
        MirRvalue::Unary(_, _) | MirRvalue::Binary(_, _, _) => true,
        MirRvalue::Call(call) => matches!(
            call.semantic_kind,
            BuiltinSemanticKind::Elementwise
                | BuiltinSemanticKind::Reduction
                | BuiltinSemanticKind::LinearAlgebra
                | BuiltinSemanticKind::ShapeTransform(_)
        ),
        MirRvalue::Use(_)
        | MirRvalue::Range { .. }
        | MirRvalue::Aggregate { .. }
        | MirRvalue::Index { .. }
        | MirRvalue::Member { .. }
        | MirRvalue::DynamicMember { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End
        | MirRvalue::Future { .. }
        | MirRvalue::Spawn(_) => false,
    }
}

pub fn compile_semantic_function_registry(
    hir: &HirAssembly,
    mir: &MirAssembly,
) -> Result<HashMap<FunctionId, SemanticFunctionBytecode>, CompileError> {
    let layout = derive_layout(hir, mir)
        .map_err(|err| CompileError::new(format!("failed to derive VM layout: {err:?}")))?;
    compile_semantic_functions(hir, mir, &layout, None)
}

fn compile_semantic_functions(
    hir: &HirAssembly,
    mir: &MirAssembly,
    layout: &crate::layout::VmAssemblyLayout,
    entrypoint: Option<EntrypointId>,
) -> Result<HashMap<FunctionId, SemanticFunctionBytecode>, CompileError> {
    let entry_target = entrypoint
        .and_then(|entrypoint| layout.entrypoints.get(&entrypoint))
        .map(|entry| entry.target);
    let mut functions = HashMap::new();
    for function in &hir.functions {
        if Some(function.id) == entry_target {
            continue;
        }
        let mut compiler = Compiler::new_for_function(hir, mir, layout.clone(), function.id)?;
        compiler.compile()?;
        let function_layout = layout.functions.get(&function.id).ok_or_else(|| {
            CompileError::new(format!("missing VM layout for function {:?}", function.id))
        })?;
        functions.insert(
            function.id,
            SemanticFunctionBytecode {
                function: function.id,
                display_name: function_layout.display_name.clone(),
                source_id: None,
                instructions: compiler.instructions,
                instr_spans: compiler.instr_spans,
                call_arg_spans: compiler.call_arg_spans,
                var_count: compiler.var_count,
                input_slots: function_layout
                    .frame_abi
                    .fixed_inputs
                    .iter()
                    .filter(|slot| Some(**slot) != function_layout.frame_abi.varargin)
                    .map(|slot| slot.0)
                    .collect(),
                varargin_slot: function_layout.frame_abi.varargin.map(|slot| slot.0),
                output_slots: function_layout
                    .frame_abi
                    .fixed_outputs
                    .iter()
                    .filter(|slot| Some(**slot) != function_layout.frame_abi.varargout)
                    .map(|slot| slot.0)
                    .collect(),
                varargout_slot: function_layout.frame_abi.varargout.map(|slot| slot.0),
                capture_slots: function_layout
                    .captures
                    .iter()
                    .map(|capture| capture.slot.0)
                    .collect(),
            },
        );
    }
    Ok(functions)
}

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::Instr;
    use futures::executor::block_on;
    use runmat_builtins::Value;
    use runmat_hir::{lower, CallableFallbackPolicy, FunctionId, LoweringContext};
    use runmat_mir::lowering::lower_assembly;
    use runmat_mir::MirTerminatorKind;
    use std::collections::HashMap;
    use std::sync::Arc;

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

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_records_semantic_fusion_metadata() {
        let ast = runmat_parser::parse("x = 1 + 2; y = x * 3;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count > 0,
            "expected non-zero MIR fusion signal count"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count
                > 0,
            "expected non-zero MIR fusion candidate group count"
        );
        assert!(
            !bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups
                .is_empty(),
            "expected non-empty MIR fusion candidate groups"
        );
        assert!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_groups
                .iter()
                .all(|group| group.stmt_end > group.stmt_start),
            "expected candidate groups to carry non-empty statement spans"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_semantically_gates_bytecode_fusion_groups() {
        let ast = runmat_parser::parse("x = 1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "expected no semantic fusion candidate groups"
        );
        assert!(
            bytecode.accel_graph.is_none(),
            "expected accel graph to be omitted when semantic candidate groups are absent"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "expected bytecode fusion groups to be gated off when semantic candidates are absent"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group() {
        let ast = runmat_parser::parse("x = 1 + 2;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count > 0,
            "expected non-zero fusion signal count for arithmetic operation"
        );
        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "expected no semantic candidate groups for a single-operation run"
        );
        assert!(
            bytecode.accel_graph.is_none(),
            "expected accel graph omission to follow semantic candidate-group gating"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "expected no executable bytecode fusion groups without semantic candidates"
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target() {
        let source = "x = 1; function z = helper(a); t = a + 1; z = t * 2; end;";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert_eq!(
            bytecode.semantic_fusion_metadata.mir_fusion_signal_count, 0,
            "non-entrypoint helper MIR bodies should not drive entrypoint fusion signal metadata"
        );
        assert_eq!(
            bytecode
                .semantic_fusion_metadata
                .mir_fusion_candidate_group_count,
            0,
            "non-entrypoint helper MIR bodies should not drive entrypoint fusion candidate metadata"
        );
        assert!(
            bytecode.accel_graph.is_none(),
            "entrypoint with no semantic candidates should omit accel graph even if helper bodies are fusible"
        );
        assert!(
            bytecode.fusion_groups.is_empty(),
            "entrypoint with no semantic candidates should not emit executable fusion groups"
        );
    }

    #[test]
    fn primary_compile_records_semantic_spawn_site_metadata() {
        let ast = runmat_parser::parse("fut = make(); task = spawn(fut);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(
            bytecode.semantic_async_metadata.mir_spawn_site_count > 0,
            "expected non-zero spawn site count"
        );
        assert!(
            !bytecode.semantic_async_metadata.mir_spawn_sites.is_empty(),
            "expected spawn site metadata entries"
        );
        assert_eq!(
            bytecode.semantic_async_metadata.mir_spawn_site_count,
            bytecode.semantic_async_metadata.mir_spawn_sites.len(),
            "spawn site count should match listed sites"
        );
        let unique_sites = bytecode
            .semantic_async_metadata
            .mir_spawn_sites
            .iter()
            .map(|site| (site.function, site.block, site.stmt_index))
            .collect::<std::collections::HashSet<_>>();
        assert!(
            unique_sites.len() == bytecode.semantic_async_metadata.mir_spawn_sites.len(),
            "spawn site metadata entries should be distinct"
        );
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
            [
                Instr::LoadConst(9.0),
                Instr::CallBuiltinMulti(name, 1, 1),
                Instr::StoreVar(_),
            ] if name == "sqrt"
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
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::IndexSlice(2, 2, 0, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[y_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_interprets_simple_colon_slice() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; y = x(:, 2);").expect("parse");
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
            .any(|instr| matches!(instr, Instr::IndexSlice(2, 1, 1, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[y_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![2.0, 4.0]);
    }

    #[test]
    fn primary_compile_lowers_ambiguous_local_index_to_slice() {
        let ast =
            runmat_parser::parse("x = [10 20 30 40]; a = find([0 1 1]); idx = a + 1; y = x(idx);")
                .expect("parse");
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
            .any(|instr| matches!(instr, Instr::IndexSlice(1, _, _, _))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Index(1))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[y_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![30.0, 40.0]);
    }

    #[test]
    fn primary_compile_lowers_ambiguous_local_store_index_to_slice() {
        let ast =
            runmat_parser::parse("x = [10 20 30 40]; idx = [2 4]; x(idx) = [9 8];").expect("parse");
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
            .any(|instr| matches!(instr, Instr::StoreSlice(1, _, _, _))));
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreIndex(1))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(tensor.data, vec![10.0, 9.0, 30.0, 8.0]);
    }

    #[test]
    fn primary_compile_lowers_statement_semantic_call_to_zero_outputs() {
        let ast =
            runmat_parser::parse("function y = f(x); y = nargout(); end; f(10);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::CallSemanticFunctionMulti(_, _, 0))));
    }

    #[test]
    fn primary_compile_lowers_method_calls_with_explicit_object_dispatch_policy() {
        let ast = runmat_parser::parse("obj = 1; obj.method(1);").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CallMethodOrMemberIndexMulti {
                fallback_policy: CallableFallbackPolicy::ObjectDispatch,
                ..
            }
        )));
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
        assert!(!bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::StoreSlice(2, 2, 0, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 9.0, 4.0]);
    }

    #[test]
    fn primary_compile_interprets_simple_slice_assignment() {
        let ast = runmat_parser::parse("x = [1 2; 3 4]; x(:, 2) = [9; 8];").expect("parse");
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
            .any(|instr| matches!(instr, Instr::StoreSlice(2, 1, 1, 0))));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        let Value::Tensor(tensor) = &vars[x_export.slot.0] else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 9.0, 8.0]);
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

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_interprets_simple_cell_indexed_assignment() {
        let ast = runmat_parser::parse("c = {1, 2}; c{2} = 9; x = c{2};").expect("parse");
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
            .any(|instr| matches!(instr, Instr::StoreIndexCell { num_indices: 1, .. })));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_selector_metadata_for_reads() {
        let ast = runmat_parser::parse("c = {1, 2, 3}; x = c{end};").expect("parse");
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

        assert!(bytecode.instructions.iter().any(|instr| {
            matches!(
                instr,
                Instr::IndexCell {
                    num_indices: 1,
                    end_offsets
                } if end_offsets == &vec![(0, 0)]
            ) || matches!(
                instr,
                Instr::IndexCellList {
                    num_indices: 1,
                    end_offsets
                } if end_offsets == &vec![(0, 0)]
            )
        }));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(3.0));
    }

    #[test]
    fn primary_compile_carries_cell_end_selector_metadata_for_stores() {
        let ast = runmat_parser::parse("c = {1, 2, 3}; c{end} = 9; x = c{3};").expect("parse");
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

        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::StoreIndexCell {
                num_indices: 1,
                end_offsets
            } if end_offsets == &vec![(0, 0)]
        )));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(9.0));
    }

    #[test]
    fn primary_compile_supports_mixed_cell_colon_expansion() {
        let ast = runmat_parser::parse("c = {1,2;3,4}; [a,b] = c{:,2}; z = a + b;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");

        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::IndexCellExpand {
                num_indices,
                out_count,
                ..
            } if *num_indices == 2 && *out_count == 2
        )));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(6.0));
    }

    #[test]
    fn primary_compile_3d_slice_roundtrip_uses_slice_expr_paths() {
        let ast = runmat_parser::parse(
            r#"
            A = reshape([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24], 3, 4, 2);
            S = A(1:2, 2:3, end);
            A(1:2, 2:3, end) = S;
        "#,
        )
        .expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let mut saw_index_expr = false;
        let mut saw_store_expr = false;
        for instr in &bytecode.instructions {
            if let Instr::IndexSliceExpr {
                dims,
                end_mask,
                end_numeric_exprs,
                range_dims,
                ..
            } = instr
            {
                if *dims == 3 {
                    saw_index_expr = true;
                    assert_eq!(*end_mask, 0);
                    assert_eq!(end_numeric_exprs.len(), 1);
                    assert_eq!(range_dims, &vec![0, 1]);
                }
            }
            if let Instr::StoreSliceExpr {
                dims,
                numeric_count,
                colon_mask,
                end_mask,
                range_dims,
                range_has_step,
                end_numeric_exprs,
                ..
            } = instr
            {
                if *dims == 3 {
                    saw_store_expr = true;
                    assert_eq!(*numeric_count, 1);
                    assert_eq!(*colon_mask, 0);
                    assert_eq!(*end_mask, 0);
                    assert_eq!(range_dims, &vec![0, 1]);
                    assert_eq!(range_has_step, &vec![false, false]);
                    assert_eq!(end_numeric_exprs.len(), 1);
                }
            }
        }
        assert!(saw_index_expr);
        assert!(saw_store_expr);

        let run = block_on(crate::interpret(&bytecode));
        assert!(
            run.is_ok(),
            "roundtrip script should interpret successfully: {run:?}"
        );
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

    #[test]
    fn primary_compile_interprets_basic_switch_statement() {
        let ast =
            runmat_parser::parse("switch 2; case 1; x = 1; case 2; x = 2; otherwise; x = 3; end")
                .expect("parse");
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
            .any(|instr| matches!(instr, Instr::Equal)));

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[x_export.slot.0], Value::Num(2.0));
    }

    #[test]
    fn primary_compile_lowers_unreachable_terminator() {
        let ast = runmat_parser::parse("x = 1;").expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mut mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let function = hir.assembly.entrypoints[0].target;
        let body = mir.bodies.get_mut(&function).expect("entry body");
        body.blocks.last_mut().expect("entry block").terminator.kind =
            MirTerminatorKind::Unreachable;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");

        assert!(bytecode
            .instructions
            .iter()
            .any(|instr| matches!(instr, Instr::Return)));
    }

    #[test]
    fn primary_compile_external_semantic_function_handle_keeps_identity() {
        let ast = runmat_parser::parse("h = @remote_inc; y = feval(h, 2);").expect("parse");
        let mut semantic_functions = HashMap::new();
        semantic_functions.insert("remote_inc".to_string(), FunctionId(9001));
        let context = LoweringContext::empty().with_semantic_functions(&semantic_functions);
        let hir = lower(&ast, &context).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CreateSemanticFunctionHandle(FunctionId(9001), _)
        )));

        let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(
            Some(Arc::new(|name| {
                if name == "remote_inc" {
                    Some(9001)
                } else {
                    None
                }
            })),
        );
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9001);
                assert_eq!(args, &[Value::Num(2.0)]);
                assert_eq!(requested_outputs, 1);
                Box::pin(async move { Ok(Value::Num(3.0)) })
            })),
        );

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert!(vars
            .iter()
            .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
    }

    #[test]
    fn primary_compile_external_semantic_direct_call_uses_host_invoker() {
        let ast = runmat_parser::parse("y = remote_inc(2);").expect("parse");
        let mut semantic_functions = HashMap::new();
        semantic_functions.insert("remote_inc".to_string(), FunctionId(9001));
        let context = LoweringContext::empty().with_semantic_functions(&semantic_functions);
        let hir = lower(&ast, &context).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        assert!(bytecode.instructions.iter().any(|instr| matches!(
            instr,
            Instr::CallSemanticFunctionMulti(FunctionId(9001), 1, 1)
        )));

        let _resolver_guard = runmat_runtime::user_functions::install_semantic_function_resolver(
            Some(Arc::new(|name| {
                if name == "remote_inc" {
                    Some(9001)
                } else {
                    None
                }
            })),
        );
        let _invoker_guard = runmat_runtime::user_functions::install_semantic_function_invoker(
            Some(Arc::new(|function, args, requested_outputs| {
                assert_eq!(function, 9001);
                assert_eq!(args, &[Value::Num(2.0)]);
                assert_eq!(requested_outputs, 1);
                Box::pin(async move { Ok(Value::Num(3.0)) })
            })),
        );

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert!(vars
            .iter()
            .any(|value| matches!(value, Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
    }

    #[test]
    fn primary_compile_interprets_async_call_and_await_via_semantic_value_lane() {
        let source = "async function y = inc(x); y = x + 1; end; t = inc(2); z = await(t);";
        let ast = runmat_parser::parse(source).expect("parse");
        let hir = lower(&ast, &LoweringContext::empty()).expect("lower HIR");
        let mir = lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;

        let bytecode = compile(&hir.assembly, &mir, entrypoint).expect("compile");
        let layout = bytecode.layout.as_ref().expect("layout");
        let z_export = layout.entrypoints[&entrypoint]
            .exports
            .iter()
            .find(|export| export.name == "z")
            .expect("z export");

        let vars = block_on(crate::interpret(&bytecode)).expect("interpret");
        assert_eq!(vars[z_export.slot.0], Value::Num(3.0));
    }
}
