use super::*;
use crate::bytecode::Instr;
use runmat_accelerate::fusion::{FusionGroupPlan, FusionOp};
use runmat_accelerate::graph::PrimitiveOp;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

fn assert_complete_external_dag_inputs(selected: &FusionGroupPlan) {
    let mut produced = HashMap::new();
    for operation in &selected.operations {
        let (inputs, output) = match operation {
            FusionOp::Primitive { inputs, output, .. }
            | FusionOp::Builtin { inputs, output, .. } => (inputs, output),
        };
        assert!(produced
            .insert(output.expect("one operation output"), inputs)
            .is_none());
    }
    assert_eq!(
        selected.inputs.len(),
        1,
        "one authentic external tensor leaf"
    );
    assert!(selected.inputs.iter().all(|id| !produced.contains_key(id)));
    let mut pending = vec![selected.reduction_data.expect("selected reduction root")];
    let mut seen = HashSet::new();
    while let Some(value) = pending.pop() {
        if selected.inputs.contains(&value) {
            continue;
        }
        if !seen.insert(value) {
            continue;
        }
        let inputs = produced.get(&value).expect("no hidden intermediate leaf");
        pending.extend(inputs.iter().copied());
    }
    assert_eq!(seen.len(), 3, "all three Add nodes feed the reduction root");
}

fn compiled(source: &str) -> Bytecode {
    let ast = runmat_parser::parse(source).expect("parse authentic source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).expect("bytecode")
}

fn pure_dag() -> Bytecode {
    compiled("x = gpuArray(single(ones(4, 4))); y = min((x + x) + (x + x)); z = gather(y);")
}

#[test]
fn private_chain_later_unassigned_and_missing_formal_reads_keep_ordinary_errors() {
    for (parameters, initialization, identifier) in [
        (
            "",
            "if false; missing = 1; end;",
            "RunMat:UndefinedVariable",
        ),
        ("missing", "", "RunMat:NotEnoughInputs"),
    ] {
        runmat_gc::gc_test_context(|| {
            let bytecode = compiled(&format!(
                "caught = false; try; result = probe(); catch e; caught = strcmp(e.identifier, '{identifier}'); end; assert(caught);\nfunction value = probe({parameters})\n{initialization}\nvalue = (ones(1,3) + [1,2,3]) .* missing;\nend\n"
            ));
            futures::executor::block_on(crate::interpreter::runner::interpret(&bytecode))
                .expect("later semantic read must be caught, not replaced with a zero-filled slot");
        });
    }
}

#[test]
fn authentic_elementwise_private_dag_closes_without_materializing_private_stores() {
    for expression in ["(x + x) + (x + x)", "exp(-x) .* sin(4*x) + tanh(x).^2"] {
        let bytecode = compiled(&format!(
            "x = gpuArray(single(ones(4, 1))); y = {expression}; z = gather(y);"
        ));
        assert_elementwise_private_closure(&bytecode);
    }
}

fn assert_elementwise_private_closure(bytecode: &Bytecode) {
    let edges = bridges(bytecode);
    let output_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "y").then_some(*slot))
        .expect("source expression output binding");
    let (_, stores) = crate::accel::graph::build_accel_graph_with_stores(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    let output_store = stores
        .iter()
        .find(|store| store.binding.index == output_slot)
        .expect("source expression output store");
    let prior_publication = stores
        .iter()
        .filter(|store| {
            store.pc < output_store.pc && bytecode.var_names.contains_key(&store.binding.index)
        })
        .map(|store| store.pc)
        .max();
    let expression_edges = edges
        .iter()
        .filter(|edge| {
            prior_publication.is_none_or(|pc| pc < edge.producer_pc)
                && edge.consumer_pc < output_store.pc
        })
        .collect::<Vec<_>>();
    assert!(
        expression_edges.len() >= 2,
        "source-expression private branches are sealed: {edges:?}"
    );
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let plan = plan.expect("checked elementwise plan");
    let graph = graph.expect("selected instruction graph");
    let selected = plan
        .groups
        .iter()
        .find(|selected| {
            selected.group.kind == FusionKind::ElementwiseChain
                && expression_edges.iter().all(|edge| {
                    selected.group.span.start <= edge.producer_pc
                        && edge.consumer_pc <= selected.group.span.end
                })
        })
        .expect("one complete elementwise dependency closure");
    assert!(
        selected.kernel.supported,
        "complete private DAG retains fusion"
    );
    assert!(
        selected.materialized_stores.is_empty(),
        "private temporaries are not publication outputs"
    );
    assert!(
        selected.group.stack_layout.is_some(),
        "exact bytecode-derived stack contract"
    );
    for edge in expression_edges {
        let producer = graph
            .nodes
            .iter()
            .find(|node| node.span.end == edge.producer_pc)
            .unwrap();
        assert!(graph.node_binding(producer.id).is_none());
        assert!(graph.var_binding(producer.outputs[0]).is_none());
    }
    assert!(
        selected
            .inputs
            .iter()
            .all(|input| !selected.operations.iter().any(|operation| {
                match operation {
                    FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => {
                        *output == Some(*input)
                    }
                }
            })),
        "no hidden private intermediate input"
    );
}

#[test]
fn function_elementwise_private_dag_survives_portable_roundtrip() {
    let ast = runmat_parser::parse("function y = f(x)\ny = (x + x) + (x + x);\nend\n").unwrap();
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let function = hir.assembly.functions[0].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir).unwrap(),
    );
    assert_elementwise_private_closure(
        &registry
            .get(function)
            .unwrap()
            .execution_bytecode(&registry),
    );
    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry).unwrap();
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded).unwrap();
    assert_elementwise_private_closure(
        &decoded.get(function).unwrap().execution_bytecode(&decoded),
    );
}

#[test]
fn nonscalar_matrix_product_is_not_a_private_elementwise_bridge() {
    let bytecode = compiled("x = gpuArray(single(ones(4, 4))); y = sin(x*x); z = gather(y);");
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let matmul_pc = graph
        .as_ref()
        .and_then(|graph| {
            graph.nodes.iter().find(|node| {
                node.category == runmat_accelerate::graph::AccelOpCategory::MatMul
                    && matches!(
                        node.label,
                        runmat_accelerate::graph::AccelNodeLabel::Builtin {
                            origin: runmat_accelerate::graph::FusionCallOrigin::Operator(
                                runmat_types::OperatorKind::MatrixMultiply
                            ),
                            ..
                        }
                    )
            })
        })
        .expect("the non-scalar product retains its matrix operator")
        .span
        .start;
    if let Some(plan) = plan {
        assert!(plan.groups.iter().any(|selected| {
            selected.group.kind == FusionKind::MatmulEpilogue
                && selected.group.span.start <= matmul_pc
                && matmul_pc <= selected.group.span.end
        }));
        assert!(!plan.groups.iter().any(|selected| {
            selected.group.kind == FusionKind::ElementwiseChain
                && selected.group.span.start <= matmul_pc
                && matmul_pc <= selected.group.span.end
        }));
    }
}

#[test]
fn elementwise_closure_never_elides_observable_binding_stores() {
    let bytecode = compiled(
        "x = gpuArray(single(ones(4, 1))); a = x + x; y = (a + a) + (a + a); z = gather(y);",
    );
    let edges = bridges(&bytecode);
    let (plan, _) = bytecode.prepare_vm_fusion_plan();
    let plan = plan.expect("checked elementwise DAG");
    let (_, stores) = crate::accel::graph::build_accel_graph_with_stores(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    for selected in &plan.groups {
        if selected.group.kind != FusionKind::ElementwiseChain {
            continue;
        }
        for store in stores.iter().filter(|store| {
            selected.group.span.start <= store.pc && store.pc <= selected.group.span.end
        }) {
            if edges.iter().any(|edge| edge.store_pc == store.pc) {
                continue;
            }
            assert!(
                selected
                    .materialized_stores
                    .iter()
                    .any(|materialized| materialized.value_id == store.value_id
                        && materialized.binding == store.binding),
                "observable store cannot disappear inside chain"
            );
        }
    }
}

#[test]
fn canonical_elementwise_markers_cannot_authorize_added_statement_effects() {
    let bytecode = compiled("x = gpuArray(single(ones(4, 1))); y = exp(-x) + x; z = gather(y);");
    let edge = bridges(&bytecode).into_iter().find(|edge| {
        matches!(&bytecode.instructions[edge.producer_pc], Instr::CallBuiltinMulti(name, 1, 1) if name == "exp")
    }).expect("authentic canonical exp producer bridge");
    for effect in [
        runmat_types::EffectKind::WorkspaceWrite,
        runmat_types::EffectKind::HostCallback,
        runmat_types::EffectKind::Unknown,
        runmat_types::EffectKind::MaySuspend,
    ] {
        let mut changed = bytecode.clone();
        let projection =
            std::sync::Arc::make_mut(changed.fusion_metadata.stage_projection.as_mut().unwrap());
        projection
            .statements
            .iter_mut()
            .find(|statement| {
                statement.pc.start <= edge.producer_pc && edge.producer_pc < statement.pc.end
            })
            .unwrap()
            .contract
            .effects
            .0
            .insert(effect);
        changed.reseal_fusion_stage_after_owned_remap();
        assert!(changed.validate_fusion_stage_projection().is_ok());
        assert!(!bridges(&changed).iter().any(|candidate| candidate.slot == edge.slot),
            "neither external effects nor undeclared numerical markers may borrow canonical admission");
    }
}

fn function_dag() -> (crate::bytecode::FunctionRegistry, runmat_hir::FunctionId) {
    let ast = runmat_parser::parse("function y = f(x)\ny = min((x + x) + (x + x));\nend\n")
        .expect("parse authentic function source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    assert!(hir.assembly.entrypoints.is_empty());
    let function = hir.assembly.functions[0].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
            .expect("function bytecode"),
    );
    (registry, function)
}

#[test]
fn authentic_function_private_dag_survives_dispatch_and_program_v2_roundtrip() {
    let (registry, function) = function_dag();
    let body = registry
        .get(function)
        .expect("selected compiled definition");
    let executable = body.execution_bytecode(&registry);
    assert!(
        executable.layout.as_ref().is_some_and(
            |layout| layout.entrypoints.is_empty() && layout.functions.contains_key(&function)
        ),
        "ordinary function runner retains only the selected function layout"
    );
    let edges = bridges(&executable);
    assert_eq!(
        edges.len(),
        3,
        "three function-owned private edges: {edges:?}"
    );
    let (plan, graph) = executable.prepare_vm_fusion_plan();
    assert!(graph.is_some(), "real function instruction graph");
    let plan = plan.expect("checked function DAG reduction plan");
    let selected = plan
        .groups
        .iter()
        .find(|group| {
            group.group.kind == FusionKind::Reduction
                && group.group.nodes.len() == 4
                && group.materialized_stores.is_empty()
                && edges.iter().all(|edge| {
                    group.group.span.start <= edge.producer_pc
                        && edge.consumer_pc <= group.group.span.end
                })
        })
        .expect("complete function DAG reduction");
    assert_complete_external_dag_inputs(selected);

    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("portable function artifact");
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded)
        .expect("canonical portable definition");
    let decoded_body = decoded.get(function).expect("decoded selected definition");
    let decoded_execution = decoded_body.execution_bytecode(&decoded);
    assert!(decoded_execution.layout.as_ref().is_some_and(
        |layout| layout.entrypoints.is_empty() && layout.functions.contains_key(&function)
    ));
    assert_eq!(bridges(&decoded_execution).len(), edges.len());
    let (decoded_plan, decoded_graph) = decoded_execution.prepare_vm_fusion_plan();
    assert!(decoded_graph.is_some());
    let decoded_plan = decoded_plan.expect("decoded function DAG plan");
    let decoded = decoded_plan
        .groups
        .iter()
        .find(|group| {
            group.group.kind == FusionKind::Reduction
                && group.group.nodes.len() == 4
                && group.materialized_stores.is_empty()
        })
        .expect("complete decoded function DAG reduction");
    assert_complete_external_dag_inputs(decoded);

    let detached = body.clone();
    assert!(
        bridges(&detached.execution_bytecode(&registry)).is_empty(),
        "a same-ID detached function body cannot borrow selected registry admission"
    );
}

#[test]
fn authentic_compiler_private_dag_closes_into_one_checked_reduction() {
    let bytecode = pure_dag();
    let edges = bridges(&bytecode);
    assert!(edges.iter().all(|edge| {
        matches!(bytecode.instructions[edge.store_pc], Instr::StoreVar(slot) if slot == edge.slot)
            && matches!(bytecode.instructions[edge.load_pc], Instr::LoadVar(slot) if slot == edge.slot)
            && edge.producer_pc < edge.store_pc
            && edge.load_pc < edge.consumer_pc
    }));
    let groups = bytecode.runtime_fusion_groups();
    let reduction = groups
        .iter()
        .find(|group| group.kind == FusionKind::Reduction)
        .expect("one folded reduction window");
    let reduction_edges = edges
        .iter()
        .filter(|edge| {
            reduction.span.start <= edge.producer_pc && edge.consumer_pc <= reduction.span.end
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reduction_edges.len(),
        3,
        "three reduction-owned private edges: {edges:?}"
    );
    assert_eq!(
        groups
            .iter()
            .filter(|group| group.kind == FusionKind::Reduction)
            .count(),
        1,
    );
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let plan = plan.expect("checked multi-producer plan");
    let graph = graph.expect("same authentic instruction graph");
    let selected = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("reduction plan");
    assert_eq!(selected.group.span, reduction.span);
    assert_eq!(selected.group.nodes.len(), 4, "three Adds and one Minimum");
    assert!(selected.materialized_stores.is_empty());
    assert_complete_external_dag_inputs(selected);
    assert_eq!(
        selected
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    FusionOp::Primitive {
                        op: PrimitiveOp::Add,
                        ..
                    }
                )
            })
            .count(),
        3,
    );
    for edge in reduction_edges {
        let producer = graph
            .nodes
            .iter()
            .find(|node| node.span.end == edge.producer_pc)
            .expect("exact producer node");
        let [output] = producer.outputs.as_slice() else {
            panic!("one producer output");
        };
        assert!(graph.var_binding(*output).is_none());
        assert!(graph.node_binding(producer.id).is_none());
    }
}

fn assert_private_dag_uses_exact_vm_stack(bytecode: &Bytecode) {
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let plan = plan.expect("compiler-certified DAG plan");
    let graph = graph.expect("same instruction graph");
    let selected = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("one folded reduction");
    assert!(
        selected.kernel.supported,
        "complete reduction retains fusion"
    );
    assert_complete_external_dag_inputs(selected);
    let source_group = bytecode
        .runtime_fusion_groups_for_graph(&graph)
        .into_iter()
        .find(|group| group.id == selected.group.id && group.span == selected.group.span)
        .expect("same compiler-owned group");
    let source_layout = source_group
        .stack_layout
        .as_ref()
        .expect("instruction-derived VM stack layout");
    assert_eq!(
        selected.group.stack_layout.as_ref(),
        Some(source_layout),
        "bound graph leaves must not inflate or rebind the instruction-derived stack slice"
    );
    assert_eq!(
        source_layout.required_stack_operands, 2,
        "the first Add consumes exactly the two already-loaded operands"
    );

    let leaf = selected.inputs[0];
    let occurrences: Vec<_> = source_layout
        .bindings
        .iter()
        .filter(|binding| binding.value_id == leaf)
        .map(|binding| binding.stack_offset)
        .collect();
    assert_eq!(occurrences, vec![0, 1]);
    assert!(source_layout
        .bindings
        .iter()
        .all(|binding| binding.stack_offset < source_layout.required_stack_operands));

    // The ordinary VM has two live operands above an unrelated lower stack
    // value at the first producer PC. Gather must consume only those operands.
    let lower = Value::Num(-37.0);
    let mut stack = crate::interpreter::stack::InterpreterStack::from_unbounded(vec![
        lower.clone(),
        Value::Num(3.0),
        Value::Num(5.0),
    ]);
    let original = stack.as_slice().to_vec();
    let vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![
        Value::Num(11.0);
        bytecode.var_count
    ]);
    let mut context = crate::bytecode::program::ExecutionContext::default();
    let (_, stores, loads) = crate::accel::graph::build_accel_graph_with_transcript(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    let transport = crate::accel::fusion::input_transport::VmFusionInputTransport::project(
        selected, &graph, &loads, &stores,
    )
    .expect("every external input has a certified transport");
    let crate::accel::fusion::GatheredFusionInputs { guard, request } =
        crate::accel::fusion::gather_fusion_inputs(
            selected,
            &transport,
            &mut stack,
            &vars,
            &mut context,
            None,
        )
        .expect("gather has no terminal resource failure")
        .expect("the one external leaf has its exact evaluated stack source");
    assert_eq!(request.inputs, vec![Value::Num(3.0)]);
    drop(request);
    guard.commit();
    assert_eq!(stack.as_slice(), std::slice::from_ref(&lower));

    let mut too_shallow =
        crate::interpreter::stack::InterpreterStack::from_unbounded(vec![lower.clone()]);
    assert!(crate::accel::fusion::gather_fusion_inputs(
        selected,
        &transport,
        &mut too_shallow,
        &vars,
        &mut context,
        None,
    )
    .expect("insufficient operands are an admission decline, not a terminal failure")
    .is_none());
    assert_eq!(too_shallow.as_slice(), std::slice::from_ref(&lower));

    // A malformed plan that asks for an internal private result must decline
    // before execution, preserving the exact original stack slice.
    let internal = graph
        .nodes
        .iter()
        .find(|node| selected.group.nodes.contains(&node.id) && !node.is_reduction())
        .and_then(|node| node.outputs.first())
        .copied()
        .expect("private producer output");
    let mut malformed = selected.clone();
    malformed.inputs.push(internal);
    stack = crate::interpreter::stack::InterpreterStack::from_unbounded(original.clone());
    let gathered = crate::accel::fusion::gather_fusion_inputs(
        &malformed,
        &transport,
        &mut stack,
        &vars,
        &mut context,
        None,
    )
    .expect("malformed transport declines without a terminal resource failure");
    assert!(
        gathered.is_none(),
        "internal output has no independent source"
    );
    drop(gathered);
    assert_eq!(stack.as_slice(), original.as_slice());
}

#[test]
fn entrypoint_private_dag_preserves_exact_stack_inputs_and_transaction() {
    assert_private_dag_uses_exact_vm_stack(&pure_dag());
}

#[test]
fn ordinary_function_private_dag_preserves_exact_stack_inputs_and_transaction() {
    let (registry, function) = function_dag();
    let executable = registry
        .get(function)
        .expect("selected function")
        .execution_bytecode(&registry);
    assert!(
        executable.layout.as_ref().is_some_and(
            |layout| layout.entrypoints.is_empty() && layout.functions.contains_key(&function)
        ),
        "ordinary function retains its selected function layout"
    );
    assert_private_dag_uses_exact_vm_stack(&executable);
}

#[test]
fn private_dag_rejects_graph_fanout_and_preserves_user_store() {
    let bytecode = pure_dag();
    let edges = bridges(&bytecode);
    let groups = bytecode.runtime_fusion_groups();
    let (mut graph, stores) = crate::accel::graph::build_accel_graph_with_stores(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    let edge = edges
        .iter()
        .find(|edge| {
            groups.iter().any(|group| {
                group.kind == FusionKind::Reduction
                    && group.span.start <= edge.producer_pc
                    && edge.consumer_pc <= group.span.end
            }) && !matches!(
                crate::bytecode::compile::instr_fusion_signal_kind(
                    &bytecode.instructions[edge.consumer_pc]
                ),
                Some(crate::bytecode::FusionInstructionKind::Reduction)
            )
        })
        .expect("intermediate private edge");
    let producer = graph
        .nodes
        .iter()
        .find(|node| node.span.end == edge.producer_pc)
        .expect("producer");
    let output = producer.outputs[0];
    let unrelated = graph
        .nodes
        .iter_mut()
        .find(|node| {
            node.span.start
                == groups
                    .iter()
                    .find(|group| group.kind == FusionKind::Reduction)
                    .unwrap()
                    .span
                    .end
        })
        .expect("different reduction consumer");
    unrelated.inputs.push(output);
    assert!(!project_private_bindings(
        &mut graph,
        &groups,
        &edges,
        &[],
        &mut [],
        &stores,
        &[]
    ));

    let named = compiled(
        "x = gpuArray(single(ones(4, 4))); a = x + x; y = min((a + a) + (a + a)); z = gather(y);",
    );
    let named_store = named
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "a").then_some(*slot))
        .expect("observable binding");
    let (plan, graph) = named.prepare_vm_fusion_plan();
    let plan = plan.expect("nested private suffix may still fold");
    let graph = graph.expect("same instruction graph");
    let named_pc = named
        .instructions
        .iter()
        .position(
            |instruction| matches!(instruction, Instr::StoreVar(slot) if *slot == named_store),
        )
        .expect("observable store");
    assert!(plan
        .groups
        .iter()
        .filter(|group| group.group.kind == FusionKind::Reduction)
        .all(|group| group.group.span.start > named_pc));
    assert!(graph.var_bindings.values().any(|binding| {
        binding.kind == runmat_accelerate::graph::VarKind::Global && binding.index == named_store
    }));
}

#[test]
fn intermediate_effect_or_extra_use_cannot_hide_inside_dag_region() {
    let original = pure_dag();
    let edges = bridges(&original);
    let reduction = original
        .runtime_fusion_groups()
        .into_iter()
        .find(|group| group.kind == FusionKind::Reduction)
        .expect("selected reduction");
    let first = edges
        .iter()
        .filter(|edge| {
            reduction.span.start <= edge.producer_pc && edge.consumer_pc <= reduction.span.end
        })
        .min_by_key(|edge| edge.producer_pc)
        .expect("first private producer");
    let intervening = edges
        .iter()
        .find(|edge| first.producer_pc < edge.producer_pc && edge.producer_pc < first.consumer_pc)
        .expect("independent pure producer between first store and consumer");

    let mut effectful = original.clone();
    let projection =
        std::sync::Arc::make_mut(effectful.fusion_metadata.stage_projection.as_mut().unwrap());
    projection
        .statements
        .iter_mut()
        .find(|statement| {
            statement.pc.start <= intervening.store_pc && intervening.store_pc < statement.pc.end
        })
        .expect("intervening selected statement")
        .contract
        .effects
        .0
        .insert(runmat_types::EffectKind::WorkspaceWrite);
    effectful.reseal_fusion_stage_after_owned_remap();
    assert!(effectful.validate_fusion_stage_projection().is_ok());
    assert!(
        bridges(&effectful)
            .iter()
            .all(|edge| edge.slot != first.slot && edge.slot != intervening.slot),
        "an effectful intervening statement cannot be omitted by a private bridge"
    );

    let mut escaped = original.clone();
    let last = escaped.instructions.len() - 1;
    escaped.instructions[last] = Instr::LoadVar(first.slot);
    escaped.reseal_fusion_stage_after_owned_remap();
    assert!(escaped.validate_fusion_stage_projection().is_ok());
    assert!(
        bridges(&escaped).iter().all(|edge| edge.slot != first.slot),
        "whole-stream no-escape proof must see an additional load"
    );
}
