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
        executable.layout.is_none(),
        "ordinary function runner has no entrypoint layout"
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
    assert!(decoded_execution.layout.is_none());
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
    assert_eq!(edges.len(), 3, "three sealed private edges: {edges:?}");
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
    assert_eq!(
        groups
            .iter()
            .filter(|group| group.kind == FusionKind::Reduction)
            .count(),
        1,
    );
    assert!(edges.iter().all(|edge| {
        reduction.span.start <= edge.producer_pc && edge.consumer_pc <= reduction.span.end
    }));
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
    for edge in &edges {
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

    // The ordinary VM has two live operands above an unrelated lower stack
    // value at the first producer PC. Gather must consume only those operands.
    let lower = Value::Num(-37.0);
    let mut stack = vec![lower.clone(), Value::Num(3.0), Value::Num(5.0)];
    let original = stack.clone();
    let mut vars = vec![Value::Num(11.0); bytecode.var_count];
    let mut context = crate::bytecode::program::ExecutionContext::default();
    let (guard, request, _) = crate::accel::fusion::gather_fusion_inputs(
        selected,
        &graph,
        &mut stack,
        &mut vars,
        &mut context,
        None,
    )
    .expect("the one external leaf has its exact variable source");
    assert_eq!(request.inputs, vec![Value::Num(11.0)]);
    drop(request);
    guard.commit();
    assert_eq!(stack, vec![lower.clone()]);

    let mut too_shallow = vec![lower.clone()];
    assert!(crate::accel::fusion::gather_fusion_inputs(
        selected,
        &graph,
        &mut too_shallow,
        &mut vars,
        &mut context,
        None,
    )
    .is_err());
    assert_eq!(too_shallow, vec![lower.clone()]);

    // A malformed plan that asks for an internal private result must fail
    // before execution, restoring the exact original stack slice on error.
    let internal = graph
        .nodes
        .iter()
        .find(|node| selected.group.nodes.contains(&node.id) && !node.is_reduction())
        .and_then(|node| node.outputs.first())
        .copied()
        .expect("private producer output");
    let mut malformed = selected.clone();
    malformed.inputs.push(internal);
    stack = original.clone();
    let error = crate::accel::fusion::gather_fusion_inputs(
        &malformed,
        &graph,
        &mut stack,
        &mut vars,
        &mut context,
        None,
    )
    .err()
    .expect("internal output has no independent source");
    assert!(error.to_string().contains("missing input"));
    assert_eq!(stack, original);
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
        executable.layout.is_none(),
        "ordinary function has no entrypoint layout"
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
            !matches!(
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
        &stores
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
    let first = edges
        .iter()
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
