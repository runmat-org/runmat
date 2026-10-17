use super::super::{
    bridges, empty_controls, private_reduction_store_pcs, project_private_bindings,
};
use super::compile_source;
use crate::bytecode::Instr;
use runmat_accelerate::fusion::prepare_fusion_plan;
use runmat_accelerate::FusionKind;

#[test]
fn selected_empty_axis_control_projects_only_its_private_binding() {
    let bytecode =
        compile_source("x = gpuArray(single([1, 2; 3, 4])); y = min(x + x, [], 2); z = gather(y);");
    let controls = empty_controls(&bytecode);
    let [control] = controls.as_slice() else {
        panic!("one sealed empty-axis control: {:?}", bytecode.instructions);
    };
    assert_eq!(
        bridges(&bytecode).len(),
        1,
        "producer bridge remains separate"
    );
    let (raw_graph, stores) = crate::accel::graph::build_accel_graph_with_stores(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    let raw_control = raw_graph
        .nodes
        .iter()
        .find(|node| node.span.end == control.producer_pc)
        .expect("empty literal graph node");
    let [control_output] = raw_control.outputs.as_slice() else {
        panic!("one control output");
    };
    assert!(raw_graph.var_binding(*control_output).is_some());
    assert!(raw_graph.node_binding(raw_control.id).is_some());
    let mut bound_control = raw_graph.clone();
    let groups = bytecode.runtime_fusion_groups();
    assert!(project_private_bindings(
        &mut bound_control,
        &groups,
        &bridges(&bytecode),
        &[],
        &mut [],
        &stores,
        &[],
    ));
    let selected_groups = bytecode.runtime_fusion_groups_for_graph(&bound_control);
    assert!(
        prepare_fusion_plan(
            Some(&bound_control),
            &selected_groups,
            bytecode.fusion_metadata.mir_fusion_candidate_group_count,
        )
        .is_none(),
        "an unprojected StoreVar control remains an observable barrier"
    );

    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let graph = graph.expect("authentic graph");
    let projected_control = graph.node(raw_control.id).expect("same graph node");
    assert!(graph.var_binding(*control_output).is_none());
    assert!(graph.node_binding(projected_control.id).is_none());
    let plan = plan.expect("selected control does not break the checked reduction DAG");
    let reduction = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("folded reduction");
    assert_eq!(reduction.group.nodes.len(), 2, "Add and Min only");
    assert!(reduction.materialized_stores.is_empty());
    assert!(
        reduction.kernel.supported,
        "the selected kernel is executable"
    );
    assert!(!crate::accel::fusion::fusion_span_has_vm_barrier(
        &bytecode.instructions,
        &reduction.group.span,
    ));
    let bridge = bridges(&bytecode);
    let private_without_control = private_reduction_store_pcs(&reduction.group, &bridge, &[]);
    assert!(
        crate::accel::fusion::final_only_span_requires_ordinary_stores_with_private(
            &reduction.group.kind,
            &bytecode.instructions,
            &reduction.group.span,
            &private_without_control,
        )
    );
    let private_with_control = private_reduction_store_pcs(&reduction.group, &bridge, &controls);
    assert!(private_with_control.contains(&control.store_pc));
    assert!(
        !crate::accel::fusion::final_only_span_requires_ordinary_stores_with_private(
            &reduction.group.kind,
            &bytecode.instructions,
            &reduction.group.span,
            &private_with_control,
        )
    );

    let mut changed = bytecode.clone();
    changed.instructions[control.producer_pc] = Instr::CreateMatrix(1, 1);
    changed.reseal_fusion_stage_after_owned_remap();
    assert!(
        empty_controls(&changed).is_empty(),
        "freshly resealed nonempty control cannot borrow [] authority"
    );
    let mut unowned = bytecode.clone();
    std::sync::Arc::make_mut(
        unowned
            .fusion_metadata
            .stage_projection
            .as_mut()
            .expect("stage projection"),
    )
    .private_temporary_slots
    .retain(|pair| pair.slot != control.slot);
    unowned.reseal_fusion_stage_after_owned_remap();
    assert!(
        empty_controls(&unowned).is_empty(),
        "missing sealed Temporary pair"
    );
    assert_eq!(
        empty_controls(&bytecode).len(),
        1,
        "clone mutation does not revoke original control"
    );
}

#[test]
fn selected_empty_axis_control_survives_function_artifact_roundtrip() {
    let ast = runmat_parser::parse("function y = f(x)\ny = min(x + x, [], 2);\nend\n")
        .expect("parse function");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let function = hir.assembly.functions[0].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
            .expect("function bytecode"),
    );
    let body = registry.get(function).expect("compiled function");
    let executable = body.execution_bytecode(&registry);
    assert_eq!(empty_controls(&executable).len(), 1);
    let (plan, _) = executable.prepare_vm_fusion_plan();
    let plan = plan.expect("function reduction graph");
    assert!(plan.groups.iter().any(|group| {
        group.group.kind == FusionKind::Reduction
            && group.group.nodes.len() == 2
            && group.materialized_stores.is_empty()
    }));
    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("portable function program");
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded)
        .expect("canonical function artifact");
    let decoded_execution = decoded
        .get(function)
        .expect("decoded function")
        .execution_bytecode(&decoded);
    assert_eq!(empty_controls(&decoded_execution).len(), 1);
    assert!(decoded_execution.prepare_vm_fusion_plan().0.is_some());
}
