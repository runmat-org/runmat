use super::*;
use crate::bytecode::Instr;
use runmat_accelerate::fusion::FusionOp;
use runmat_accelerate::graph::{PrimitiveOp, VarBinding, VarKind};

fn compile_source(source: &str) -> Bytecode {
    let ast = runmat_parser::parse(source).expect("parse authentic source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).expect("bytecode")
}

fn folded_source() -> Bytecode {
    compile_source("x = gpuArray(single([1, 2; 3, 4])); y = sum(x + x); z = gather(y);")
}

fn folded_function() -> (crate::bytecode::FunctionRegistry, runmat_hir::FunctionId) {
    let ast = runmat_parser::parse("function y = f(x)\ny = min(x + x);\nend\n")
        .expect("parse real function");
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
fn authentic_function_private_temp_survives_ordinary_dispatch_and_portable_roundtrip() {
    let (registry, function) = folded_function();
    let body = registry.get(function).expect("compiled definition");
    let executable = body.execution_bytecode(&registry);
    assert!(
        executable.layout.is_none(),
        "ordinary runner has no entrypoint layout"
    );
    let private_bridges = bridges(&executable);
    let [bridge] = private_bridges.as_slice() else {
        panic!(
            "exact function-owned bridge required: {:?}",
            executable.instructions
        );
    };
    let groups = executable.runtime_fusion_groups();
    let (plan, graph) = executable.prepare_vm_fusion_plan();
    assert!(graph.is_some(), "actual function instruction graph");
    let plan = plan.expect("checked function reduction plan");
    assert!(groups.iter().any(|group| {
        group.kind == FusionKind::Reduction
            && group.span.start < bridge.store_pc
            && group.span.end > bridge.load_pc
    }));
    assert!(plan.groups.iter().any(|group| {
        group.group.kind == FusionKind::Reduction
            && group.group.nodes.len() == 2
            && group.materialized_stores.is_empty()
    }));

    let encoded = crate::bytecode::encode_interpreter_program_v2(&registry)
        .expect("portable function artifact");
    let decoded = crate::bytecode::decode_interpreter_program_v2(&encoded)
        .expect("valid portable definition");
    let decoded_body = decoded.get(function).expect("decoded definition");
    assert_eq!(bridges(&decoded_body.execution_bytecode(&decoded)).len(), 1);

    let detached = body.clone();
    assert!(
        bridges(&detached.execution_bytecode(&registry)).is_empty(),
        "a same-ID detached body cannot borrow the registry's admission"
    );
}

#[test]
fn authentic_private_mir_temp_forms_one_reduction_region_and_graph_closure() {
    let bytecode = folded_source();
    let evidence = bridges(&bytecode);
    assert_eq!(
        evidence.len(),
        1,
        "one private single-use temporary; projection={:#?}; layout={:#?}; candidates={:#?}; instructions={:#?}",
        bytecode.fusion_metadata.stage_projection,
        bytecode.layout,
        bytecode.fusion_metadata.mir_fusion_candidate_groups,
        bytecode.instructions,
    );
    let bridge = evidence[0];
    assert!(matches!(
        bytecode.instructions[bridge.store_pc],
        Instr::StoreVar(slot) if slot == bridge.slot
    ));
    assert!(matches!(
        bytecode.instructions[bridge.load_pc],
        Instr::LoadVar(slot) if slot == bridge.slot
    ));
    let groups = bytecode.runtime_fusion_groups();
    let reduction = groups
        .iter()
        .find(|group| group.kind == FusionKind::Reduction)
        .expect("real compiled reduction");
    assert_eq!(reduction.span.start + 1, bridge.store_pc);
    assert_eq!(reduction.span.end, bridge.load_pc + 1);
    let (plan, graph) = bytecode.prepare_vm_fusion_plan();
    let plan = plan.expect("one checked in-span producer plan");
    let graph = graph.expect("same source graph");
    let selected = plan
        .groups
        .iter()
        .find(|group| group.group.kind == FusionKind::Reduction)
        .expect("reduction plan");
    assert_eq!(selected.group.span, reduction.span);
    assert_eq!(selected.group.nodes.len(), 2);
    assert!(
        selected.materialized_stores.is_empty(),
        "private reduction store is not published"
    );
    assert!(selected.operations.iter().any(|operation| {
        matches!(
            operation,
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                ..
            }
        )
    }));
    assert!(graph
        .nodes
        .iter()
        .find(|node| node.span.end + 1 == bridge.store_pc)
        .and_then(|node| node.outputs.first())
        .is_some_and(|value| graph.var_binding(*value).is_none()));
}

#[test]
fn graph_binding_elision_requires_exact_global_value_and_producer_node_ownership() {
    let bytecode = folded_source();
    let bridge = bridges(&bytecode)[0];
    let groups = bytecode.runtime_fusion_groups();
    let (graph, stores) = crate::accel::graph::build_accel_graph_with_stores(
        &bytecode.instructions,
        &bytecode.var_types,
    );
    let producer = graph
        .nodes
        .iter()
        .find(|node| node.span.end + 1 == bridge.store_pc)
        .expect("authentic producer");
    let output = *producer.outputs.first().expect("one output");
    let expected = VarBinding {
        kind: VarKind::Global,
        index: bridge.slot,
    };
    assert_eq!(graph.var_binding(output), Some(&expected));
    assert_eq!(graph.node_binding(producer.id), Some(&expected));
    assert!(stores.iter().any(|store| {
        store.pc == bridge.store_pc && store.value_id == output && store.binding == expected
    }));

    let mut missing_value = graph.clone();
    missing_value.var_bindings.remove(&output);
    assert!(!project_private_bindings(
        &mut missing_value,
        &groups,
        &[bridge],
        &[],
        &stores
    ));

    let mut missing_node = graph.clone();
    missing_node.node_bindings.remove(&producer.id);
    assert!(!project_private_bindings(
        &mut missing_node,
        &groups,
        &[bridge],
        &[],
        &stores
    ));

    let mut wrong_kind = graph.clone();
    wrong_kind.node_bindings.insert(
        producer.id,
        VarBinding {
            kind: VarKind::Local,
            index: bridge.slot,
        },
    );
    assert!(!project_private_bindings(
        &mut wrong_kind,
        &groups,
        &[bridge],
        &[],
        &stores
    ));

    let mut wrong_pc = stores.clone();
    wrong_pc
        .iter_mut()
        .find(|store| store.pc == bridge.store_pc)
        .unwrap()
        .pc += 1;
    assert!(!project_private_bindings(
        &mut graph.clone(),
        &groups,
        &[bridge],
        &[],
        &wrong_pc
    ));
}

#[test]
fn user_binding_is_not_a_private_bridge() {
    let bytecode =
        compile_source("x = gpuArray(single([1, 2; 3, 4])); t = x + x; y = sum(t); z = gather(y);");
    assert!(bridges(&bytecode).is_empty());
    assert!(bytecode
        .runtime_fusion_groups()
        .iter()
        .filter(|group| group.kind == FusionKind::Reduction)
        .all(|group| group.span.start == group.span.end));
}

#[test]
fn altered_stream_or_binding_ownership_cannot_reuse_private_proof() {
    let bytecode = folded_source();
    let bridge = bridges(&bytecode)[0];

    let mut extra_use = bytecode.clone();
    extra_use.instructions.push(Instr::LoadVar(bridge.slot));
    extra_use.catch_scopes.push(None);
    extra_use.instr_spans.push(runmat_hir::Span::default());
    assert!(bridges(&extra_use).is_empty(), "whole-stream unique use");

    let mut foreign_catch = bytecode.clone();
    foreign_catch.catch_scopes[bridge.load_pc] = Some(runmat_mir::CatchScopeId(7));
    assert!(bridges(&foreign_catch).is_empty(), "catch boundary");

    let mut foreign_source = bytecode.clone();
    foreign_source.source_id = Some(crate::bytecode::BytecodeSourceRef::Program(
        runmat_types::ProgramSourceId(u32::MAX),
    ));
    assert!(
        bridges(&foreign_source).is_empty(),
        "sealed source identity"
    );

    let mut observable = bytecode.clone();
    let function = observable
        .fusion_metadata
        .mir_fusion_candidate_groups
        .first()
        .expect("MIR candidate")
        .function;
    let layout = observable
        .layout
        .as_mut()
        .and_then(|layout| layout.functions.get_mut(&function))
        .expect("same function layout");
    let binding = *layout.binding_slots.keys().next().expect("user binding");
    layout
        .binding_slots
        .insert(binding, crate::layout::VmSlotId(bridge.slot));
    assert!(bridges(&observable).is_empty(), "binding ownership");

    let mut changed_slot = bytecode.clone();
    let layout = changed_slot
        .layout
        .as_mut()
        .and_then(|layout| layout.functions.get_mut(&function))
        .expect("same function layout");
    let sealed = bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .and_then(|projection| {
            projection
                .private_temporary_slots
                .iter()
                .find(|pair| pair.slot == bridge.slot)
        })
        .expect("sealed private temporary");
    layout.mir_local_slots.insert(
        runmat_mir::MirLocalId(sealed.local as usize),
        crate::layout::VmSlotId(bridge.slot + 1),
    );
    assert!(bridges(&changed_slot).is_empty(), "exact sealed slot join");

    let mut changed_role = bytecode.clone();
    std::sync::Arc::make_mut(
        changed_role
            .fusion_metadata
            .stage_projection
            .as_mut()
            .unwrap(),
    )
    .private_temporary_slots
    .clear();
    assert!(
        bridges(&changed_role).is_empty(),
        "role inventory is fingerprinted"
    );
}

#[test]
fn compiler_mir_role_and_fieldless_roundtrip_do_not_mint_private_bridge() {
    let original = folded_source();
    let bridge = bridges(&original)[0];
    let sealed = original
        .fusion_metadata
        .stage_projection
        .as_ref()
        .and_then(|projection| {
            projection
                .private_temporary_slots
                .iter()
                .find(|pair| pair.slot == bridge.slot)
        })
        .expect("compiled temporary");
    let source = "x = gpuArray(single([1, 2; 3, 4])); y = sum(x + x); z = gather(y);";
    let ast = runmat_parser::parse(source).expect("parse authentic source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mut mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let function = original
        .fusion_metadata
        .mir_fusion_candidate_groups
        .first()
        .expect("candidate")
        .function;
    mir.bodies.get_mut(&function).expect("MIR body").locals[sealed.local as usize].kind =
        runmat_mir::MirLocalKind::Binding;
    let relabeled = crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("same code with changed MIR role");
    assert!(bridges(&relabeled).is_empty(), "non-Temporary MIR role");

    let mut fieldless = original.clone();
    let projection = std::sync::Arc::make_mut(
        fieldless
            .fusion_metadata
            .stage_projection
            .as_mut()
            .expect("projection"),
    );
    projection.private_temporary_slots.clear();
    fieldless.reseal_fusion_stage_after_owned_remap();
    let encoded = crate::bytecode::encode_interpreter_script_v2(&fieldless)
        .expect("canonical fieldless bytes");
    let restored = crate::bytecode::decode_interpreter_script_v2(&encoded)
        .expect("fieldless current schema remains canonical");
    let wire = serde_json::to_value(restored.fusion_metadata.stage_projection.as_ref().unwrap())
        .expect("JSON");
    assert!(wire.get("private_temporary_slots").is_none());
    assert!(bridges(&restored).is_empty());
    let new_encoded =
        crate::bytecode::encode_interpreter_script_v2(&original).expect("new inventory artifact");
    let new_restored = crate::bytecode::decode_interpreter_script_v2(&new_encoded)
        .expect("new inventory roundtrip");
    assert_eq!(bridges(&new_restored).len(), 1);
}

#[test]
fn function_or_legacy_origin_cannot_mint_entrypoint_bridge_even_with_valid_seal() {
    let original = folded_source();
    for origin in [
        Some(crate::bytecode::FusionStageSelectionOrigin::FunctionBody),
        None,
    ] {
        let mut changed = original.clone();
        std::sync::Arc::make_mut(changed.fusion_metadata.stage_projection.as_mut().unwrap())
            .selection_origin = origin;
        changed.reseal_fusion_stage_after_owned_remap();
        assert!(changed.validate_fusion_stage_projection().is_ok());
        assert!(
            bridges(&changed).is_empty(),
            "entrypoint-only intermediate proof"
        );
    }
}

#[test]
fn freshly_resealed_nonload_slot_references_defeat_whole_stream_uniqueness() {
    let original = folded_source();
    let slot = bridges(&original)[0].slot;
    let foreign_uses = [
        Instr::EmitVar {
            var_index: slot,
            label: crate::bytecode::EmitLabel::Ans,
        },
        Instr::EmitStackTop {
            label: crate::bytecode::EmitLabel::Var(slot),
        },
        Instr::DeclareGlobal(vec![slot]),
        Instr::CallBuiltinMultiUsingOutputSlot("sum".into(), 1, slot),
    ];
    for foreign_use in foreign_uses {
        let mut changed = original.clone();
        let last = changed.instructions.len() - 1;
        changed.instructions[last] = foreign_use;
        changed.reseal_fusion_stage_after_owned_remap();
        assert!(changed.validate_fusion_stage_projection().is_ok());
        assert!(
            bridges(&changed).is_empty(),
            "third slot reference blocks bridge"
        );
    }
}

#[test]
fn only_terminal_reduction_throw_or_suspend_effects_cross_private_bridge() {
    let original = folded_source();
    let bridge = bridges(&original)[0];
    let projection = original.fusion_metadata.stage_projection.as_ref().unwrap();
    let reduction = projection
        .statements
        .iter()
        .find(|statement| statement.pc.start <= bridge.load_pc && bridge.load_pc < statement.pc.end)
        .expect("certified terminal reduction statement");
    assert!(reduction
        .contract
        .effects
        .0
        .contains(&runmat_types::EffectKind::MayThrow));
    assert!(reduction
        .contract
        .effects
        .0
        .contains(&runmat_types::EffectKind::MaySuspend));

    for (pc, effect) in [
        (bridge.store_pc, runmat_types::EffectKind::MayThrow),
        (bridge.load_pc, runmat_types::EffectKind::WorkspaceWrite),
        (bridge.load_pc, runmat_types::EffectKind::Unknown),
    ] {
        let mut changed = original.clone();
        let projection =
            std::sync::Arc::make_mut(changed.fusion_metadata.stage_projection.as_mut().unwrap());
        projection
            .statements
            .iter_mut()
            .find(|statement| statement.pc.start <= pc && pc < statement.pc.end)
            .expect("selected statement")
            .contract
            .effects
            .0
            .insert(effect);
        changed.reseal_fusion_stage_after_owned_remap();
        assert!(changed.validate_fusion_stage_projection().is_ok());
        assert!(
            bridges(&changed).is_empty(),
            "effect role must remain exact"
        );
    }
}

#[test]
fn final_only_store_gate_excludes_only_certified_private_pc() {
    let bytecode = folded_source();
    let bridge = bridges(&bytecode)[0];
    let group = bytecode
        .runtime_fusion_groups()
        .into_iter()
        .find(|group| group.kind == FusionKind::Reduction)
        .expect("reduction");
    assert!(
        crate::accel::fusion::final_only_span_requires_ordinary_stores(
            &group.kind,
            &bytecode.instructions,
            &group.span,
        )
    );
    assert!(
        !crate::accel::fusion::final_only_span_requires_ordinary_stores_with_private(
            &group.kind,
            &bytecode.instructions,
            &group.span,
            &[bridge.store_pc],
        )
    );
}

#[path = "tests/empty_control.rs"]
mod empty_control;
