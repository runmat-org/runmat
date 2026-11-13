use super::*;
use crate::bytecode::{Bytecode, Instr};
use runmat_accelerate::fusion::image_reduction::ImageMeanForm;
use runmat_accelerate::fusion::image_reduction::{
    ImageSourceAdmissionFailure, ImageSourceInvocation, ImageSourceLeaf, ImageSourceTranscript,
};
use runmat_accelerate::graph::{AccelNodeLabel, FusionCallOrigin};

pub(super) fn compiled_image() -> Bytecode {
    let source = r#"
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+single(1e-6)));
        out = single(((imgs-mu)./sigma).^single(1.8));
    "#;
    let ast = runmat_parser::parse(source).expect("parse selected image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile selected image source")
}

pub(super) fn compiled_clamped_image() -> Bytecode {
    let source = r#"
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+eps0));
        out = single(((imgs-mu)./sigma)*gain+bias);
        out = max(out,single(0));
        out = single(out.^gamma);
    "#;
    let ast = runmat_parser::parse(source).expect("parse full selected image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile full selected image source")
}

#[test]
fn random_vecdim_image_source_seals_its_selected_region() {
    // The execution fixture begins with a random host Single source rather
    // than the constant `ones` source used by the other publication tests.
    // Detection alone does not establish a selected VM region.
    let source = r#"
        rng(0); B=2; H=4; W=5;
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);
        imgs = rand(B, H, W, 'single');
        mu = single(mean(imgs, [2 3], 'native'));
        sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
        out = single(((imgs - mu) ./ sigma) * gain + bias);
        out = max(out, single(0));
        out = single(out .^ gamma);
        err = out - imgs;
        mse = mean(err .* err, 'all');
        fprintf('RESULT_ok MSE=%.6e\n', double(mse));
    "#;
    let ast = runmat_parser::parse(source).expect("parse random image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let bytecode = crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile random image source");
    let selected = bytecode
        .sealed_selected_image_regions()
        .expect("random image source has a sealed selected descriptor");
    assert_eq!(selected.regions.len(), 1);
    let region = &selected.regions[0];
    let input = region
        .plan
        .inputs
        .iter()
        .position(|id| {
            matches!(
                selected.graph.value(*id).map(|value| &value.origin),
                Some(ValueOrigin::NodeOutput { .. })
            ) && selected.graph.var_binding(*id).is_some()
        })
        .expect("random image is a stored, live node output");
    let id = region.plan.inputs[input];
    assert!(matches!(
        &region.entry_inputs[input],
        ImageEntryInput::Binding(binding)
            if binding.kind == runmat_accelerate::graph::VarKind::Global
    ));
    assert!(region
        .plan
        .inputs
        .iter()
        .zip(&region.entry_inputs)
        .any(|(id, source)| {
            selected.graph.var_binding(*id).is_none()
                && matches!(source, ImageEntryInput::Binding(_))
        }), "a compiler-private input is sealed by its ordered store even when its graph binding is projected away");

    let graph = SelectedMultiStoreGraph::for_image(&bytecode).expect("same selected store walk");
    let mut wrong_binding = graph.stores.clone();
    let source_store = wrong_binding
        .iter()
        .rposition(|store| store.pc < region.publication.pcs().start && store.value_id == id)
        .expect("random source was stored before its selected region");
    wrong_binding[source_store].binding.index = usize::MAX;
    assert!(capture::sealed_entry_inputs(
        &selected.graph,
        &wrong_binding,
        &region.plan,
        &region.publication,
        &region.source,
    )
    .is_none());

    let mut overwritten = graph.stores;
    let prior_pc = overwritten[source_store].pc;
    let prior_binding = overwritten[source_store].binding.clone();
    let overwrite_pc = region.publication.pcs().start - 1;
    assert!(prior_pc < overwrite_pc);
    overwritten.push(crate::accel::graph::AccelStore {
        pc: overwrite_pc,
        value_id: region.plan.inputs[(input + 1) % region.plan.inputs.len()],
        binding: prior_binding,
    });
    overwritten.sort_by_key(|store| store.pc);
    assert!(capture::sealed_entry_inputs(
        &selected.graph,
        &overwritten,
        &region.plan,
        &region.publication,
        &region.source,
    )
    .is_none());
}

#[test]
fn function_only_full_image_source_uses_its_selected_function_layout() {
    // Function-local fusion metadata is carried by the selected definition;
    // no synthetic script entrypoint is available or authorized here.
    let source = r#"
        function out = imageFunction()
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+eps0));
        out = single(((imgs-mu)./sigma)*gain+bias);
        out = max(out,single(0));
        out = single(out.^gamma);
        end
        function out = otherFunction()
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.7); eps0=single(1e-6);
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+eps0));
        out = single(((imgs-mu)./sigma)*gain+bias);
        out = max(out,single(0));
        out = single(out.^gamma);
        end
    "#;
    let ast = runmat_parser::parse(source).expect("parse function-only image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    assert!(hir.assembly.entrypoints.is_empty());
    let function = hir.assembly.functions[0].id;
    let other_function = hir.assembly.functions[1].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
            .expect("compile function-only registry"),
    );
    let ordinary_view = registry
        .get(function)
        .expect("selected function")
        .execution_bytecode(&registry);
    let repeated_view = registry
        .get(function)
        .expect("same immutable selected function")
        .execution_bytecode(&registry);
    match (
        &ordinary_view.sealed_function_image,
        &repeated_view.sealed_function_image,
    ) {
        (
            crate::bytecode::program::SealedFunctionImage::Selected { regions: first, .. },
            crate::bytecode::program::SealedFunctionImage::Selected {
                regions: repeated, ..
            },
        ) => assert!(std::rc::Rc::ptr_eq(first, repeated)),
        _ => panic!("function descriptor was not finalized once per registry generation"),
    }
    let mut changed_registry = registry.clone();
    changed_registry
        .functions
        .get_mut(&function)
        .expect("authoring body")
        .instructions[0] = Instr::LoadConst(99.0);
    let changed_view = changed_registry
        .get(function)
        .expect("mutated body")
        .execution_bytecode(&changed_registry);
    assert!(matches!(
        changed_view.sealed_function_image,
        crate::bytecode::program::SealedFunctionImage::Unsealed
    ));
    assert!(ordinary_view
        .sealed_selected_image_regions()
        .is_some_and(|selected| !selected.regions.is_empty()));
    let layout = crate::layout::derive_layout(&hir.assembly, &mir).expect("selected layout");
    let mut conflicting_layout = layout.clone();
    let other_resume_points = registry
        .get(other_function)
        .expect("other real compiled function")
        .resume_points
        .clone();
    assert!(!other_resume_points.is_empty());
    assert_eq!(
        other_resume_points.len(),
        registry.get(function).unwrap().resume_points.len(),
        "different compiled function has the same statement projection shape"
    );
    conflicting_layout
        .functions
        .get_mut(&function)
        .expect("image function layout")
        .resume_points = other_resume_points.clone();
    let conflicting = Bytecode::for_function(
        registry.get(function).expect("selected image body"),
        registry.clone(),
        conflicting_layout,
    );
    assert!(conflicting
        .prospective_image_normalize_publications()
        .is_empty());
    let bytecode = Bytecode::for_function(
        registry.get(function).expect("selected image body"),
        registry.clone(),
        layout,
    );
    assert_eq!(bytecode.active_function, Some(function));
    assert!(bytecode.layout.as_ref().unwrap().entrypoints.is_empty());
    let detached_body = registry.get(function).unwrap().clone();
    let detached = Bytecode::for_function(
        &detached_body,
        registry.clone(),
        bytecode.layout.clone().unwrap(),
    );
    assert_eq!(
        serde_json::to_vec(&detached.instructions).unwrap(),
        serde_json::to_vec(&bytecode.instructions).unwrap()
    );
    assert!(detached.fusion_metadata.stage_projection.is_none());
    assert!(detached
        .prospective_image_normalize_publications()
        .is_empty());
    let joined = bytecode.prospective_image_normalize_publications();
    assert!(joined
        .iter()
        .any(|region| region.output_checkpoints().count() == 3));
    let first = bytecode
        .sealed_selected_image_regions()
        .expect("selected bytecode descriptor");
    let second = bytecode
        .sealed_selected_image_regions()
        .expect("reuse the exact descriptor");
    assert!(std::rc::Rc::ptr_eq(&first, &second));
    let mut next_invocation = bytecode.clone();
    if !next_invocation.initially_unassigned_slots.insert(0) {
        next_invocation.initially_unassigned_slots.remove(&0);
    }
    let reused = next_invocation
        .sealed_selected_image_regions()
        .expect("invocation state cannot invalidate immutable source");
    assert!(std::rc::Rc::ptr_eq(&first, &reused));
    let mut changed = bytecode.clone();
    changed.instructions[0] = Instr::LoadConst(99.0);
    assert!(changed.sealed_selected_image_regions().is_none());
    assert!(bytecode.sealed_selected_image_regions().is_some());
    let portable_registry: crate::bytecode::FunctionRegistry =
        serde_json::from_slice(&serde_json::to_vec(&registry).unwrap()).unwrap();
    let portable_view = Bytecode::for_function(
        portable_registry
            .get(function)
            .expect("portable selected image body"),
        portable_registry.clone(),
        bytecode.layout.clone().unwrap(),
    );
    assert!(portable_view
        .prospective_image_normalize_publications()
        .iter()
        .any(|region| region.output_checkpoints().count() == 3));
    let restored_view: Bytecode =
        serde_json::from_slice(&serde_json::to_vec(&portable_view).unwrap()).unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        restored_view
            .fusion_metadata
            .stage_projection
            .as_ref()
            .unwrap(),
        restored_view
            .function_registry
            .get(function)
            .unwrap()
            .fusion_stage_projection
            .as_ref()
            .unwrap(),
    ));
    // A portable view carries definitions, not the caller's active frame.
    // Restore selection only through the selected-body constructor below.
    assert!(restored_view.active_function.is_none());
    assert!(restored_view
        .prospective_image_normalize_publications()
        .is_empty());
    let hydrated_view = Bytecode::for_function(
        restored_view.function_registry.get(function).unwrap(),
        restored_view.function_registry.clone(),
        restored_view.layout.unwrap(),
    );
    assert!(hydrated_view
        .prospective_image_normalize_publications()
        .iter()
        .any(|region| region.output_checkpoints().count() == 3));

    let mut wrong_owner = bytecode.clone();
    wrong_owner.active_function = Some(other_function);
    assert!(wrong_owner
        .prospective_image_normalize_publications()
        .is_empty());
    let mut wrong_owner_with_copied_map = bytecode.clone();
    wrong_owner_with_copied_map.active_function = Some(other_function);
    wrong_owner_with_copied_map
        .layout
        .as_mut()
        .unwrap()
        .functions
        .get_mut(&other_function)
        .unwrap()
        .resume_points = bytecode.resume_points.clone();
    assert!(wrong_owner_with_copied_map
        .prospective_image_normalize_publications()
        .is_empty());
    let mut forged_entrypoint = wrong_owner_with_copied_map;
    let entrypoint = runmat_hir::EntrypointId(0);
    forged_entrypoint
        .layout
        .as_mut()
        .unwrap()
        .entrypoints
        .insert(
            entrypoint,
            crate::layout::VmEntrypointLayout {
                entrypoint,
                target: other_function,
                workspace_export: runmat_hir::WorkspaceExportPolicy::ReturnFunctionOutputs,
                exports: Vec::new(),
            },
        );
    assert!(forged_entrypoint
        .prospective_image_normalize_publications()
        .is_empty());
    let mut missing_selected_body = forged_entrypoint;
    assert!(missing_selected_body
        .function_registry
        .remove(other_function)
        .is_some());
    assert!(missing_selected_body
        .prospective_image_normalize_publications()
        .is_empty());
    let mut wrong_body_projection = bytecode.clone();
    wrong_body_projection
        .layout
        .as_mut()
        .unwrap()
        .functions
        .get_mut(&function)
        .unwrap()
        .resume_points = other_resume_points;
    assert!(wrong_body_projection
        .prospective_image_normalize_publications()
        .is_empty());
    let mut wrong_stream = bytecode;
    wrong_stream
        .layout
        .as_mut()
        .unwrap()
        .functions
        .get_mut(&function)
        .unwrap()
        .resume_points
        .clear();
    assert!(wrong_stream
        .prospective_image_normalize_publications()
        .is_empty());
}

#[test]
fn compiled_image_stages_join_exact_graph_producers_and_ordered_mir_stores() {
    let bytecode = compiled_image();
    let graph = crate::accel::graph::build_accel_graph(&bytecode.instructions, &bytecode.var_types);
    let group = graph
        .detect_fusion_groups()
        .into_iter()
        .find(|group| group.kind == FusionKind::ImageNormalize)
        .expect("selected source has image graph pattern");
    let projection = bytecode.fusion_metadata.stage_projection.as_ref().unwrap();
    let selected = SelectedMultiStoreGraph::for_image(&bytecode).unwrap();
    let proposal = selected_image_region(&bytecode, projection, &selected, &group)
        .expect("selected MIR has exactly ordered image stores");
    assert!(proposal
        .stores
        .iter()
        .filter(|entry| entry.named)
        .filter_map(|entry| bytecode.var_names.get(&entry.store.binding.index))
        .map(String::as_str)
        .eq(["mu", "sigma", "out"]));
    let joined = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find(|joined| {
            [
                joined.checkpoint(ImageCheckpoint::Mu).unwrap(),
                joined.checkpoint(ImageCheckpoint::Sigma).unwrap(),
                joined.checkpoint(ImageCheckpoint::Final).unwrap(),
            ]
            .into_iter()
            .filter_map(|store| bytecode.var_names.get(&store.binding.index))
            .map(String::as_str)
            .eq(["mu", "sigma", "out"])
        })
        .expect("one selected bytecode graph walk joins image publications");
    let named = proposal
        .stores
        .iter()
        .filter(|entry| entry.named)
        .collect::<Vec<_>>();
    assert_eq!(
        joined.checkpoint(ImageCheckpoint::Mu).unwrap().pc,
        named[0].store.pc
    );
    assert_eq!(
        joined.checkpoint(ImageCheckpoint::Sigma).unwrap().pc,
        named[1].store.pc
    );
    assert_eq!(joined.output_checkpoints().count(), 1);
    assert_eq!(
        joined.checkpoint(ImageCheckpoint::Final).unwrap().pc,
        named[2].store.pc
    );

    let mut swapped = proposal.clone();
    let named_indices = swapped
        .stores
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| entry.named.then_some(index))
        .collect::<Vec<_>>();
    swapped.stores.swap(named_indices[0], named_indices[1]);
    assert!(reconcile_image_publications(&swapped, &graph, &group).is_none());

    let mut reordered = proposal.clone();
    let first_pc = reordered.stores[named_indices[0]].store.pc;
    reordered.stores[named_indices[1]].store.pc = first_pc;
    assert!(reconcile_image_publications(&reordered, &graph, &group).is_none());

    let mut aliased = proposal.clone();
    let first_binding = aliased.stores[named_indices[0]].store.binding.clone();
    aliased.stores[named_indices[1]].store.binding = first_binding;
    assert!(reconcile_image_publications(&aliased, &graph, &group).is_none());

    let mut extra = proposal.clone();
    let extra_store = extra.stores[named_indices[2]].clone();
    extra.stores.push(extra_store);
    assert!(reconcile_image_publications(&extra, &graph, &group).is_none());

    let mut wrong_value = proposal.clone();
    let first_value = wrong_value.stores[named_indices[0]].store.value_id;
    wrong_value.stores[named_indices[1]].store.value_id = first_value;
    assert!(reconcile_image_publications(&wrong_value, &graph, &group).is_none());

    let mut wrong_graph = graph.clone();
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref().unwrap() else {
        unreachable!("selected image group")
    };
    let route = &pattern.reduction_routes.as_ref().unwrap().mu;
    let mean_node = match &route.form {
        ImageMeanForm::VecDims { call, .. } => call.node,
        ImageMeanForm::Nested { outer, .. } => outer.node,
    };
    let AccelNodeLabel::Builtin { origin, .. } = &mut wrong_graph.nodes[mean_node as usize].label
    else {
        unreachable!("selected mean call")
    };
    *origin = FusionCallOrigin::NamedUnresolved;
    assert!(reconcile_image_publications(&proposal, &wrong_graph, &group).is_none());

    // Local ValueIds would be allocated in the same order after this source
    // mutation, but the selected MIR projection is bound to the original
    // instruction stream. Public selection cannot supply the old graph.
    let mut changed_source = bytecode.clone();
    let literal = changed_source
        .instructions
        .iter_mut()
        .find(|instruction| matches!(instruction, Instr::LoadConst(_)))
        .expect("image source emits a numeric literal");
    *literal = Instr::LoadConst(99.0);
    assert!(changed_source
        .prospective_image_normalize_publications()
        .is_empty());
}

#[test]
fn compiled_gain_bias_clamp_keeps_every_out_publication_in_source_order() {
    // Requires the owning canonical pairwise max catalog binding. Until that
    // migration lands this test must fail closed, not accept an unresolved
    // spelling as selected source evidence.
    let bytecode = compiled_clamped_image();
    let graph = crate::accel::graph::build_accel_graph(&bytecode.instructions, &bytecode.var_types);
    let group = graph
        .detect_fusion_groups()
        .into_iter()
        .find(|group| group.kind == FusionKind::ImageNormalize)
        .expect("compiled gain/bias/clamp image pattern");
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref().unwrap() else {
        unreachable!("image group has image pattern")
    };
    assert!(pattern.clamp_zero);
    let producers = pattern
        .reduction_routes
        .as_ref()
        .unwrap()
        .stage_producers()
        .unwrap();
    assert!(producers.pre_clamp_output.is_some());
    assert!(producers.clamp_output.is_some());
    let source = ImageSourceTranscript::from_graph(&graph, &group)
        .expect("full selected graph retains its ordered arithmetic source");
    assert_eq!(source.checkpoints.len(), 5);
    assert!(source
        .steps
        .windows(2)
        .all(|pair| pair[0].pc.end < pair[1].pc.start));
    assert!(source.steps.iter().any(|step| {
        matches!(
            &step.label,
            AccelNodeLabel::Primitive(
                runmat_accelerate::graph::PrimitiveOp::ElemMul
                    | runmat_accelerate::graph::PrimitiveOp::Mul
            )
        )
    }));
    assert!(source.steps.iter().any(|step| {
        matches!(
            &step.label,
            AccelNodeLabel::Primitive(runmat_accelerate::graph::PrimitiveOp::Add)
        )
    }));
    assert!(source.steps.iter().any(|step| {
        matches!(&step.label, AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("max"))
            && step.selected_binding.is_some()
    }));
    let selected_constant = |id| {
        source.leaves.iter().any(|leaf| {
            matches!(leaf, ImageSourceLeaf::Constant { id: selected, .. } if *selected == id)
        })
    };
    assert!(graph.values.iter().any(|info| {
        matches!(&info.constant, Some(runmat_value::Value::Num(value)) if *value == 2.0)
            && selected_constant(info.id)
    }));
    assert!(
        graph.values.iter().any(|info| {
            matches!(&info.constant, Some(runmat_value::Value::Tensor(tensor)) if tensor.len() == 2)
                && selected_constant(info.id)
        }),
        "dimension-vector constants must not disappear from the source transcript"
    );
    assert!(
        graph.values.iter().any(|info| {
            matches!(&info.constant, Some(runmat_value::Value::String(value)) if value == "native")
                && selected_constant(info.id)
        }),
        "mean control strings must not disappear from the source transcript"
    );
    assert!(source
        .leaves
        .iter()
        .any(|leaf| matches!(leaf, ImageSourceLeaf::Live { id } if *id == pattern.input)));
    let mut stale_constant = graph.clone();
    stale_constant.values[pattern.input as usize].constant = Some(runmat_value::Value::Num(999.0));
    let same_live_authority = ImageSourceTranscript::from_graph(&stale_constant, &group)
        .expect("live input origin wins over a stale graph constant");
    assert_eq!(same_live_authority.leaves, source.leaves);
    let selected_plan =
        runmat_accelerate::fusion::FusionPlan::from_graph(&graph, std::slice::from_ref(&group))
            .groups
            .remove(0);
    let mut invocation_inputs = selected_plan
        .inputs
        .iter()
        .map(|id| {
            use runmat_accelerate::fusion::ImageScalar;
            let single = |value| {
                runmat_value::Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![value], vec![1, 1]).unwrap(),
                )
            };
            if *id == pattern.input {
                return runmat_value::Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![1.0; 40], vec![2, 4, 5]).unwrap(),
                );
            }
            if source.leaves.iter().any(
                |leaf| matches!(leaf, ImageSourceLeaf::Constant { id: constant, .. } if constant == id),
            ) {
                return selected_plan
                    .const_values
                    .get(id)
                    .cloned()
                    .or_else(|| graph.value(*id).and_then(|value| value.constant.clone()))
                    .unwrap_or_else(|| panic!("selected source constant {id} has no carrier"));
            }
            let routes = pattern.reduction_routes.as_ref().unwrap();
            for form in [&routes.mu.form, &routes.variance.form] {
                let ImageMeanForm::VecDims { call, .. } = form else {
                    panic!("compiled clamped fixture uses vecdim means");
                };
                if call.axes_value == *id {
                    return runmat_value::Value::Tensor(
                        runmat_value::Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
                    );
                }
                let mean = source
                    .steps
                    .iter()
                    .find(|step| step.node == call.node)
                    .expect("selected mean call remains in the source transcript");
                if mean.inputs.get(2) == Some(id) {
                    return runmat_value::Value::String("native".into());
                }
            }
            for (control, value) in [
                (Some(&pattern.epsilon), 1e-6_f32),
                (pattern.gain.as_ref(), 1.0123_f32),
                (pattern.bias.as_ref(), -0.02_f32),
                (pattern.gamma.as_ref(), 1.8_f32),
            ] {
                if matches!(control, Some(ImageScalar::Value(value_id)) if value_id == id) {
                    return single(value);
                }
            }
            if pattern.clamp_zero_value == Some(*id) {
                return single(0.0);
            }
            panic!("unresolved live selected source input {id}")
        })
        .collect::<Vec<_>>();
    for (control, expected) in [
        (Some(&pattern.epsilon), 1e-6_f32),
        (pattern.gain.as_ref(), 1.0123_f32),
        (pattern.bias.as_ref(), -0.02_f32),
        (pattern.gamma.as_ref(), 1.8_f32),
    ] {
        let Some(runmat_accelerate::fusion::ImageScalar::Value(id)) = control else {
            continue;
        };
        let Some(position) = selected_plan
            .inputs
            .iter()
            .position(|candidate| candidate == id)
        else {
            assert!(
                selected_plan.const_values.contains_key(id)
                    || graph
                        .value(*id)
                        .and_then(|value| value.constant.as_ref())
                        .is_some(),
                "selected scalar control is either a live input or a compiled constant"
            );
            continue;
        };
        let runmat_value::Value::Tensor(tensor) = &invocation_inputs[position] else {
            panic!("live selected scalar control has a Single tensor carrier");
        };
        assert_eq!(tensor.len(), 1);
        assert_eq!(tensor.shape(), &[1, 1]);
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F32);
        let Some(runmat_value::NumericScalar::F32(actual)) = tensor.numeric_value_at(0) else {
            panic!("live selected scalar control has an F32 value");
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let admitted = futures::executor::block_on(ImageSourceInvocation::bind(
        &graph,
        &selected_plan,
        &invocation_inputs,
        runmat_accelerate_api::ProviderPrecision::F32,
        None,
    ))
    .expect("full selected source is request-bound before hook projection");
    let power_control_classes = source
        .steps
        .iter()
        .filter(|step| {
            matches!(
                &step.label,
                AccelNodeLabel::Primitive(runmat_accelerate::graph::PrimitiveOp::ElemPow)
            )
        })
        .map(|step| {
            admitted
                .source_fact(step.inputs[1])
                .expect("exact selected exponent fact is retained")
                .numeric()
                .expect("power control is numeric")
                .class
        })
        .collect::<Vec<_>>();
    assert_eq!(
        power_control_classes,
        vec![
            runmat_types::NumericClass::Double,
            runmat_types::NumericClass::Single
        ],
        "compiled square and final power retain their distinct source classes"
    );
    enum SourceTamper {
        Epsilon,
        Gamma,
        MeanAxes,
    }
    for tamper in [
        SourceTamper::Epsilon,
        SourceTamper::Gamma,
        SourceTamper::MeanAxes,
    ] {
        let mut other_plan = selected_plan.clone();
        let Some(runmat_accelerate::fusion::FusionPattern::ImageNormalize(pattern)) =
            other_plan.group.pattern.as_mut()
        else {
            panic!("selected full image pattern");
        };
        match tamper {
            SourceTamper::Epsilon => {
                pattern.epsilon = runmat_accelerate::fusion::ImageScalar::Constant(0.25)
            }
            SourceTamper::Gamma => {
                pattern.gamma = Some(runmat_accelerate::fusion::ImageScalar::Constant(1.75))
            }
            SourceTamper::MeanAxes => {
                let routes = pattern.reduction_routes.as_mut().unwrap();
                let ImageMeanForm::VecDims { axes, .. } = &mut routes.mu.form else {
                    panic!("full fixture has one vecdim mean");
                };
                axes.swap(0, 1);
            }
        }
        assert!(matches!(
            futures::executor::block_on(ImageSourceInvocation::bind(
                &graph,
                &other_plan,
                &invocation_inputs,
                runmat_accelerate_api::ProviderPrecision::F32,
                None,
            )),
            Err(ImageSourceAdmissionFailure::Source)
        ));
    }
    let output_hooks = admitted
        .requested_output_program()
        .expect("full output suffix has ordered direct provider-hook requests");
    assert!(output_hooks.validate());
    assert_eq!(
        output_hooks
            .calls
            .iter()
            .filter(|call| call.checkpoint)
            .count(),
        3,
        "mean and sigma precede three actual output stores in this fixture"
    );
    assert!(output_hooks
        .calls
        .iter()
        .any(|call| { call.hook == runmat_accelerate_api::ProviderImageOutputHook::ScalarMul }));
    assert!(output_hooks
        .calls
        .iter()
        .any(|call| { call.hook == runmat_accelerate_api::ProviderImageOutputHook::ScalarAdd }));
    assert!(output_hooks
        .calls
        .iter()
        .any(|call| { call.hook == runmat_accelerate_api::ProviderImageOutputHook::ScalarMax }));
    let bound_hooks = output_hooks.clone();
    let sigma_step = source
        .steps
        .iter()
        .position(|step| step.output == producers.sigma)
        .unwrap();
    let output_controls = source.steps[sigma_step + 1..]
        .iter()
        .flat_map(|step| step.inputs.iter().copied())
        .collect::<std::collections::HashSet<_>>();
    let live_scalar_slot = source
        .leaves
        .iter()
        .find_map(|leaf| match leaf {
            ImageSourceLeaf::Live { id }
                if *id != pattern.input && output_controls.contains(id) => selected_plan
                .inputs
                .iter()
                .position(|candidate| candidate == id)
                .filter(|slot| {
                    matches!(&invocation_inputs[*slot], runmat_value::Value::Tensor(tensor) if tensor.len() == 1)
                }),
            _ => None,
        })
        .expect("full source has a live scalar control");
    let mut changed_control = invocation_inputs.clone();
    changed_control[live_scalar_slot] =
        runmat_value::Value::Tensor(runmat_value::Tensor::from_f32(vec![3.0], vec![1, 1]).unwrap());
    if let Ok(rebound) = futures::executor::block_on(ImageSourceInvocation::bind(
        &graph,
        &selected_plan,
        &changed_control,
        runmat_accelerate_api::ProviderPrecision::F32,
        None,
    )) {
        assert_ne!(rebound.requested_output_program(), Some(&bound_hooks));
    }
    assert_eq!(admitted.requested_output_program(), Some(&bound_hooks));
    let source_constant_id = source
        .leaves
        .iter()
        .find_map(|leaf| match leaf {
            ImageSourceLeaf::Constant { id, .. } if selected_plan.inputs.contains(id) => Some(*id),
            _ => None,
        })
        .expect("at least one source constant is an invocation input");
    let source_constant_slot = selected_plan
        .inputs
        .iter()
        .position(|id| *id == source_constant_id)
        .unwrap();
    invocation_inputs[source_constant_slot] = runmat_value::Value::Num(999.0);
    let rejected = futures::executor::block_on(ImageSourceInvocation::bind(
        &graph,
        &selected_plan,
        &invocation_inputs,
        runmat_accelerate_api::ProviderPrecision::F32,
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(rejected, ImageSourceAdmissionFailure::ConstantChanged(id) if id == source_constant_id)
    );
    let mut forged_plan = selected_plan.clone();
    match &mut forged_plan.operations[0] {
        runmat_accelerate::fusion::FusionOp::Primitive { output, .. }
        | runmat_accelerate::fusion::FusionOp::Builtin { output, .. } => *output = None,
    }
    assert!(matches!(
        futures::executor::block_on(ImageSourceInvocation::bind(
            &graph,
            &forged_plan,
            &invocation_inputs,
            runmat_accelerate_api::ProviderPrecision::F32,
            None,
        )),
        Err(ImageSourceAdmissionFailure::Operation(_))
    ));
    let mut forged_cast = graph.clone();
    let cast = source
        .steps
        .iter()
        .find(|step| matches!(&step.label, AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("single")))
        .expect("full source contains selected casts");
    let AccelNodeLabel::Builtin { origin, .. } = &mut forged_cast.nodes[cast.node as usize].label
    else {
        unreachable!("selected cast step")
    };
    *origin = FusionCallOrigin::NamedUnresolved;
    assert!(ImageSourceTranscript::from_graph(&forged_cast, &group).is_none());

    let joined = bytecode.prospective_image_normalize_publications();
    let selected = joined
        .iter()
        .find(|joined| {
            [
                joined.checkpoint(ImageCheckpoint::Mu).unwrap(),
                joined.checkpoint(ImageCheckpoint::Sigma).unwrap(),
            ]
            .into_iter()
            .filter_map(|store| bytecode.var_names.get(&store.binding.index))
            .map(String::as_str)
            .eq(["mu", "sigma"])
        })
        .expect("selected full source joins existing graph, MIR and stores");
    assert_eq!(selected.output_checkpoints().count(), 3);
    assert!(selected.output_checkpoints().all(|store| {
        bytecode
            .var_names
            .get(&store.binding.index)
            .map(String::as_str)
            == Some("out")
    }));
    assert!(selected
        .output_checkpoints()
        .collect::<Vec<_>>()
        .windows(2)
        .all(|pair| pair[0].pc < pair[1].pc));

    let mut unselected_graph = graph.clone();
    let clamp = producers.clamp_output.unwrap();
    let runmat_accelerate::graph::ValueOrigin::NodeOutput { node, .. } =
        &unselected_graph.value(clamp).unwrap().origin
    else {
        panic!("selected clamp must have a graph producer")
    };
    let node = *node as usize;
    let AccelNodeLabel::Builtin { origin, .. } = &mut unselected_graph.nodes[node].label else {
        panic!("selected clamp must be a public builtin")
    };
    *origin = FusionCallOrigin::NamedUnresolved;
    let projection = bytecode.fusion_metadata.stage_projection.as_ref().unwrap();
    let selected = SelectedMultiStoreGraph::for_image(&bytecode).unwrap();
    let region = selected_image_region(&bytecode, projection, &selected, &group).unwrap();
    assert!(reconcile_image_publications(&region, &unselected_graph, &group).is_none());

    let mut forged = bytecode.clone();
    std::sync::Arc::make_mut(forged.fusion_metadata.stage_projection.as_mut().unwrap())
        .statements[0]
        .pc
        .end += 1;
    assert!(forged.prospective_image_normalize_publications().is_empty());

    let mut wrong_owner = bytecode.clone();
    std::sync::Arc::make_mut(
        wrong_owner
            .fusion_metadata
            .stage_projection
            .as_mut()
            .unwrap(),
    )
    .owner = runmat_types::ProgramFunctionId(u32::MAX);
    assert!(wrong_owner
        .prospective_image_normalize_publications()
        .is_empty());

    let mut remapped = bytecode.clone();
    remapped.instructions[0] = Instr::LoadConst(99.0);
    assert!(remapped
        .prospective_image_normalize_publications()
        .is_empty());
}
