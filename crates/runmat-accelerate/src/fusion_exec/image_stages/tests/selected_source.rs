//! Request-bound coverage of the held selected-source constructor. This
//! exercises its real graph/plan binding, not the physical fixture shortcut;
//! it does not mint a production arithmetic or resource proof.

use super::*;
use crate::fusion::{detect_fusion_groups, FusionKind, FusionPattern, FusionPlan};
use crate::graph::{
    AccelGraph, AccelGraphTag, AccelNodeLabel, PrimitiveOp, ShapeInfo, ValueId, ValueOrigin,
};
use runmat_builtins::Type;
use runmat_value::Tensor;

#[path = "selected_source/sigma_operations.rs"]
mod sigma_operation_tests;
#[path = "selected_source/sigma_cutover.rs"]
mod sigma_cutover;
#[path = "selected_source/real_binary_cutover.rs"]
mod real_binary_cutover;
#[path = "selected_source/real_scalar_cast_cutover.rs"]
mod real_scalar_cast_cutover;

fn compiled_image_graph([batch, height, width]: [usize; 3]) -> AccelGraph {
    compiled_image_graph_with_epsilon([batch, height, width], "single(1e-6)")
}

fn compiled_image_graph_with_epsilon(
    [batch, height, width]: [usize; 3],
    epsilon: &str,
) -> AccelGraph {
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    let source = format!(
        r#"
        B={batch}; H={height}; W={width};
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0={epsilon};
        imgs=rand(B,H,W,'single');
        mu=single(mean(imgs,[2 3],'native'));
        sigma=single(sqrt(mean((imgs-mu).^2,[2 3],'native')+eps0));
        out=single(((imgs-mu)./sigma)*gain+bias);
        out=single(max(out,single(0)));
        out=single(out.^gamma);
    "#
    );
    let ast = runmat_parser::parse(source.trim_start()).expect("parse image source");
    let hir =
        runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower image HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower image MIR");
    let entrypoint = hir.assembly.entrypoints[0].id;
    let graph = runmat_vm::compile(&hir.assembly, &mir, entrypoint)
        .expect("compile image source")
        .accel_graph
        .expect("compiled acceleration graph");
    // VM's dependency uses the library copy of this crate, while this unit
    // test uses its test copy. Cross that boundary through the graph's existing
    // serialized structure. Runtime constants are intentionally absent from
    // that wire format; retain them from these exact compiled value records.
    let mut projected: AccelGraph =
        serde_json::from_slice(&serde_json::to_vec(&graph).expect("serialize compiled graph"))
            .expect("decode compiled graph into the test crate");
    assert_eq!(projected.values.len(), graph.values.len());
    for (target, source) in projected.values.iter_mut().zip(&graph.values) {
        assert_eq!(target.id, source.id);
        target.constant = source.constant.clone();
    }
    projected
}

fn signed_literal(graph: &AccelGraph, id: ValueId) -> f64 {
    let info = graph.value(id).expect("compiled control operand");
    if let Some(Value::Num(number)) = &info.constant {
        return *number;
    }
    let ValueOrigin::NodeOutput { node, .. } = info.origin else {
        panic!("control operand is not a compiled numeric literal");
    };
    let producer = graph.node(node).expect("signed literal producer");
    assert!(matches!(
        producer.label,
        AccelNodeLabel::Primitive(PrimitiveOp::Neg)
    ));
    let [operand] = producer.inputs.as_slice() else {
        panic!("literal negation must have one operand");
    };
    let Some(Value::Num(number)) = &graph.value(*operand).unwrap().constant else {
        panic!("literal negation must consume a compiled numeric literal");
    };
    -*number
}

fn exact_source_input(graph: &AccelGraph, id: ValueId, group_start: usize) -> Value {
    let info = graph.value(id).expect("selected external value");
    // Literal arrays are represented as folded producer outputs, not only
    // Constant origins. Use their exact compiled payload as the control value.
    if let Some(constant) = &info.constant {
        return constant.clone();
    }
    let node = match &info.origin {
        ValueOrigin::NodeOutput { node, .. } => *node,
        ValueOrigin::Variable { kind, index } => {
            let selected = crate::graph::VarBinding {
                kind: *kind,
                index: *index,
            };
            graph
                .nodes
                .iter()
                .filter(|node| {
                    node.span.end < group_start && graph.node_binding(node.id) == Some(&selected)
                })
                .max_by_key(|node| node.span.end)
                .expect("selected variable has a prior bound producer")
                .id
        }
        _ => panic!("selected live control {id} has no compiled producer"),
    };
    let producer = graph.node(node).expect("compiled control producer");
    assert!(matches!(
        &producer.label,
        AccelNodeLabel::Builtin { name, .. } if name == "single"
    ));
    let [source] = producer.inputs.as_slice() else {
        panic!("single control must have one literal operand");
    };
    Value::Tensor(
        Tensor::from_f32(vec![signed_literal(graph, *source) as f32], vec![1, 1]).unwrap(),
    )
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_constructor_uses_bound_compiled_source_and_rejects_changed_view() {
    selected_constructor_for_shape([2, 3, 4]).await;
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_constructor_uses_bound_compiled_source_with_larger_reduction_axes() {
    selected_constructor_for_shape([2, 4, 5]).await;
}

async fn selected_constructor_for_shape(shape: [usize; 3]) {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let graph = compiled_image_graph(shape);
    let groups = detect_fusion_groups(&graph);
    let plans = FusionPlan::from_graph(&graph, &groups);
    let plan = plans
        .groups
        .iter()
        .find(|plan| plan.group.kind == FusionKind::ImageNormalize)
        .unwrap_or_else(|| {
            panic!(
                "compiled full image source has no selected group; nodes={:?}; values={:?}",
                graph.nodes, graph.values
            )
        });
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref().unwrap() else {
        panic!("selected image pattern");
    };
    let input = provider
        .allocations
        .allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .expect("owned image input");
    let other = provider
        .allocations
        .allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .expect("distinct same-shaped selected-owner view");
    let values = plan
        .inputs
        .iter()
        .map(|id| {
            if *id == pattern.input {
                Value::GpuTensor(input.handle.clone())
            } else {
                exact_source_input(&graph, *id, plan.group.span.start)
            }
        })
        .collect::<Vec<_>>();
    let folded = graph
        .nodes
        .iter()
        .find(|node| {
            node.span.start > plan.group.span.start
                && node.tags.contains(&AccelGraphTag::FoldedLiteralArray)
        })
        .expect("compiled mean axes have a mid-span literal producer");
    let folded_id = folded.outputs[0];
    assert!(pattern
        .bound_mean_axes
        .iter()
        .any(|(id, _)| *id == folded_id));
    let folded_node = folded.id as usize;
    let transcript = |graph: &AccelGraph| {
        crate::fusion::image_reduction::ImageSourceTranscript::from_graph(graph, &plan.group)
    };
    assert!(
        transcript(&graph).is_some(),
        "compiled selected source transcript rejected: group={:?}; nodes={:?}",
        plan.group,
        graph.nodes
    );
    let mut missing_tag = graph.clone();
    missing_tag.nodes[folded_node].tags.clear();
    assert!(transcript(&missing_tag).is_none());
    let mut wrong_shape = graph.clone();
    wrong_shape.values[folded_id as usize].shape = ShapeInfo::Tensor(vec![Some(2), Some(1)]);
    assert!(transcript(&wrong_shape).is_none());
    let mut wrong_type = graph.clone();
    wrong_type.values[folded_id as usize].ty = Type::Num;
    assert!(transcript(&wrong_type).is_none());
    let mut wrong_value = graph.clone();
    wrong_value.values[folded_id as usize].constant = Some(Value::Tensor(
        Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap(),
    ));
    assert!(transcript(&wrong_value).is_none());
    let mut wrong_element = graph.clone();
    wrong_element.values[folded.inputs[0] as usize].constant = Some(Value::Num(9.0));
    assert!(transcript(&wrong_element).is_none());
    let mut late_producer = graph.clone();
    late_producer.nodes[folded_node].span.end = plan.group.span.end;
    assert!(transcript(&late_producer).is_none());
    let folded_zero = graph
        .nodes
        .iter()
        .find(|node| {
            node.span.start > plan.group.span.start
                && node.tags.contains(&AccelGraphTag::UnaryScalarFold)
                && !plan.group.nodes.contains(&node.id)
        })
        .expect("clamp has a source-folded zero control");
    assert_eq!(
        pattern.clamp_zero_value,
        Some(folded_zero.outputs[0]),
        "a published folded clamp remains a live selected input"
    );
    let zero_position = plan
        .inputs
        .iter()
        .position(|id| *id == folded_zero.outputs[0])
        .expect("published clamp has an actual request binding");
    let mut changed_zero = values.clone();
    changed_zero[zero_position] = Value::Tensor(Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap());
    assert!(matches!(
        ImageSourceInvocation::bind(&graph, plan, &changed_zero, ProviderPrecision::F64, None)
            .await,
        Err(crate::fusion::image_reduction::ImageSourceAdmissionFailure::ConstantChanged(id))
            if id == folded_zero.outputs[0]
    ));
    let mut forged_zero = graph.clone();
    forged_zero.values[folded_zero.outputs[0] as usize].constant = Some(Value::Tensor(
        Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
    ));
    assert!(transcript(&forged_zero).is_none());
    let mut wrong_zero_class = graph.clone();
    wrong_zero_class.values[folded_zero.outputs[0] as usize].constant = Some(Value::Num(0.0));
    assert!(transcript(&wrong_zero_class).is_none());
    let mut wrong_zero_type = graph.clone();
    wrong_zero_type.values[folded_zero.outputs[0] as usize].ty = Type::Num;
    assert!(transcript(&wrong_zero_type).is_none());
    let mut rebound = values.clone();
    let position = plan
        .inputs
        .iter()
        .position(|id| *id == folded_id)
        .expect("folded axes are an actual request leaf");
    rebound[position] = Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap());
    assert!(matches!(
        ImageSourceInvocation::bind(&graph, plan, &rebound, ProviderPrecision::F64, None).await,
        Err(crate::fusion::image_reduction::ImageSourceAdmissionFailure::ConstantChanged(id)) if id == folded_id
    ));
    let invocation =
        ImageSourceInvocation::bind(&graph, plan, &values, ProviderPrecision::F64, None)
            .await
            .expect("same compiled source and exact live bindings");
    let Some(crate::fusion::ImageScalar::Value(gamma_id)) = pattern.gamma.as_ref() else {
        panic!("compiled gamma is a live selected control");
    };
    let gamma_position = plan
        .inputs
        .iter()
        .position(|id| id == gamma_id)
        .expect("compiled gamma has an actual request binding");
    let mut changed_gamma = values.clone();
    changed_gamma[gamma_position] = Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    let rebound_gamma =
        ImageSourceInvocation::bind(&graph, plan, &changed_gamma, ProviderPrecision::F64, None)
            .await
            .expect("a changed live gamma is rebound through selected power arithmetic");
    let (_, rebound_source) = rebound_gamma
        .selected_staged_mean_power(&input.handle)
        .expect("changed live gamma retains an exact selected source");
    assert_eq!(rebound_source.final_power.exponent_value, *gamma_id);
    assert_eq!(
        rebound_source.final_power.represented_exponent.to_bits(),
        2.0f64.to_bits()
    );
    let rebound_output = graph
        .node(rebound_source.final_power.node)
        .expect("selected final POWER producer")
        .outputs[0];
    assert_eq!(
        rebound_gamma
            .selected_power(rebound_output)
            .expect("rebound POWER arithmetic")
            .arithmetic
            .exponent_scalar,
        Some(runmat_accelerate_api::PowerScalarControl::SingleBits(
            2.0f32.to_bits()
        ))
    );
    assert_ne!(
        rebound_gamma.semantic_identity(),
        invocation.semantic_identity()
    );
    let source = transcript(&graph).expect("selected compiled transcript");
    let div = source
        .steps
        .iter()
        .find(|step| matches!(&step.label, AccelNodeLabel::Primitive(PrimitiveOp::ElemDiv)))
        .expect("compiled selected division");
    let mut changed_binary = graph.clone();
    changed_binary.nodes[div.node as usize].inputs.swap(0, 1);
    assert!(
        ImageSourceInvocation::bind(&changed_binary, plan, &values, ProviderPrecision::F64, None)
            .await
            .is_err(),
        "changed source operand order cannot reuse the selected binary certificate"
    );
    let powers: Vec<_> = source
        .steps
        .iter()
        .filter(|step| matches!(&step.label, AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)))
        .map(|step| {
            invocation
                .selected_power(step.output)
                .expect("selected arithmetic descriptor")
        })
        .collect();
    assert_eq!(powers.len(), 2);
    assert_eq!(
        powers[0].arithmetic.base_id,
        source
            .steps
            .iter()
            .find(|step| step.output == powers[0].output)
            .unwrap()
            .inputs[0]
    );
    assert_eq!(
        powers[0].arithmetic.mode,
        runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: runmat_accelerate_api::RealPowerOperand::Base
        }
    );
    assert_eq!(
        powers[0].arithmetic.exponent_scalar,
        Some(
            runmat_runtime::builtins::common::spec::FusionScalarControl::DoubleBits(
                2.0f64.to_bits()
            )
        )
    );
    assert_eq!(
        powers[1].arithmetic.mode,
        runmat_accelerate_api::RealPowerArithmeticMode::NativeSingle
    );
    assert_eq!(
        powers[1].arithmetic.exponent_scalar,
        Some(
            runmat_runtime::builtins::common::spec::FusionScalarControl::SingleBits(
                1.8f32.to_bits()
            )
        )
    );
    assert!(
        invocation.requested_output_program().is_some(),
        "source output projection rejected; canonical steps={:?}",
        source
            .steps
            .iter()
            .map(|step| (
                step.output,
                &step.label,
                invocation.source_fact(step.output)
            ))
            .collect::<Vec<_>>()
    );
    let binary = invocation
        .selected_binary_calls()
        .expect("sealed exact real binary source calls");
    assert_eq!(binary.semantic_identity(), invocation.semantic_identity());
    assert_eq!(
        binary.sigma_center().operation(),
        runmat_accelerate_api::ProviderRealBinaryOperation::Sub
    );
    assert_eq!(
        binary.sigma_center().output(),
        u64::from(
            source
                .steps
                .iter()
                .find(|step| matches!(&step.label, AccelNodeLabel::Primitive(PrimitiveOp::ElemPow)))
                .unwrap()
                .inputs[0]
        )
    );
    assert_eq!(
        binary.output_sub().operation(),
        runmat_accelerate_api::ProviderRealBinaryOperation::Sub
    );
    assert_eq!(
        binary.output_div().operation(),
        runmat_accelerate_api::ProviderRealBinaryOperation::Div
    );
    assert_eq!(
        binary.output_sub().inputs()[1],
        binary.sigma_center().inputs()[1]
    );
    assert_eq!(
        binary.output_div().inputs()[0],
        binary.output_sub().output()
    );
    let exact_moments = invocation
        .requested_staged_workload()
        .unwrap()
        .moment_program
        .as_ref()
        .unwrap();
    assert_eq!(
        binary.sigma_center().inputs(),
        [
            exact_moments.mu[0].source_input,
            exact_moments.mu.last().unwrap().public_output,
        ]
    );
    assert_eq!(binary.output_sub().inputs(), binary.sigma_center().inputs());
    assert_eq!(
        binary.output_div().inputs()[1],
        exact_moments.sigma.public_output
    );
    let binary_wire = invocation
        .requested_output_program()
        .unwrap()
        .calls
        .iter()
        .filter(|call| {
            matches!(
                call.hook,
                runmat_accelerate_api::ProviderImageOutputHook::ElemSub
                    | runmat_accelerate_api::ProviderImageOutputHook::ElemDiv
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(binary_wire.len(), 2);
    assert_eq!(
        binary_wire[0].operands.as_slice(),
        [
            runmat_accelerate_api::ProviderImageOutputOperand::Input,
            runmat_accelerate_api::ProviderImageOutputOperand::Mean,
        ]
    );
    assert_eq!(
        binary_wire[1].operands.as_slice(),
        [
            runmat_accelerate_api::ProviderImageOutputOperand::Produced(
                binary.output_sub().output()
            ),
            runmat_accelerate_api::ProviderImageOutputOperand::Sigma,
        ]
    );
    let session = ImageNormalizeStageSession::begin_selected(
        provider,
        &input.handle,
        &invocation,
        ImageArithmeticProof::fixture(),
    )
    .expect("held selected admission")
    .expect("request-bound staged source");
    assert!(session.program.output_program.is_some());
    assert!(session.selected_binary_plans.is_some());
    let workload = invocation
        .requested_staged_workload()
        .expect("frozen entire selected workload");
    let selected = workload
        .selected_powers
        .as_ref()
        .expect("typed selected POWER program");
    assert_eq!(
        selected.square.control,
        runmat_accelerate_api::PowerScalarControl::DoubleBits(2.0_f64.to_bits())
    );
    assert_eq!(selected.square.mode, powers[0].arithmetic.mode);
    assert_eq!(
        selected.final_power.control,
        runmat_accelerate_api::PowerScalarControl::SingleBits(1.8_f32.to_bits())
    );
    assert_eq!(selected.final_power.mode, powers[1].arithmetic.mode);
    let moments = workload
        .moment_program
        .as_ref()
        .expect("selected moment publications");
    assert_eq!(
        (moments.mu[0].raw_output, moments.mu[0].public_output),
        (18, 19)
    );
    assert_eq!(
        (moments.sigma.raw_output, moments.sigma.public_output),
        (29, 30)
    );
    assert_eq!(moments.conversion_calls(), 2);
    let selected_casts = invocation
        .selected_moment_bindings()
        .expect("sealed source conversion bindings");
    sigma_operation_tests::require_selected_operations(
        &invocation,
        &session,
        plan,
        &values,
        &source,
    );
    assert_eq!(
        selected_casts.semantic_identity(),
        invocation.semantic_identity()
    );
    assert_eq!(
        selected_casts.mu()[0].len(),
        moments.mu[0].conversions.len()
    );
    assert_eq!(
        selected_casts.sigma().len(),
        moments.sigma.conversions.len()
    );
    assert_eq!(
        selected_casts.sigma_raw().binding().builtin,
        runmat_builtins::SQRT_CATALOG_ENTRY.identity
    );
    assert_eq!(
        selected_casts.mu()[0][0].source_input(),
        moments.mu[0].raw_output
    );
    assert_eq!(
        selected_casts.mu()[0][0].source_output(),
        moments.mu[0].public_output
    );
    assert_eq!(
        selected_casts.mu()[0][0].binding().builtin,
        runmat_builtins::SINGLE_CATALOG_ENTRY.identity
    );
    assert_eq!(session.program, *workload);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.cache_reservations.load(Ordering::SeqCst), 0);
    assert!(provider.trace().is_empty());
    assert!(ImageNormalizeStageSession::begin_selected(
        provider,
        &other.handle,
        &invocation,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .is_none());
    input.probe.mark_unavailable();
    assert!(ImageNormalizeStageSession::begin_selected(
        provider,
        &input.handle,
        &invocation,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .is_none());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.trace().is_empty());
}
