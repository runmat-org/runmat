use super::*;

#[test]
fn selected_admission_work_distinguishes_host_proof_from_entered_provider_probe() {
    use runmat_runtime::builtins::common::spec::{FusionAdmissionWork, FusionProofStage};

    let entry = &runmat_builtins::POWER_CATALOG_ENTRY;
    let binding = entry.binding_identity(&entry.bindings[0]);
    let host = FusionAdmissionWork::DomainProof {
        binding,
        provider_precision: ProviderPrecision::F32,
        stages: vec![FusionProofStage::HostScan { elements: 1 }],
    };
    assert!(!host.provider_effect_observed());
    let readback = FusionAdmissionWork::DomainProof {
        binding,
        provider_precision: ProviderPrecision::F32,
        stages: vec![FusionProofStage::ScalarReadback { elements: 1 }],
    };
    assert!(readback.provider_effect_observed());
    let paired = FusionAdmissionWork::ProviderProbe {
        binding,
        precision: ProviderPrecision::F32,
        provider_precision: ProviderPrecision::F32,
        inputs: Vec::new(),
    };
    assert!(paired.provider_effect_observed());
}

#[test]
fn runtime_admission_failure_retains_identifier_source_and_no_replay_policy() {
    let runtime = runmat_runtime::build_runtime_error("domain probe failed")
        .with_identifier("RunMat:acosh:Internal")
        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
        .with_source(std::io::Error::other("injected provider failure"))
        .build();
    let execution = FusionResolveFailure::from(runtime)
        .into_execution_or_structural()
        .expect_err("runtime failure must not select shared-runtime fallback");
    let original = std::error::Error::source(&execution)
        .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
        .expect("original typed runtime error must survive provider boundary");
    assert_eq!(original.identifier(), Some("RunMat:acosh:Internal"));
    assert_eq!(
        original.gpu_gather_retry(),
        runmat_runtime::GpuGatherRetry::Never
    );
    assert!(original
        .source
        .as_ref()
        .is_some_and(|source| source.to_string().contains("injected provider failure")));
}

#[test]
fn resident_probe_failure_retains_nested_effect_disposition() {
    let provider = runmat_accelerate_api::ProviderExecutionError::from_source(
        std::io::Error::other("injected probe completion failure"),
    )
    .after_effect();
    let runtime = runmat_runtime::build_runtime_error("domain probe failed")
        .with_identifier("RunMat:acosh:Internal")
        .with_source(provider)
        .build();
    let execution = FusionResolveFailure::from(runtime)
        .after_observed_probe()
        .into_execution_or_structural()
        .expect_err("an observed resident probe cannot authorize replay");
    assert!(execution.is_after_effect());
    assert!(!execution.is_unsupported());
}

#[test]
fn selected_pairwise_zero_clamp_evidence_crosses_typed_cast_to_fractional_power() {
    runmat_gc::gc_test_context(|| {
        let mut source = plan(
            vec![
                FusionOp::Builtin {
                    origin: Default::default(),
                    name: "max".into(),
                    inputs: vec![0, 1],
                    output: Some(2),
                },
                cast("single", 2, 3),
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs: vec![3, 5],
                    output: Some(4),
                },
            ],
            4,
        );
        source.const_values.insert(
            5,
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.5], vec![1, 1]).unwrap()),
        );
        let signal = Value::Tensor(
            runmat_value::Tensor::from_f32(vec![-4.0, f32::NAN], vec![1, 2]).unwrap(),
        );
        let zero = Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
        let program = resolve_policy(
            &source,
            &[signal.clone(), zero],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .expect("bound clamp domain");
        assert_eq!(
            program.evidence.get(&4),
            Some(&runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                runmat_accelerate_api::PowerResultDomain::Real,
            ))
        );
        let negative_zero =
            Value::Tensor(runmat_value::Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap());
        assert!(resolve_policy(
            &source,
            &[signal.clone(), negative_zero],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .is_err());
        let positive_nonzero =
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
        assert!(resolve_policy(
            &source,
            &[signal, positive_nonzero],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .is_err());
        let mut no_clamp = plan(
            vec![
                cast("single", 0, 3),
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs: vec![3, 5],
                    output: Some(4),
                },
            ],
            4,
        );
        no_clamp.const_values = source.const_values.clone();
        assert!(resolve_policy(
            &no_clamp,
            &[
                Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![-4.0, f32::NAN], vec![1, 2]).unwrap(),
                ),
                Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
            ],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .is_err());
        let mut different_binding = source.clone();
        let FusionOp::Builtin { name, .. } = &mut different_binding.operations[0] else {
            unreachable!("selected pairwise builtin fixture")
        };
        *name = "min".into();
        assert!(resolve_policy(
            &different_binding,
            &[
                Value::Tensor(runmat_value::Tensor::from_f32(vec![-4.0], vec![1, 1]).unwrap(),),
                Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
            ],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .is_err());
    });
}

#[test]
fn inverse_hyperbolic_chain_admits_exact_host_domain_and_double_output() {
    runmat_gc::gc_test_context(|| {
        let add = |left, right, output| FusionOp::Primitive {
            op: PrimitiveOp::Add,
            inputs: vec![left, right],
            output: Some(output),
        };
        let mut source = plan(
            vec![
                cast("asinh", 0, 2),
                cast("atanh", 0, 3),
                add(2, 3, 4),
                add(0, 1, 5),
                cast("acosh", 5, 6),
                add(4, 6, 7),
            ],
            7,
        );
        source.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(3)]);
        let values = [
            Value::Tensor(runmat_value::Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).unwrap()),
            Value::Num(1.0),
        ];
        let program = resolve(&source, &values).expect("represented inverse-domain admission");
        assert_eq!(program.floating_class(7).unwrap(), NumericClass::Double);
        source
            .generate_typed_wgsl(&[7], &program, ScalarType::F64)
            .unwrap();
    });
}

#[test]
fn real_inputs_do_not_fabricate_power_domain_evidence() {
    let plan = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    assert!(resolve(&plan, &[Value::Num(-1.0), Value::Num(0.5)]).is_err());
}

#[test]
fn domain_rejection_retains_completed_admission_work_without_gpu_success() {
    use crate::placement::{begin_trace, report, PlacementEventKind};
    let plan = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let trace = begin_trace("fusion.power_complex_admission", None);
    let correlation = trace.correlation();
    let result = futures::executor::block_on(ResolvedProgram::resolve(
        &plan,
        &[Value::Num(-1.0), Value::Num(0.5)],
        ProviderPrecision::F64,
        Some(correlation),
    ));
    assert!(result.is_err());
    trace.complete(None, None);
    let report = report();
    let recorded = report
        .traces
        .iter()
        .find(|trace| trace.correlation == correlation)
        .unwrap();
    let event = recorded
        .events
        .iter()
        .find(|event| {
            event.kind == PlacementEventKind::Prepare
                && event.reason.as_deref() == Some("domain_host_preparation")
        })
        .unwrap();
    assert_eq!(event.reason.as_deref(), Some("domain_host_preparation"));
    assert!(event.duration_ns.is_some());
    assert!(event
        .attributes
        .iter()
        .any(|attribute| attribute.key == "completed" && attribute.value == 1));
    let leaf_preparation = recorded
        .events
        .iter()
        .find(|event| {
            event.kind == PlacementEventKind::Prepare
                && event.reason.as_deref() == Some("fusion_leaf_index_wall")
        })
        .expect("structural preparation precedes domain rejection");
    assert!(leaf_preparation.duration_ns.is_some());
    assert!(leaf_preparation
        .attributes
        .iter()
        .any(|attribute| attribute.key == "completed" && attribute.value == 1));
    assert!(!recorded.events.iter().any(|event| matches!(
        event.kind,
        PlacementEventKind::Upload | PlacementEventKind::Kernel
    )));
}
