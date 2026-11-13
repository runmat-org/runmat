use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, ShapeInfo};

fn new_resolver<'a>(plan: &'a FusionGroupPlan, inputs: &[&'a Value]) -> RealRangeResolver<'a> {
    let leaves = FusionInvocationLeaves::new(plan, inputs).expect("valid invocation leaves");
    RealRangeResolver::new(plan, &leaves).expect("bounded source geometry")
}

fn plan(operations: Vec<FusionOp>, output: ValueId) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 0 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(output),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

fn add_op(left: ValueId, right: ValueId, output: ValueId) -> FusionOp {
    FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![left, right],
        output: Some(output),
    }
}

#[test]
fn indexed_walk_handles_deep_chains_shared_dags_and_cycles() {
    let mut operations = Vec::new();
    for output in 2..4098 {
        operations.push(add_op(if output == 2 { 0 } else { output - 1 }, 1, output));
    }
    let deep = plan(operations, 4097);
    let values = [Value::Num(1.0), Value::Num(0.0)];
    let inputs: Vec<_> = values.iter().collect();
    let mut resolver = new_resolver(&deep, &inputs);
    assert!(resolver.is_valid());
    let (completed, examined) = resolver.visit(4097, &deep);
    assert!(completed.is_some());
    assert_eq!(resolver.visit_order.len(), 4098);
    assert!(examined >= resolver.visit_order.len() as u64);
    assert_eq!(resolver.visit_order.last(), Some(&4097));

    let shared = plan(vec![add_op(0, 1, 2), add_op(2, 2, 3)], 3);
    let mut resolver = new_resolver(&shared, &inputs);
    assert!(resolver.visit(3, &shared).0.is_some());
    assert_eq!(
        resolver.visit_order.len(),
        4,
        "shared producer is traversed only once"
    );

    let cyclic = plan(vec![add_op(3, 0, 2), add_op(2, 0, 3)], 3);
    let mut resolver = new_resolver(&cyclic, &inputs);
    assert!(resolver.visit(3, &cyclic).0.is_none());
    assert!(resolver.visit(99, &cyclic).0.is_none());
    let duplicate = plan(vec![add_op(0, 1, 2), add_op(0, 1, 2)], 2);
    assert!(FusionInvocationLeaves::new(&duplicate, &inputs).is_err());
    let leaf_collision = plan(vec![add_op(0, 1, 0)], 0);
    assert!(FusionInvocationLeaves::new(&leaf_collision, &inputs).is_err());
}

#[test]
fn indexed_work_counts_unique_invocation_leaves_not_shadow_constants() {
    let mut source = plan(vec![add_op(0, 1, 2)], 2);
    source.const_values.insert(0, Value::Num(99.0));
    let values = [Value::Num(1.0), Value::Num(2.0)];
    let inputs: Vec<_> = values.iter().collect();
    let resolver = new_resolver(&source, &inputs);
    let mut work = FusionAdmissionWork::None;
    resolver
        .report_index(
            &mut work,
            BuiltinBindingIdentity {
                builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
                variant: "default",
            },
            ProviderPrecision::F64,
        )
        .unwrap();
    assert!(resolver.is_valid());
    let FusionAdmissionWork::DomainProof { stages, .. } = work else {
        panic!("valid index must report preparation work");
    };
    assert_eq!(
        stages,
        vec![FusionProofStage::GraphIndex {
            operations: 1,
            leaves: 2
        }]
    );
}

#[test]
fn shared_leaf_is_scanned_once_and_reused_without_fabricated_work() {
    let plan = plan(vec![add_op(0, 1, 2), add_op(2, 2, 3)], 3);
    let values = [Value::Num(1.25), Value::Num(0.25)];
    let inputs: Vec<_> = values.iter().collect();
    let fact = runmat_runtime::value_fact::value_fact(&values[0]);
    let facts = HashMap::from([
        (0, fact.clone()),
        (1, fact.clone()),
        (2, fact.clone()),
        (3, fact),
    ]);
    let mut resolver = new_resolver(&plan, &inputs);
    let snapshots = vec![None; plan.operations.len()];
    let binding = BuiltinBindingIdentity {
        builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
        variant: "default",
    };
    let mut work = FusionAdmissionWork::None;
    let first = futures::executor::block_on(prove(
        RealDomainProofContext {
            plan: &plan,
            snapshots: &snapshots,
            facts: &facts,
            binding,
            internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
            provider_precision: ProviderPrecision::F64,
        },
        &mut resolver,
        2,
        RealUnaryPredicate::AtLeastOne,
        &mut work,
    ))
    .unwrap()
    .unwrap();
    assert_eq!(first.value_id, 2);
    let FusionAdmissionWork::DomainProof { stages, .. } = work else {
        panic!("host proof must report work");
    };
    assert_eq!(stages.len(), 4);
    assert!(matches!(stages[0], FusionProofStage::GraphWalk { .. }));
    assert_eq!(
        stages
            .iter()
            .filter(|stage| matches!(stage, FusionProofStage::HostScan { elements: 1 }))
            .count(),
        2
    );
    assert!(matches!(
        stages[3],
        FusionProofStage::RangePropagation { operands: 2 }
    ));

    let mut reused_work = FusionAdmissionWork::None;
    let second = futures::executor::block_on(prove(
        RealDomainProofContext {
            plan: &plan,
            snapshots: &snapshots,
            facts: &facts,
            binding,
            internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
            provider_precision: ProviderPrecision::F64,
        },
        &mut resolver,
        3,
        RealUnaryPredicate::AtLeastOne,
        &mut reused_work,
    ))
    .unwrap()
    .unwrap();
    assert_eq!(second.value_id, 3);
    let FusionAdmissionWork::DomainProof { stages, .. } = reused_work else {
        panic!("reused range still reports graph and arithmetic work");
    };
    assert_eq!(stages.len(), 2);
    assert!(matches!(stages[0], FusionProofStage::GraphWalk { .. }));
    assert!(matches!(
        stages[1],
        FusionProofStage::RangePropagation { operands: 2 }
    ));
}

#[test]
fn many_requirements_reuse_one_leaf_index_and_cached_ranges() {
    let mut plan = plan(vec![add_op(0, 1, 512)], 512);
    plan.inputs = (0..512).collect();
    let values = vec![Value::Num(1.0); 512];
    let inputs: Vec<_> = values.iter().collect();
    let fact = runmat_runtime::value_fact::value_fact(&values[0]);
    let mut facts: HashMap<_, _> = (0..512).map(|id| (id, fact.clone())).collect();
    facts.insert(512, fact);
    let mut resolver = new_resolver(&plan, &inputs);
    assert_eq!(resolver.leaves.len(), 512);
    let binding = BuiltinBindingIdentity {
        builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
        variant: "default",
    };
    let snapshots = [None];
    let mut retained_stages = 0usize;
    for requirement in 0..512 {
        let mut work = FusionAdmissionWork::None;
        if requirement == 0 {
            resolver
                .report_index(&mut work, binding, ProviderPrecision::F64)
                .unwrap();
        }
        let evidence = futures::executor::block_on(prove(
            RealDomainProofContext {
                plan: &plan,
                snapshots: &snapshots,
                facts: &facts,
                binding,
                internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
                provider_precision: ProviderPrecision::F64,
            },
            &mut resolver,
            512,
            RealUnaryPredicate::AtLeastOne,
            &mut work,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(evidence.value_id, 512);
        let FusionAdmissionWork::DomainProof { stages, .. } = work else {
            panic!("proof must report work");
        };
        retained_stages += stages.len();
        if requirement == 0 {
            assert!(matches!(
                stages[0],
                FusionProofStage::GraphIndex {
                    operations: 1,
                    leaves: 512
                }
            ));
            assert_eq!(stages.len(), 5);
        } else {
            assert_eq!(stages.len(), 1, "cached proof does not rescan leaves");
            assert!(matches!(stages[0], FusionProofStage::GraphWalk { .. }));
        }
    }
    assert!(retained_stages <= proof_stage_upper(512, 1, 512).unwrap());
}

#[test]
fn unproved_operation_remains_unknown_with_actual_work_recorded() {
    let mut unsupported = add_op(0, 1, 2);
    if let FusionOp::Primitive { op, .. } = &mut unsupported {
        *op = PrimitiveOp::Sub;
    }
    let plan = plan(vec![unsupported], 2);
    let values = [Value::Num(2.0), Value::Num(0.5)];
    let inputs: Vec<_> = values.iter().collect();
    let fact = runmat_runtime::value_fact::value_fact(&values[0]);
    let facts = HashMap::from([(0, fact.clone()), (1, fact.clone()), (2, fact)]);
    let mut resolver = new_resolver(&plan, &inputs);
    let binding = BuiltinBindingIdentity {
        builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
        variant: "default",
    };
    let mut work = FusionAdmissionWork::None;
    assert!(futures::executor::block_on(prove(
        RealDomainProofContext {
            plan: &plan,
            snapshots: &[None],
            facts: &facts,
            binding,
            internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
            provider_precision: ProviderPrecision::F64,
        },
        &mut resolver,
        2,
        RealUnaryPredicate::AtLeastOne,
        &mut work,
    ))
    .unwrap()
    .is_none());
    let FusionAdmissionWork::DomainProof { stages, .. } = work else {
        panic!("unknown operation still consumed proof work");
    };
    assert_eq!(stages.len(), 4);
    assert!(matches!(
        stages[3],
        FusionProofStage::RangePropagation { operands: 2 }
    ));
}

#[test]
fn represented_single_addition_is_not_a_double_precision_domain_proof() {
    let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
    let half_ulp = f32::EPSILON / 4.0;
    let below_half = f32::from_bits(half_ulp.to_bits() - 1);
    let above_half = f32::from_bits(half_ulp.to_bits() + 1);
    let mode = runmat_types::RealFloatingAddMode::Single;
    let sum = mode.add_represented(below_one, f64::from(below_half));
    assert_eq!(sum, below_one, "just below the tie stays below one");
    assert_eq!(mode.add_represented(below_one, f64::from(half_ulp)), 1.0);
    assert_eq!(mode.add_represented(below_one, f64::from(above_half)), 1.0);
    assert_eq!(
        classification(
            RealRange {
                minimum: sum,
                maximum: sum,
                class: NumericClass::Single,
                resident_arithmetic: false,
            },
            RealUnaryPredicate::AtLeastOne,
        ),
        RealUnaryClassification::Indeterminate,
    );
}

#[test]
fn mixed_single_tensor_double_scalar_addition_proves_only_the_represented_side_of_boundary() {
    let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
    let half_ulp = f64::from(f32::EPSILON) / 4.0;
    let single = Value::Tensor(
        runmat_value::Tensor::from_f32(vec![below_one as f32], vec![1, 1])
            .expect("native single tensor"),
    );
    let binding = BuiltinBindingIdentity {
        builtin: runmat_builtins::ACOSH_CATALOG_ENTRY.identity,
        variant: "default",
    };
    for scalar_value in [half_ulp - f64::EPSILON, half_ulp + f64::EPSILON] {
        for reverse in [false, true] {
            let scalar = Value::Num(scalar_value);
            let values = if reverse {
                [scalar, single.clone()]
            } else {
                [single.clone(), scalar]
            };
            let inputs: Vec<_> = values.iter().collect();
            let left = runmat_runtime::value_fact::value_fact(&values[0]);
            let right = runmat_runtime::value_fact::value_fact(&values[1]);
            for named in [false, true] {
                let operation = if named {
                    FusionOp::Builtin {
                        origin: crate::graph::FusionCallOrigin::named("plus"),
                        name: "plus".into(),
                        inputs: vec![0, 1],
                        output: Some(2),
                    }
                } else {
                    add_op(0, 1, 2)
                };
                let selected = if named {
                    crate::fusion::semantic_bindings::SemanticBinding::capture(&operation)
                        .expect("canonical plus capture")
                } else {
                    None
                };
                let output_fact = if let Some(selected) = selected {
                    let inferred = runmat_builtins::infer_catalog_call(
                        selected.entry,
                        &runmat_types::CallRequest {
                            arguments: vec![left.clone(), right.clone()],
                            literals: runmat_types::LiteralContext::default(),
                            outputs: runmat_types::OutputSelection::default(),
                        },
                    );
                    assert!(inferred.diagnostics.is_empty());
                    inferred
                        .outputs
                        .into_iter()
                        .next()
                        .expect("one plus result")
                } else {
                    let inferred =
                        runmat_types::infer_binary(runmat_types::OperatorKind::Add, &left, &right);
                    assert!(inferred.diagnostics.is_empty());
                    inferred.fact
                };
                assert_eq!(
                    output_fact.numeric().map(|numeric| numeric.class),
                    Some(NumericClass::Single)
                );
                let facts =
                    HashMap::from([(0, left.clone()), (1, right.clone()), (2, output_fact)]);
                let plan = plan(vec![operation], 2);
                let mut resolver = new_resolver(&plan, &inputs);
                let mut work = FusionAdmissionWork::None;
                let evidence = futures::executor::block_on(prove(
                    RealDomainProofContext {
                        plan: &plan,
                        snapshots: &[selected],
                        facts: &facts,
                        binding,
                        internal_error: &runmat_builtins::ACOSH_ERROR_INTERNAL,
                        provider_precision: ProviderPrecision::F64,
                    },
                    &mut resolver,
                    2,
                    RealUnaryPredicate::AtLeastOne,
                    &mut work,
                ))
                .unwrap()
                .expect("represented Add range");
                let NumericalDomainEvidence::RealUnary { classification, .. } =
                    evidence.numerical_domain
                else {
                    panic!("represented Add must produce real-unary evidence")
                };
                assert_eq!(
                    classification,
                    if scalar_value < half_ulp {
                        RealUnaryClassification::Indeterminate
                    } else {
                        RealUnaryClassification::Inside
                    }
                );
                let FusionAdmissionWork::DomainProof { stages, .. } = work else {
                    panic!("represented Add must report proof work")
                };
                assert_eq!(stages.len(), 4);
                assert!(matches!(
                    stages[3],
                    FusionProofStage::RangePropagation { operands: 2 }
                ));
                if named {
                    assert!(
                        operation_range(&plan.operations[0], None, &facts, &resolver.ranges, 2,)
                            .is_none(),
                        "unselected named Add cannot claim the proof"
                    );
                }
                let mut uncertain = facts.clone();
                uncertain.get_mut(&0).unwrap().storage = runmat_types::StorageFact::Unknown;
                assert!(
                    operation_range(
                        &plan.operations[0],
                        selected,
                        &uncertain,
                        &resolver.ranges,
                        2,
                    )
                    .is_none(),
                    "unknown storage cannot claim represented arithmetic"
                );
            }
        }
    }
    assert_eq!(
        (below_one + half_ulp - f64::EPSILON) as f32,
        below_one as f32
    );
    assert_eq!((below_one + half_ulp + f64::EPSILON) as f32, 1.0f32);
}

#[test]
fn represented_add_does_not_prove_indeterminate_infinity_sum() {
    let left = runmat_runtime::value_fact::value_fact(&Value::Num(f64::INFINITY));
    let right = runmat_runtime::value_fact::value_fact(&Value::Num(f64::NEG_INFINITY));
    let output = runmat_types::infer_binary(runmat_types::OperatorKind::Add, &left, &right).fact;
    let facts = HashMap::from([(0, left), (1, right), (2, output)]);
    let ranges = HashMap::from([
        (
            0,
            Some(RealRange {
                minimum: f64::INFINITY,
                maximum: f64::INFINITY,
                class: NumericClass::Double,
                resident_arithmetic: false,
            }),
        ),
        (
            1,
            Some(RealRange {
                minimum: f64::NEG_INFINITY,
                maximum: f64::NEG_INFINITY,
                class: NumericClass::Double,
                resident_arithmetic: false,
            }),
        ),
    ]);
    assert!(operation_range(&add_op(0, 1, 2), None, &facts, &ranges, 2).is_none());
}

#[test]
fn represented_add_uses_only_a_certified_resident_numeric_carrier() {
    let mut resident = runmat_runtime::value_fact::value_fact(&Value::Num(0.25));
    resident.storage = StorageFact::Opaque;
    let scalar = runmat_runtime::value_fact::value_fact(&Value::Num(1.0));
    let mut output =
        runmat_types::infer_binary(runmat_types::OperatorKind::Add, &resident, &scalar).fact;
    output.storage = StorageFact::Unknown;
    let facts = HashMap::from([(0, resident.clone()), (1, scalar), (2, output)]);
    let mut ranges = HashMap::from([
        (
            0,
            Some(RealRange {
                minimum: 0.25,
                maximum: 0.75,
                class: NumericClass::Double,
                resident_arithmetic: false,
            }),
        ),
        (
            1,
            Some(RealRange {
                minimum: 1.0,
                maximum: 1.0,
                class: NumericClass::Double,
                resident_arithmetic: false,
            }),
        ),
    ]);
    assert!(operation_range(&add_op(0, 1, 2), None, &facts, &ranges, 2).is_none());

    ranges
        .get_mut(&0)
        .unwrap()
        .as_mut()
        .unwrap()
        .resident_arithmetic = true;
    let sum = operation_range(&add_op(0, 1, 2), None, &facts, &ranges, 2)
        .expect("exact-owner resident probe certifies represented add");
    assert_eq!((sum.minimum, sum.maximum), (1.25, 1.75));
    assert!(
        sum.resident_arithmetic,
        "derived range retains the certificate"
    );
    assert_eq!(
        classification(sum, RealUnaryPredicate::AtLeastOne),
        RealUnaryClassification::Inside
    );

    ranges.get_mut(&0).unwrap().as_mut().unwrap().minimum = -0.25;
    let unsafe_sum = operation_range(&add_op(0, 1, 2), None, &facts, &ranges, 2)
        .expect("represented negative resident range remains known");
    assert_eq!(unsafe_sum.minimum, 0.75);
    assert_eq!(
        classification(unsafe_sum, RealUnaryPredicate::AtLeastOne),
        RealUnaryClassification::Indeterminate
    );

    let mut sparse = facts.clone();
    sparse.get_mut(&0).unwrap().storage = StorageFact::Sparse;
    assert!(operation_range(&add_op(0, 1, 2), None, &sparse, &ranges, 2).is_none());
    let mut foreign = facts;
    foreign.get_mut(&0).unwrap().kind = runmat_types::ValueKindFact::Unknown;
    assert!(operation_range(&add_op(0, 1, 2), None, &foreign, &ranges, 2).is_none());
}

#[test]
fn certified_single_resident_add_retains_canonical_rounding() {
    let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
    let half_ulp = f64::from(f32::EPSILON) / 4.0;
    let mut resident = runmat_runtime::value_fact::value_fact(&Value::Tensor(
        runmat_value::Tensor::from_f32(vec![below_one as f32], vec![1, 1]).unwrap(),
    ));
    resident.storage = StorageFact::Opaque;
    let scalar = runmat_runtime::value_fact::value_fact(&Value::Num(half_ulp));
    let mode = represented_add_mode(
        &resident,
        &scalar,
        RealRange {
            minimum: below_one,
            maximum: below_one,
            class: NumericClass::Single,
            resident_arithmetic: true,
        },
        RealRange {
            minimum: half_ulp,
            maximum: half_ulp,
            class: NumericClass::Double,
            resident_arithmetic: false,
        },
    )
    .expect("resident mixed-precision add uses canonical mode");
    assert_eq!(mode, RealFloatingAddMode::MixedDoubleThenSingle);
    assert_eq!(mode.add_represented(below_one, half_ulp), 1.0);
}

#[test]
fn real_domain_endpoints_are_closed_but_outside_is_not_admitted() {
    let range = RealRange {
        minimum: -1.0,
        maximum: 1.0,
        class: NumericClass::Double,
        resident_arithmetic: false,
    };
    assert_eq!(
        classification(range, RealUnaryPredicate::ClosedUnitInterval),
        RealUnaryClassification::Inside,
    );
    assert_eq!(
        classification(range, RealUnaryPredicate::AtLeastOne),
        RealUnaryClassification::Indeterminate,
    );
}
