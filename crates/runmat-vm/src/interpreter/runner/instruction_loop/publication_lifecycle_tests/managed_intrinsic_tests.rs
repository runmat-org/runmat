//! Authentic managed SharedRuntime execution for primitive owner consumers.
//! The test observes the canonical policy; it does not pin a production route.

use super::*;
use runmat_accelerate::graph::{AccelNodeLabel, PrimitiveOp};
use runmat_accelerate::placement::{
    ExplorationMode, ManagedPlacementCapacity, PlacementEventKind, PlacementSessionConfig,
    PlacementVariant,
};
use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
use runmat_runtime::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
};
use runmat_value::NumericDType;

const SOURCE: &str = "w = sin(-(2 * (-(+x))));";

struct ManagedParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            cpu_millicores_available: 1_000,
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn slot(bytecode: &crate::bytecode::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .unwrap_or_else(|| panic!("compiled source has no {name} slot"))
}

fn assert_numerical_group(bytecode: &crate::bytecode::Bytecode) {
    let (graph, groups) = bytecode
        .reconciled_runtime_fusion_graph()
        .expect("compiled intrinsic source has a reconciled fusion graph");
    let named = runmat_accelerate::graph::FusionCallOrigin::named("sin");
    assert!(matches!(
        named,
        runmat_accelerate::graph::FusionCallOrigin::Named { .. }
    ));
    assert!(
        groups.iter().any(|group| {
            let operations = group
                .nodes
                .iter()
                .filter_map(|id| graph.node(*id))
                .filter_map(|node| match &node.label {
                    AccelNodeLabel::Primitive(operation) => Some(*operation),
                    _ => None,
                })
                .filter(|operation| {
                    matches!(
                        operation,
                        PrimitiveOp::UPlus | PrimitiveOp::Neg | PrimitiveOp::Mul
                    )
                })
                .collect::<Vec<_>>();
            let has_named_call = group.nodes.iter().filter_map(|id| graph.node(*id)).any(
                |node| matches!(&node.label, AccelNodeLabel::Builtin { origin, .. } if *origin == named),
            );
            has_named_call && operations
                == [
                    PrimitiveOp::UPlus,
                    PrimitiveOp::Neg,
                    PrimitiveOp::Mul,
                    PrimitiveOp::Neg,
                ]
        }),
        "one source-ordered UPlus/Neg/scalar-Mul/Neg/sin group required: {groups:?}; {graph:?}"
    );
}

fn assert_f32(value: &Value, expected: &[f64]) {
    let Value::Tensor(tensor) = value else {
        panic!("managed intrinsic output must be an F32 Tensor, got {value:?}");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(tensor.materialize_f64(), expected);
}

#[test]
#[ignore = "requires an actual WGPU adapter and an authentic SharedRuntime choice"]
fn managed_intrinsics_consume_source_ordered_owners_and_refund_exact_outputs() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        let ledger = ManagedHostEffectLedger::new_with_epoch(64 * 1024 * 1024, 49);
        let session = Rc::new(
            PlacementSession::new_with_managed_observations(
                PlacementSessionConfig {
                    exploration: ExplorationMode::TransactionalEvery { interval: 2 },
                    exploration_max_overhead_basis_points: u32::MAX,
                    ..PlacementSessionConfig::default()
                },
                &ledger,
                ManagedPlacementCapacity {
                    in_flight_slots: 64,
                    feedback_series: 128,
                    signature_bytes_per_series: 4_096,
                },
            )
            .expect("managed placement observation backing"),
        );
        let capture = Rc::new(CapturingPlacement {
            session,
            reduction_selection: RefCell::new(Vec::new()),
            delivered: Arc::new(Mutex::new(Vec::new())),
            invalidate_selected_provider: Cell::new(false),
            revoked_before_publication: Cell::new(0),
        });
        let bindings = std::collections::HashMap::from([("x".to_string(), 0)]);
        let bytecode = compile_source_with_bindings(SOURCE, &bindings);
        assert_numerical_group(&bytecode);
        let slots = [slot(&bytecode, "x"), slot(&bytecode, "w")];
        let one_output = HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &[1, 2])
            .expect("two-element F32 retained output");

        for _ in 0..24 {
            runmat_accelerate::placement::reset();
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_placement(capture.clone())
                    .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
            );
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let receipt = HostAllocationReceipt::admit(
                Some(&runtime),
                HostAllocationPlan::new(0, one_output).expect("input allocation plan"),
            )
            .expect("managed input admission")
            .expect("managed input receipt");
            let mut input = receipt
                .exact_output_tensor(&runtime, NumericDType::F32, &[1, 2])
                .expect("exact managed F32 input");
            input
                .as_f32_slice_mut()
                .expect("F32 input backing")
                .copy_from_slice(&[1.0, 2.0]);
            vars[slots[0]] = Value::Tensor(input.into_tensor().expect("managed input tensor"));
            drop(receipt);
            let outcome = block_on(scoped_faultable_run(
                bytecode.clone(),
                crate::InterpreterVars::from_owned(vars),
                runtime.clone(),
                None,
            ))
            .unwrap_or_else(|error| {
                panic!(
                    "authentic intrinsic trial: {error:?}; instructions: {:?}; private bridges: {:?}",
                    bytecode.instructions,
                    crate::bytecode::fusion_private_temp::bridges(&bytecode),
                )
            });
            block_on(runtime.scope(async {}));
            let selected = runmat_accelerate::placement_report()
                .traces
                .iter()
                .any(|trace| {
                    trace.operation == "fusion.elementwise"
                        && trace.events.iter().any(|event| {
                            event.kind == PlacementEventKind::Selected
                                && event.variant == Some(PlacementVariant::SharedRuntime)
                        })
                });
            if !selected {
                drop(outcome);
                continue;
            }

            let crate::InterpreterOutcome::Completed(completion) = outcome;
            let input = (*completion.values.get(slots[0]).unwrap()).clone();
            let output = (*completion.values.get(slots[1]).unwrap()).clone();
            assert_f32(&input, &[1.0, 2.0]);
            assert_f32(
                &output,
                &[f64::from(2.0_f32.sin()), f64::from(4.0_f32.sin())],
            );
            // Only the two immutable tensor headers are retained; they share
            // the original numeric buffers. Retire the sole frame and its
            // root metadata before measuring those independent owners.
            drop(completion);
            let retained = ledger.snapshot().live_claimed_bytes;
            drop((input, output));
            let after_drop = ledger.snapshot().live_claimed_bytes;
            assert_eq!(
                retained - after_drop,
                2 * one_output,
                "only the input and published result retain exact owners after the fused region"
            );
            return;
        }
        panic!("canonical exploration did not select an authentic SharedRuntime intrinsic group");
    });
}
