use super::*;
use runmat_accelerate::placement::{ExplorationMode, PlacementSessionConfig};
use runmat_accelerate_api::AccelProvider;
use runmat_execution::{CandidateObservationEvidence, ExecutionCandidateKind};

#[path = "host_comparison/placement.rs"]
mod placement;
use placement::{MeasuredChoice, MeasuredPlacement};

#[path = "host_comparison/performance.rs"]
mod performance;

#[derive(Clone, Copy)]
pub(super) enum ComparisonWorkload {
    Latency8x8,
    Throughput2048x2048,
}

impl ComparisonWorkload {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Latency8x8 => "latency_8x8",
            Self::Throughput2048x2048 => "throughput_2048x2048",
        }
    }

    pub(super) fn source(self) -> &'static str {
        match self {
            Self::Latency8x8 => {
                "x = gather(reshape(single(1:64), 8, 8)); y = min(x + x, [], 1); z = gather(y);"
            }
            Self::Throughput2048x2048 => {
                "x = gather(reshape(single(1:4194304), 2048, 2048)); y = min(x + x, [], 1); z = gather(y);"
            }
        }
    }

    pub(super) fn input_shape(self) -> [usize; 2] {
        match self {
            Self::Latency8x8 => [8, 8],
            Self::Throughput2048x2048 => [2048, 2048],
        }
    }

    pub(super) fn expected(self) -> Vec<f64> {
        let [rows, columns] = self.input_shape();
        (0..columns)
            .map(|column| 2.0 * (1 + rows * column) as f64)
            .collect()
    }
}

fn execute_workload(
    workload: ComparisonWorkload,
    bytecode: &runmat_vm::Bytecode,
    runtime: &RuntimeContext,
    placement: &MeasuredPlacement,
) -> MeasuredChoice {
    let prior_choices = placement.choices.borrow().len();
    let prior_receipts = placement.delivered.lock().unwrap().len();
    let vars = vec![Value::Num(0.0); bytecode.var_count];
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        block_on(runmat_vm::interpret_workspace(
            bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime.clone(),
            None,
            None,
        ))
        .outcome
        .expect("authentic VM folded reduction workload");
    let vars = &completion.values;
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered reduction output");
    let output_value = vars.get(slot).expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!("gathered output is a numeric tensor");
    };
    let [rows, columns] = workload.input_shape();
    assert_eq!(output.shape(), [1, columns]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), workload.expected());
    let input_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "x").then_some(*slot))
        .expect("published input owner");
    let input_value = vars.get(input_slot).expect("completed input slot");
    let Value::Tensor(input) = &*input_value else {
        panic!("folded reduction must retain its host input owner");
    };
    assert_eq!(input.shape(), [rows, columns]);
    assert_eq!(input.numeric_dtype(), NumericDType::F32);
    let values = input.as_f32_slice().expect("single input storage");
    assert_eq!(values.first().copied(), Some(1.0));
    assert_eq!(values.last().copied(), Some((rows * columns) as f32));
    block_on(runtime.scope(async {}));
    let mut choices = placement.choices.borrow_mut();
    assert_eq!(choices.len(), prior_choices + 1, "one reduction per call");
    choices[prior_choices].receipts_before = prior_receipts;
    let choice = choices[prior_choices].clone();
    let receipts = placement.delivered.lock().unwrap();
    let reduction_receipts = receipts[prior_receipts..]
        .iter()
        .filter(|feedback| feedback.signature.operation == "fusion.reduction")
        .collect::<Vec<_>>();
    assert_eq!(
        reduction_receipts.len(),
        1,
        "one reduction receipt, no hidden inner provider feedback"
    );
    assert_eq!(reduction_receipts[0].signature, choice.signature);
    assert_eq!(reduction_receipts[0].candidate, choice.candidate);
    assert_eq!(reduction_receipts[0].kind, choice.kind);
    assert!(reduction_receipts[0].succeeded);
    assert!(reduction_receipts[0].total_elapsed_ns > 0);
    choice
}

#[test]
#[ignore = "requires actual WGPU hardware; explicitly run in a clean process"]
fn actual_host_and_provider_full_calls_train_one_canonical_nonexploration_decision() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for CPU/GPU comparison");
        assert_eq!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        );
        let bytecode = test_helpers::compile_source(ComparisonWorkload::Latency8x8.source())
            .expect("compile authentic host-input folded Single minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        let session = Rc::new(PlacementSession::new(PlacementSessionConfig {
            exploration: ExplorationMode::TransactionalEvery { interval: 2 },
            exploration_max_overhead_basis_points: u32::MAX,
            ..PlacementSessionConfig::default()
        }));
        let placement = Rc::new(MeasuredPlacement {
            session: session.clone(),
            choices: RefCell::new(Vec::new()),
            delivered: Arc::new(Mutex::new(Vec::new())),
        });
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
        let mut winner = None;
        for _ in 0..24 {
            let choice = execute_workload(
                ComparisonWorkload::Latency8x8,
                &bytecode,
                &runtime,
                placement.as_ref(),
            );
            // A cached decision returns before the session's exploration
            // branch. Its exact offered CPU and warm provider candidates must
            // already have independent, completed FullCall evidence.
            if choice.from_cache
                && choice.cpu_observed_candidate.is_some()
                && choice.warm_provider_observed_candidate.is_some()
                && choice.selected_observed_before
            {
                winner = Some(choice);
                break;
            }
        }
        let winner = winner.unwrap_or_else(|| {
            let choices = placement.choices.borrow();
            let receipts = placement.delivered.lock().unwrap();
            for (invocation, choice) in choices.iter().enumerate() {
                let offered = choice
                    .offers
                    .iter()
                    .map(|(candidate, kind, preparation, observed_before)| {
                        let completed_before = receipts[..choice.receipts_before]
                            .iter()
                            .filter(|feedback| {
                                feedback.signature == choice.signature
                                    && feedback.candidate == *candidate
                                    && feedback.kind == *kind
                            })
                            .count();
                        (candidate, kind, preparation, observed_before, completed_before)
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "8x8 call={} signature={:?} resources={:?} selected={:?}/{} cache={} selected_observed={} offers={:?}",
                    invocation + 1,
                    choice.signature.cache_key(),
                    choice.resources,
                    choice.kind,
                    choice.candidate,
                    choice.from_cache,
                    choice.selected_observed_before,
                    offered,
                );
            }
            panic!(
                "one exact warm signature needs measured CPU and GPU routes before a cached nonexploration decision"
            );
        });
        let cpu = winner.cpu_observed_candidate.as_deref().unwrap();
        let provider = winner.warm_provider_observed_candidate.as_deref().unwrap();
        let receipts = placement.delivered.lock().unwrap();
        for (candidate, kind) in [
            (cpu, ExecutionCandidateKind::SharedRuntime),
            (provider, ExecutionCandidateKind::ProviderFusion),
        ] {
            let matching = receipts[..winner.receipts_before]
                .iter()
                .filter(|feedback| {
                    feedback.signature == winner.signature
                        && feedback.candidate == candidate
                        && feedback.kind == kind
                })
                .collect::<Vec<_>>();
            assert!(
                matching.len() >= 2,
                "two completed exact FullCall samples before the nonexploration decision"
            );
            assert!(matching
                .iter()
                .all(|feedback| feedback.succeeded && feedback.total_elapsed_ns > 0));
            assert!(matches!(
                session.observed_candidate_duration(&winner.signature, candidate, kind,),
                CandidateObservationEvidence::Observed(_)
            ));
        }
        assert!(matches!(
            winner.kind,
            ExecutionCandidateKind::SharedRuntime | ExecutionCandidateKind::ProviderFusion
        ));
        assert!(
            session
                .profile_snapshot()
                .feedback
                .iter()
                .all(|series| { series.signature != winner.signature }),
            "VM session-only learning must not enter portable profile feedback"
        );
    });
}
