//! Measurement-only release benchmark for the canonical cached
//! `PlacementSession` plan-and-permit call. It intentionally has no built-in
//! pass threshold and does not represent cold planning, observed replanning,
//! or an outer runtime-service adapter boundary.

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::placement::PlacementSession;
use runmat_execution::{CandidateExecutionLocation, ExecutionCandidateKind, PlacementPlanRequest};
use runmat_hir::LoweringContext;
use runmat_runtime::context::{
    RuntimeContext, RuntimeGraphPlacementSelection, RuntimePlacementService, RuntimeServicePorts,
};
use runmat_value::{NumericDType, Value};
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[path = "../tests/support/typed_reduction_placement.rs"]
mod typed_reduction_placement;
use typed_reduction_placement::CapturingPlacement;

#[derive(Clone, Copy)]
enum Workload {
    Latency8x8,
    Throughput2048x2048,
}

impl Workload {
    const ALL: [Self; 2] = [Self::Latency8x8, Self::Throughput2048x2048];

    fn name(self) -> &'static str {
        match self {
            Self::Latency8x8 => "latency_8x8",
            Self::Throughput2048x2048 => "throughput_2048x2048",
        }
    }

    fn shape(self) -> [usize; 2] {
        match self {
            Self::Latency8x8 => [8, 8],
            Self::Throughput2048x2048 => [2048, 2048],
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::Latency8x8 => {
                "x = reshape(single(1:64), 8, 8); y = min(x + x, [], 1); z = gather(y);"
            }
            Self::Throughput2048x2048 => {
                "x = reshape(single(1:4194304), 2048, 2048); y = min(x + x, [], 1); z = gather(y);"
            }
        }
    }
}

fn compile_source(source: &str) -> runmat_vm::Bytecode {
    let ast = runmat_parser::parse(source).expect("benchmark source parses");
    let hir = runmat_hir::lower(&ast, &LoweringContext::empty()).expect("benchmark source lowers");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
        .expect("benchmark source lowers to MIR");
    let entrypoint = hir.assembly.entrypoints[0].id;
    runmat_vm::compile(&hir.assembly, &mir, entrypoint).expect("benchmark source compiles")
}

struct PreparedBenchmark {
    session: Rc<PlacementSession>,
    request: PlacementPlanRequest,
    provider_device_id: u32,
    _bytecode: runmat_vm::Bytecode,
}

fn prepare(workload: Workload) -> PreparedBenchmark {
    register_wgpu_provider(WgpuProviderOptions::default())
        .expect("folded-reduction planner benchmark requires an actual WGPU adapter");
    let bytecode = compile_source(workload.source());
    let session = Rc::new(PlacementSession::default());
    let placement = Rc::new(CapturingPlacement {
        session: session.clone(),
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let vars = vec![Value::Num(0.0); bytecode.var_count];
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        block_on(runmat_vm::interpret_workspace(
            &bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime.clone(),
            None,
            None,
        ))
        .outcome
        .expect("benchmark setup executes authentic folded reduction");
    let vars = &completion.values;
    block_on(runtime.scope(async {}));
    let output_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("gathered benchmark output");
    let output_value = vars.get(output_slot).expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!("benchmark setup must publish a host tensor");
    };
    let [rows, columns] = workload.shape();
    assert_eq!(output.shape(), [1, columns]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(
        output.materialize_f64(),
        (0..columns)
            .map(|column| 2.0 * (1 + rows * column) as f64)
            .collect::<Vec<_>>()
    );

    let captures = placement.reduction_selection.borrow();
    let capture = captures
        .last()
        .expect("captured canonical reduction request");
    let request = capture.request.clone();
    assert_eq!(capture.signature, request.signature);
    assert_eq!(capture.kind, ExecutionCandidateKind::ProviderFusion);
    let selected_candidate = request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .find(|candidate| {
            candidate.descriptor.identity == capture.candidate
                && candidate.descriptor.kind == capture.kind
        })
        .expect("captured selection belongs to the canonical request");
    assert_eq!(
        capture.preparation,
        selected_candidate.descriptor.preparation
    );
    let CandidateExecutionLocation::Provider {
        device_id: provider_device_id,
    } = selected_candidate.descriptor.execution_location
    else {
        panic!("captured provider selection requires a provider location");
    };
    assert_eq!(capture.device_id, provider_device_id);
    drop(captures);

    // Prime the exact request once outside measurement. Default exploration is
    // disabled, so subsequent calls exercise the canonical cached boundary.
    let primed =
        RuntimePlacementService::plan_with_observation_permit(session.as_ref(), request.clone())
            .expect("captured authentic request admits cached session planning");
    assert!(primed.decision.from_cache);
    assert!(primed.observation_permit.as_ref().is_some_and(|permit| {
        let [choice] = primed.decision.selections.as_slice() else {
            return false;
        };
        permit.authorizes(&primed.decision.signature, &choice.candidate, choice.kind)
    }));
    PreparedBenchmark {
        session,
        request,
        provider_device_id,
        _bytecode: bytecode,
    }
}

fn percentile(mut values: Vec<f64>, percentile: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = (values.len() - 1) * percentile / 100;
    values[index]
}

fn validate_cached_selection(
    prepared: &PreparedBenchmark,
    selected: &RuntimeGraphPlacementSelection,
) {
    assert!(selected.decision.from_cache);
    assert_eq!(selected.decision.signature, prepared.request.signature);
    let [choice] = selected.decision.selections.as_slice() else {
        panic!("one folded-reduction choice is required");
    };
    assert!(prepared
        .request
        .graph
        .nodes
        .iter()
        .flat_map(|node| &node.candidates)
        .any(|candidate| {
            candidate.descriptor.identity == choice.candidate
                && candidate.descriptor.kind == choice.kind
        }));
    assert!(selected.observation_permit.as_ref().is_some_and(|permit| {
        permit.authorizes(&selected.decision.signature, &choice.candidate, choice.kind)
    }));
}

fn parse_args() -> (usize, usize) {
    let args = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect::<Vec<_>>();
    if args.len() > 2 {
        eprintln!("usage: folded_reduction_placement [iterations:1..100000] [samples:3..1000]");
        std::process::exit(2);
    }
    let parse = |index: usize, default, minimum, maximum| {
        args.get(index)
            .map_or(Some(default), |value| value.parse::<usize>().ok())
            .filter(|value| *value >= minimum && *value <= maximum)
            .unwrap_or_else(|| {
                eprintln!(
                    "usage: folded_reduction_placement [iterations:1..100000] [samples:3..1000]"
                );
                std::process::exit(2);
            })
    };
    (parse(0, 10_000, 1, 100_000), parse(1, 31, 3, 1_000))
}

fn main() {
    let (iterations, samples) = parse_args();
    let receipt_label = std::env::var("RUNMAT_BENCH_RECEIPT_LABEL")
        .expect("set RUNMAT_BENCH_RECEIPT_LABEL to the coordinator's source receipt label");
    println!(
        "schema,receipt_label,phase,os,arch,workload,rows,columns,provider_device,provider_provenance,sample,iterations,ns_per_plan"
    );
    for workload in Workload::ALL {
        let prepared = prepare(workload);
        let mut results = Vec::with_capacity(samples);
        for sample in 1..=samples {
            // Request cloning and vector allocation are deliberately outside
            // the measured full planner/permit boundary.
            let requests = vec![prepared.request.clone(); iterations];
            let mut measured_ns = 0.0;
            for request in requests {
                let started = Instant::now();
                let selected = RuntimePlacementService::plan_with_observation_permit(
                    black_box(prepared.session.as_ref()),
                    black_box(request),
                )
                .expect("captured authentic request admits repeated session planning");
                measured_ns += started.elapsed().as_secs_f64() * 1e9;
                black_box(&selected);
                validate_cached_selection(&prepared, &selected);
                drop(selected);
            }
            let ns_per_plan = measured_ns / iterations as f64;
            results.push(ns_per_plan);
            let [rows, columns] = workload.shape();
            println!(
                "runmat.s1.folded-planner.v2,{},cached-placement-session-plan-and-permit,{},{},{},{},{},{},{:?},{},{},{}",
                receipt_label,
                std::env::consts::OS,
                std::env::consts::ARCH,
                workload.name(),
                rows,
                columns,
                prepared.provider_device_id,
                prepared.request.signature.provider_provenance,
                sample,
                iterations,
                ns_per_plan,
            );
        }
        let median = percentile(results.clone(), 50);
        let p95 = percentile(results, 95);
        eprintln!(
            "S1_FOLDED_PLANNER_SUMMARY workload={} median_ns={} p95_ns={} budget=unassigned",
            workload.name(),
            median,
            p95,
        );
    }
}
