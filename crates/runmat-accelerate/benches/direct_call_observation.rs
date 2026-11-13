//! Standalone release microbenchmark for contextual direct-call overhead.
//! This measures the whole observer/placement-service path, not one allocation.
use futures::executor::block_on;
use runmat_accelerate::placement::PlacementSession;
use runmat_runtime::call_builtin_sequence_async_with_outputs_in_context;
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::{RootedValueSequence, RuntimeExecutionService};
use runmat_value::{Tensor, Value, ValueSequence};
use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Workload {
    StrcmpScalar,
    NumelScalar,
    NumelArray,
}

impl Workload {
    const ALL: [Self; 3] = [Self::StrcmpScalar, Self::NumelScalar, Self::NumelArray];

    fn name(self) -> &'static str {
        match self {
            Self::StrcmpScalar => "strcmp_host_scalar",
            Self::NumelScalar => "numel_host_scalar_unopted",
            Self::NumelArray => "numel_host_array_unopted",
        }
    }

    fn builtin(self) -> &'static str {
        match self {
            Self::StrcmpScalar => "strcmp",
            Self::NumelScalar | Self::NumelArray => "numel",
        }
    }

    fn arguments(self) -> Vec<Value> {
        match self {
            Self::StrcmpScalar => vec![Value::String("same".into()), Value::String("same".into())],
            Self::NumelScalar => vec![Value::Num(7.0)],
            Self::NumelArray => vec![Value::Tensor(
                Tensor::new(vec![0.0; 2_048], vec![64, 32]).expect("fixed array"),
            )],
        }
    }

    fn check(self, result: &ValueSequence) {
        let expected = match self {
            Self::StrcmpScalar => Value::Bool(true),
            Self::NumelScalar => Value::Num(1.0),
            Self::NumelArray => Value::Num(2_048.0),
        };
        assert_eq!(result.as_slice(), &[expected], "{} result", self.name());
    }
}

#[derive(Clone, Copy)]
enum Placement {
    Absent,
    Session,
}

impl Placement {
    const ALL: [Self; 2] = [Self::Absent, Self::Session];

    fn name(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Session => "real_session",
        }
    }

    fn context(self) -> RuntimeContext {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        match self {
            Self::Absent => context,
            Self::Session => context.with_service_ports(
                RuntimeServicePorts::default().with_placement(Rc::new(PlacementSession::default())),
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum Phase {
    ColdFirst,
    Warm,
}

impl Phase {
    const ALL: [Self; 2] = [Self::ColdFirst, Self::Warm];

    fn name(self) -> &'static str {
        match self {
            Self::ColdFirst => "cold_first_call",
            Self::Warm => "warm_repeated",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    workload: Workload,
    placement: Placement,
    phase: Phase,
}

fn invoke(case: Case, context: &RuntimeContext, args: &[Value]) -> RootedValueSequence {
    block_on(call_builtin_sequence_async_with_outputs_in_context(
        context,
        case.workload.builtin(),
        args,
        1,
    ))
    .expect("benchmark route must complete")
}

fn measure(case: Case, warm_iterations: u64) -> (u64, f64) {
    // Each measurement owns a fresh context. No context, session, argument, or
    // array construction is included in the timed interval.
    let context = case.placement.context();
    let args = case.workload.arguments();
    if matches!(case.phase, Phase::Warm) {
        for _ in 0..3 {
            case.workload.check(invoke(case, &context, &args).value());
        }
    }
    let iterations = if matches!(case.phase, Phase::ColdFirst) {
        1
    } else {
        warm_iterations
    };
    let started = Instant::now();
    for _ in 0..iterations {
        drop(black_box(invoke(
            case,
            black_box(&context),
            black_box(&args),
        )));
    }
    let elapsed = started.elapsed();
    // A second, untimed call validates the deterministic route. The cold row
    // still measured exactly the first call of this fresh session.
    case.workload.check(invoke(case, &context, &args).value());
    (iterations, elapsed.as_secs_f64() * 1e9 / iterations as f64)
}

fn parse_args() -> (u64, u64) {
    let args = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect::<Vec<_>>();
    if args.len() > 2 {
        eprintln!("usage: direct_call_observation [warm_iterations:1..1000000] [samples:1..1000]");
        std::process::exit(2);
    }
    let parse = |index: usize, default, maximum| {
        args.get(index)
            .map_or(Ok(default), |value| value.parse::<u64>())
            .ok()
            .filter(|value| *value > 0 && *value <= maximum)
            .unwrap_or_else(|| {
                eprintln!(
                    "usage: direct_call_observation [warm_iterations:1..1000000] [samples:1..1000]"
                );
                std::process::exit(2);
            })
    };
    (parse(0, 10_000, 1_000_000), parse(1, 31, 1_000))
}

fn main() {
    let (warm_iterations, samples) = parse_args();
    // Initialize global registration and route code before either cold or warm
    // measurements. This does not prime any measured placement session.
    for workload in Workload::ALL {
        let case = Case {
            workload,
            placement: Placement::Absent,
            phase: Phase::Warm,
        };
        let context = case.placement.context();
        let args = workload.arguments();
        for _ in 0..10 {
            workload.check(invoke(case, &context, &args).value());
        }
    }
    let mut cases = Vec::new();
    for workload in Workload::ALL {
        for placement in Placement::ALL {
            for phase in Phase::ALL {
                cases.push(Case {
                    workload,
                    placement,
                    phase,
                });
            }
        }
    }
    println!("workload,placement,phase,sample,iterations,ns_per_call");
    for sample in 0..samples {
        // Rotate all 12 cases so one route cannot always benefit from order.
        for offset in 0..cases.len() {
            let case = cases[(sample as usize + offset) % cases.len()];
            let (iterations, ns_per_call) = measure(case, warm_iterations);
            println!(
                "{},{},{},{sample},{iterations},{ns_per_call:.3}",
                case.workload.name(),
                case.placement.name(),
                case.phase.name()
            );
        }
    }
}
