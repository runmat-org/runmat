//! Standalone release microbenchmark; functional correctness lives in tests.
use runmat_execution::cancellation::CancellationSource;
use std::future::{pending, Future};
use std::hint::black_box;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::task::{Context, Wake, Waker};
use std::time::Instant;

struct NoopWake;
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

#[derive(Clone, Copy)]
enum Case {
    AtomicCheck,
    TokenCheck,
    ComposedCheck,
    PendingPoll,
    ListenerCycle,
    ComposedListenerCycle,
    RegisteredRepoll,
}

impl Case {
    const ALL: [Self; 7] = [
        Self::AtomicCheck,
        Self::TokenCheck,
        Self::ComposedCheck,
        Self::PendingPoll,
        Self::ListenerCycle,
        Self::ComposedListenerCycle,
        Self::RegisteredRepoll,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::AtomicCheck => "arc_atomic_acquire_check",
            Self::TokenCheck => "single_token_check",
            Self::ComposedCheck => "two_parent_token_check",
            Self::PendingPoll => "pending_future_poll_drop_baseline",
            Self::ListenerCycle => "single_listener_create_register_drop",
            Self::ComposedListenerCycle => "two_parent_listener_create_register_drop",
            Self::RegisteredRepoll => "registered_listener_repoll",
        }
    }

    fn measure(self, iterations: u64) -> f64 {
        // All cases use the same loop/black-box/timer structure. Generation and
        // executor-waker construction are outside timing. A cycle includes the
        // listener's own allocations, reference counts and deregistration.
        let flag = Arc::new(AtomicBool::new(false));
        let first = CancellationSource::new();
        let second = CancellationSource::new();
        let token = first.token();
        let composed = token.or(&second.token());
        let waker = Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        let mut retained = token.cancelled();
        if matches!(self, Self::RegisteredRepoll) {
            let _ = Pin::new(&mut retained).poll(&mut cx);
        } else {
            // Prime registry capacity; this benchmark measures steady-state
            // suspension, not fresh generation allocation or vector growth.
            let mut listener = composed.cancelled();
            let _ = Pin::new(&mut listener).poll(&mut cx);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            match self {
                Self::AtomicCheck => {
                    black_box(black_box(&flag).load(Ordering::Acquire));
                }
                Self::TokenCheck => {
                    black_box(black_box(&token).is_cancelled());
                }
                Self::ComposedCheck => {
                    black_box(black_box(&composed).is_cancelled());
                }
                Self::PendingPoll => {
                    let mut future = pending::<()>();
                    let _ = black_box(Pin::new(black_box(&mut future)).poll(black_box(&mut cx)));
                }
                Self::ListenerCycle | Self::ComposedListenerCycle => {
                    let selected = if matches!(self, Self::ListenerCycle) {
                        &token
                    } else {
                        &composed
                    };
                    let mut listener = black_box(selected).cancelled();
                    let _ = black_box(Pin::new(black_box(&mut listener)).poll(black_box(&mut cx)));
                    drop(black_box(listener));
                }
                Self::RegisteredRepoll => {
                    let _ = black_box(Pin::new(black_box(&mut retained)).poll(black_box(&mut cx)));
                }
            }
        }
        start.elapsed().as_secs_f64() * 1e9 / iterations as f64
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    // Cargo may pass --bench to a harness=false benchmark executable.
    let args = args
        .iter()
        .filter(|arg| arg.as_str() != "--bench")
        .collect::<Vec<_>>();
    let parse = |index: usize, default, maximum| {
        args.get(index)
            .map_or(Ok(default), |value| value.parse::<u64>())
            .ok()
            .filter(|value| *value > 0 && *value <= maximum)
            .unwrap_or_else(|| {
                eprintln!("usage: cancellation [iterations:1..50000000] [samples:1..100]");
                std::process::exit(2);
            })
    };
    if args.len() > 2 {
        eprintln!("expected iterations and samples only");
        std::process::exit(2);
    }
    let iterations = parse(0, 1_000_000, 50_000_000);
    let samples = parse(1, 9, 100);
    for case in Case::ALL {
        black_box(case.measure(iterations.min(10_000)));
    }
    println!("case,sample,iterations,ns_per_operation");
    for sample in 0..samples {
        // Rotate case order to avoid systematically favoring the first case.
        for offset in 0..Case::ALL.len() {
            let case = Case::ALL[(sample as usize + offset) % Case::ALL.len()];
            let elapsed = case.measure(iterations);
            println!("{},{sample},{iterations},{elapsed:.3}", case.name());
        }
    }
}
