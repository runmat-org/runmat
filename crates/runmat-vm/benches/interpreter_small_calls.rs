//! Tiny-vector and anonymous-call measurements through the real bytecode
//! interpreter. Parsing/compilation and output validation are outside timing;
//! execution and creation of its rooted result are inside. No JIT or GPU
//! provider is installed. This benchmark has no performance pass threshold.

use futures::executor::block_on;
use runmat_hir::LoweringContext;
use runmat_value::Value;
use std::hint::black_box;
use std::time::Instant;

fn compile(source: &str) -> runmat_vm::Bytecode {
    let ast = runmat_parser::parse(source).expect("benchmark parses");
    let hir = runmat_hir::lower(&ast, &LoweringContext::empty()).expect("benchmark lowers");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("benchmark lowers to MIR");
    runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("benchmark compiles")
}

fn main() {
    // Provision the native execution stack once, outside every measured sample.
    // The benchmark's runtime values and futures stay on this thread.
    std::thread::Builder::new()
        .name("interpreter-benchmark".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(measure)
        .expect("start benchmark execution thread")
        .join()
        .expect("benchmark execution thread failed");
}

fn measure() {
    let args = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect::<Vec<_>>();
    assert!(args.len() <= 3, "usage: interpreter_small_calls [iterations:1..1000000] [samples:1..1000] [vector|anonymous|inline]");
    let number = |index: usize, default: usize, max: usize| {
        args.get(index)
            .map_or(Some(default), |value| value.parse::<usize>().ok())
            .filter(|value| *value > 0 && *value <= max)
            .expect("benchmark count outside its documented bounds")
    };
    let iterations = number(0, 2_000, 1_000_000);
    let samples = number(1, 5, 1_000);
    let selected = args.get(2).map(String::as_str);
    assert!(
        selected.is_none_or(|name| matches!(name, "vector" | "anonymous" | "inline")),
        "unknown workload"
    );
    let source_identity = std::env::var("RUNMAT_BENCH_RECEIPT_LABEL")
        .expect("set RUNMAT_BENCH_RECEIPT_LABEL to the measured source identity");
    let cases = [
        ("vector", "u=[1;2;3]; k1=[0.5;1;1.5]; dt=0.01;", "v=u+dt/2*k1;", [1.0025,2.005,3.0075]),
        ("anonymous", "sigma=10; rho=28; b=8/3; u=[1;2;3]; f=@(t,u)[sigma*(u(2)-u(1));u(1)*(rho-u(3))-u(2);u(1)*u(2)-b*u(3)];", "v=f(0,u);", [10.,23.,-6.]),
        ("inline", "sigma=10; rho=28; b=8/3; u=[1;2;3];", "v=[sigma*(u(2)-u(1));u(1)*(rho-u(3))-u(2);u(1)*u(2)-b*u(3)];", [10.,23.,-6.]),
    ];
    for (name, setup, body, expected) in cases {
        if selected.is_some_and(|selected| selected != name) {
            continue;
        }
        let bytecode = compile(&format!("{setup}\nfor i=1:{iterations}\n{body}\nend\n"));
        let output = bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "v").then_some(*slot))
            .expect("output slot");
        // Sample zero warms the same executable and validates the same result.
        for sample in 0..=samples {
            let started = Instant::now();
            let values =
                block_on(runmat_vm::interpret(black_box(&bytecode))).expect("benchmark executes");
            let elapsed = started.elapsed();
            let Some(Value::Tensor(value)) = values.get(output) else {
                panic!("benchmark must return a tensor");
            };
            assert_eq!(value.shape(), [3, 1]);
            let actual = value.materialize_f64();
            assert!(actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
            if sample != 0 {
                println!(
                    "{}",
                    serde_json::json!({
                        "schema": "runmat.interpreter-small-calls.v1",
                        "boundary": "bytecode-interpreter-and-rooted-result",
                        "source": source_identity,
                        "os": std::env::consts::OS,
                        "arch": std::env::consts::ARCH,
                        "workload": name,
                        "iterations": iterations,
                        "sample": sample,
                        "execution_ns": elapsed.as_nanos(),
                    })
                );
            }
        }
    }
}
