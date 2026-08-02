#![cfg(target_arch = "wasm32")]

use runmat_builtins::Value;
use runmat_core::{ExecutableSource, InvocationControl, ProcedureInvocation, RunMatSession};
use runmat_test::coverage::CoverageMetric;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
async fn wasm_execution_hits_the_same_compiler_sites_as_native_core() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:wasm",
                "wasmCovered.m",
                "function y = wasmCovered(x)\n y = 1;\n if x > 0\n  y = 2;\n else\n  y = 3;\n end\nend\n",
            ),
            None,
        )
        .await
        .unwrap();
    let (_, coverage) = session
        .invoke_executable_with_coverage(
            &unit,
            ProcedureInvocation::function("wasmCovered", vec![Value::Num(1.0)]),
            &InvocationControl::default(),
        )
        .await
        .unwrap();

    let function_hits = coverage
        .sites
        .iter()
        .filter(|site| site.metric == CoverageMetric::Function)
        .map(|site| coverage.counts.get(&site.counter_key).copied().unwrap_or(0))
        .sum::<u64>();
    let statement_hits = coverage
        .sites
        .iter()
        .filter(|site| site.metric == CoverageMetric::Statement)
        .map(|site| coverage.counts.get(&site.counter_key).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    assert_eq!(function_hits, 1);
    assert!(statement_hits.iter().any(|count| *count > 0));
    assert!(statement_hits.contains(&0));
}
