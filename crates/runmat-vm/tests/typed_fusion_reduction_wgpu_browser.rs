#![cfg(all(feature = "wgpu", target_arch = "wasm32"))]

#[path = "support/mod.rs"]
mod test_helpers;
#[path = "support/typed_reduction_placement.rs"]
mod typed_reduction_placement;
#[path = "support/typed_reduction_graph.rs"]
mod typed_reduction_graph;
#[path = "typed_fusion_reduction_wgpu_browser/cross_invocation.rs"]
mod cross_invocation;
#[path = "support/typed_reduction_special_cases.rs"]
mod special_cases;
#[path = "typed_fusion_reduction_wgpu_browser/special.rs"]
mod special;

use runmat_accelerate::backend::wgpu::provider::{
    register_wgpu_provider_async, WgpuProviderOptions,
};
use runmat_accelerate::placement::{PlacementEventKind, PlacementSession, PlacementVariant};
use runmat_accelerate_api::{AccelProvider, ProviderPrecision};
use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, CandidatePreparationState,
    ExecutionCandidateKind, PlacementExecutionAuthorityRevision,
};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_value::{NumericDType, Value};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use typed_reduction_graph::{
    assert_compiled_folded_graph, assert_compiled_function_graph, expected_column_min,
    expected_row_min, expected_third_axis_min,
};
use typed_reduction_placement::CapturingPlacement;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "support/typed_elementwise_call.rs"]
mod typed_elementwise_call;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_general_chain_and_decoded_function_complete_with_receipts() {
    runmat_runtime::builtins::wasm_registry::register_all();
    register_wgpu_provider_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter required");
    for decoded_function in [false, true] {
        typed_elementwise_call::assert_general_chain_full_call(decoded_function).await;
        typed_elementwise_call::assert_original_chain_full_call(decoded_function).await;
    }
}

struct BrowserCase {
    source: &'static str,
    function: Option<&'static str>,
    adds: usize,
    scale: f64,
    shape: &'static [usize],
    expected: fn(f64) -> Vec<f64>,
}

async fn assert_browser_full_call(case: BrowserCase) {
    let bytecode = test_helpers::compile_source(case.source)
        .expect("compile authentic browser folded reduction");
    if let Some(function) = case.function {
        assert_compiled_function_graph(&bytecode, function, case.adds);
    } else {
        assert_compiled_folded_graph(&bytecode, case.adds);
    }
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
    let prior_traces: HashSet<_> = runmat_accelerate::placement_report()
        .traces
        .iter()
        .map(|trace| trace.correlation)
        .collect();
    let vars = vec![Value::Num(0.0); bytecode.var_count];
    let runmat_vm::InterpreterOutcome::Completed(completion) = runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime.clone(),
        None,
        None,
    )
    .await
    .outcome
    .expect("browser VM must execute folded typed minimum");
    let vars = &completion.values;
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let output_value = vars.get(slot).expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!(
            "browser VM did not publish gathered numeric output: {:?}",
            vars.get(slot).as_deref()
        );
    };
    assert_eq!(output.shape(), case.shape);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), (case.expected)(case.scale));
    // The VM has published the final result; the real provider fence must
    // finish before the deferred FullCall source becomes a positive receipt.
    runtime.scope(async {}).await;

    let selections = placement.reduction_selection.borrow();
    assert_eq!(
        selections.len(),
        1,
        "one physical provider reduction selection"
    );
    let selected = &selections[0];
    assert_eq!(selected.signature.operation, "fusion.reduction");
    assert_eq!(selected.signature.region, None);
    assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
    assert!(
        matches!(
            selected.preparation,
            CandidatePreparationState::Cold | CandidatePreparationState::Warm
        ),
        "selected typed pipeline must carry actual or conservative preparation phase"
    );
    assert!(matches!(
        selected.signature.revision.authority,
        PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    ));
    let report = runmat_accelerate::placement_report();
    assert!(
        report.traces.iter().any(|trace| {
            !prior_traces.contains(&trace.correlation)
                && trace.operation == selected.signature.operation
                && trace.region == selected.signature.region
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Selected
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Queue
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
        }),
        "selected browser reduction must queue on actual WebGPU"
    );
    let matching = placement
        .delivered
        .lock()
        .unwrap()
        .iter()
        .filter(|feedback| {
            feedback.signature == selected.signature
                && feedback.candidate == selected.candidate
                && feedback.kind == selected.kind
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1, "one exact fenced FullCall receipt");
    assert!(matching[0].succeeded);
    assert!(matching[0].total_elapsed_ns > 0);
    assert!(matches!(
        session.observed_candidate_duration(
            &selected.signature,
            &selected.candidate,
            selected.kind,
        ),
        CandidateObservationEvidence::Unavailable(
            CandidateCostUnavailable::InsufficientObservations {
                successful: 1,
                failed: 0
            }
        )
    ));
    assert!(session.profile_snapshot().feedback.iter().all(|series| {
        series.signature != selected.signature
            || series.candidate != selected.candidate
            || series.kind != selected.kind
    }));
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_compiled_folded_min_full_call_numerics_and_receipt() {
    runmat_runtime::builtins::wasm_registry::register_all();
    let provider = register_wgpu_provider_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter required; unavailable is failure");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    for case in [
        BrowserCase {
            source: "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x); z = gather(y);",
            function: None,
            adds: 1,
            scale: 2.0,
            shape: &[1, 128],
            expected: expected_column_min,
        },
        BrowserCase {
            source: "function y = folded_min(a)\ny = min(a + a);\nend\nx = gpuArray(reshape(single(1:16384), 128, 128)); y = folded_min(x); z = gather(y);",
            function: Some("folded_min"),
            adds: 1,
            scale: 2.0,
            shape: &[1, 128],
            expected: expected_column_min,
        },
        BrowserCase {
            source: "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min((x + x) + (x + x)); z = gather(y);",
            function: None,
            adds: 3,
            scale: 4.0,
            shape: &[1, 128],
            expected: expected_column_min,
        },
        BrowserCase {
            source: "function y = folded_dag_min(a)\ny = min((a + a) + (a + a));\nend\nx = gpuArray(reshape(single(1:16384), 128, 128)); y = folded_dag_min(x); z = gather(y);",
            function: Some("folded_dag_min"),
            adds: 3,
            scale: 4.0,
            shape: &[1, 128],
            expected: expected_column_min,
        },
        BrowserCase {
            source: "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x, [], 2); z = gather(y);",
            function: None,
            adds: 1,
            scale: 2.0,
            shape: &[128, 1],
            expected: expected_row_min,
        },
        BrowserCase {
            source: "x = gpuArray(reshape(single(1:16384), 4, 64, 64)); y = min(x + x, [], 3); z = gather(y);",
            function: None,
            adds: 1,
            scale: 2.0,
            shape: &[4, 64, 1],
            expected: expected_third_axis_min,
        },
    ] {
        assert_browser_full_call(case).await;
    }
}
