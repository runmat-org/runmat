#![cfg(feature = "wgpu")]

#[path = "support/mod.rs"]
mod test_helpers;

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::placement::{PlacementEventKind, PlacementSession, PlacementVariant};
use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, CandidatePreparationState,
    PlacementExecutionAuthorityRevision,
};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_value::{NumericDType, Value};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[path = "support/typed_reduction_graph.rs"]
mod typed_reduction_graph;
#[path = "typed_fusion_reduction_wgpu/operators.rs"]
mod operators;
#[path = "support/typed_reduction_placement.rs"]
mod typed_reduction_placement;
use typed_reduction_graph::{
    assert_compiled_folded_graph, assert_compiled_function_graph, expected_column_min,
    expected_row_min,
};
use typed_reduction_placement::CapturingPlacement;

#[path = "support/typed_elementwise_call.rs"]
mod typed_elementwise_call;
#[path = "support/fusion_diagnostics.rs"]
mod fusion_diagnostics;

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_general_chain_and_decoded_function_complete_with_receipts() {
    fusion_diagnostics::enable();
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required");
        for decoded_function in [false, true] {
            block_on(typed_elementwise_call::assert_general_chain_full_call(
                decoded_function,
            ));
            block_on(typed_elementwise_call::assert_original_chain_full_call(
                decoded_function,
            ));
        }
    });
}

fn assert_actual_folded_full_call(bytecode: &runmat_vm::Bytecode, expected: &[f64]) {
    assert_actual_folded_full_call_shape(bytecode, &[1, 128], expected);
}

fn assert_actual_folded_full_call_shape(
    bytecode: &runmat_vm::Bytecode,
    expected_shape: &[usize],
    expected: &[f64],
) {
    drop(assert_actual_folded_full_call_shape_with_dtype(
        bytecode,
        expected_shape,
        expected,
        NumericDType::F32,
    ));
}

fn assert_actual_folded_full_call_shape_with_dtype(
    bytecode: &runmat_vm::Bytecode,
    expected_shape: &[usize],
    expected: &[f64],
    expected_dtype: NumericDType,
) -> runmat_runtime::execution::RootedValues {
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
            bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime.clone(),
            None,
            None,
        ))
        .outcome
        .expect("actual VM typed folded reduction");
    let vars = completion
        .with_owned(|completion| {
            runmat_runtime::execution::RootedValues::new(
                completion
                    .values
                    .into_owned()
                    .expect("ordinary folded-reduction frame is owned"),
            )
        })
        .expect("root published folded reduction values");
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .expect("published gathered output");
    let Value::Tensor(output) = &vars[slot] else {
        panic!(
            "VM output was not gathered numeric tensor: {:?}",
            vars[slot]
        );
    };
    assert_eq!(output.shape(), expected_shape);
    assert_eq!(output.numeric_dtype(), expected_dtype);
    assert_eq!(output.materialize_f64(), expected);
    block_on(runtime.scope(async {}));

    let selections = placement.reduction_selection.borrow();
    assert_eq!(
        selections.len(),
        1,
        "exactly one typed reduction plan required"
    );
    let capture = &selections[0];
    let selected = &capture.signature;
    let selected_candidate = &capture.candidate;
    let selected_kind = &capture.kind;
    assert_eq!(selected.operation, "fusion.reduction");
    // The VM installation is in the session revision, not a portable RegionId.
    assert_eq!(selected.region, None);
    assert!(
        matches!(
            selected.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ),
        "VM-installed placement session revision required"
    );
    let report = runmat_accelerate::placement_report();
    assert!(
        report.traces.iter().any(|trace| {
            trace.operation == selected.operation
                && trace.region == selected.region
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Selected
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Queue
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
        }),
        "selected compiled reduction region must queue on actual provider"
    );
    let matching = placement
        .delivered
        .lock()
        .unwrap()
        .iter()
        .filter(|feedback| {
            feedback.signature == *selected
                && feedback.candidate == *selected_candidate
                && feedback.kind == *selected_kind
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "one ready source for the exact selected reduction"
    );
    assert!(matching[0].succeeded);
    assert!(matching[0].total_elapsed_ns > 0);
    assert!(
        matches!(
            session.observed_candidate_duration(selected, selected_candidate, *selected_kind),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0,
                }
            )
        ),
        "the first real successful receipt is recorded but cannot yet calibrate a duration"
    );
    assert!(
        session.profile_snapshot().feedback.iter().all(|series| {
            series.signature != *selected
                || series.candidate != *selected_candidate
                || series.kind != *selected_kind
        }),
        "session-only authority must not be exported into portable profile feedback"
    );
    vars
}

// The local F32 provider can authenticate Single extrema. Nonidentity Sum
// requires Double accumulation and needs separate capable-device coverage.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_folded_single_min_full_call_deposits_after_publication_and_fence() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for folded Single minimum qualification");
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min(x + x); z = gather(y);",
        )
        .expect("compile authentic folded Single minimum");
        assert_compiled_folded_graph(&bytecode, 1);
        // Column-major input has a distinct minimum in every column.
        let expected = expected_column_min(2.0);
        assert_actual_folded_full_call(&bytecode, &expected);
    });
}

#[path = "typed_fusion_reduction_wgpu/auto_two_pass.rs"]
mod auto_two_pass;
#[path = "typed_fusion_reduction_wgpu/axis.rs"]
mod axis;
#[path = "typed_fusion_reduction_wgpu/cross_invocation.rs"]
mod cross_invocation;
#[cfg(not(target_arch = "wasm32"))]
#[path = "typed_fusion_reduction_wgpu/double_sum.rs"]
mod double_sum;
#[path = "typed_fusion_reduction_wgpu/host_comparison.rs"]
mod host_comparison;
#[path = "typed_fusion_reduction_wgpu/nd.rs"]
mod nd;
#[path = "typed_fusion_reduction_wgpu/special.rs"]
mod special;
#[path = "support/typed_reduction_special_cases.rs"]
mod special_cases;
#[path = "typed_fusion_reduction_wgpu/warm.rs"]
mod warm;

// The ordinary runner consumes FunctionBytecode::execution_bytecode directly.
// `for_function` exposes the same admitted body to external tests, but can also
// attach an entrypoint layout; clear that optional layout before graph checks.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_function_owned_folded_single_min_publishes_after_fence() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for function-owned folded minimum");
        let bytecode = test_helpers::compile_source(
            "function y = folded_min(a)\ny = min(a + a);\nend\nx = gpuArray(reshape(single(1:16384), 128, 128)); y = folded_min(x); z = gather(y);",
        )
        .expect("compile ordinary function-owned folded Single minimum");
        assert_compiled_function_graph(&bytecode, "folded_min", 1);
        let expected = expected_column_min(2.0);
        assert_actual_folded_full_call(&bytecode, &expected);
    });
}

// Three pure Adds and one minimum must stay in one selected region; this is a
// separate real-hardware gate for the richer compiler-owned private DAG.
#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_vm_three_add_dag_min_full_call_deposits_after_fence() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for three-Add DAG minimum");
        let bytecode = test_helpers::compile_source(
            "x = gpuArray(reshape(single(1:16384), 128, 128)); y = min((x + x) + (x + x)); z = gather(y);",
        )
        .expect("compile authentic three-Add folded minimum");
        assert_compiled_folded_graph(&bytecode, 3);
        let expected = expected_column_min(4.0);
        assert_actual_folded_full_call(&bytecode, &expected);
    });
}

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn actual_function_owned_three_add_dag_min_full_call_deposits_after_fence() {
    runmat_gc::gc_test_context(|| {
        register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required for function-owned three-Add DAG minimum");
        let bytecode = test_helpers::compile_source(
            "function y = folded_dag_min(a)\ny = min((a + a) + (a + a));\nend\nx = gpuArray(reshape(single(1:16384), 128, 128)); y = folded_dag_min(x); z = gather(y);",
        )
        .expect("compile ordinary function-owned three-Add DAG minimum");
        assert_compiled_function_graph(&bytecode, "folded_dag_min", 3);
        let expected = expected_column_min(4.0);
        assert_actual_folded_full_call(&bytecode, &expected);
    });
}
