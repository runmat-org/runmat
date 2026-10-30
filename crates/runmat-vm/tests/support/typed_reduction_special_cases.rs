//! Authentic borrowed-input VM cases for uncommon extrema semantics.
use crate::typed_reduction_graph::assert_compiled_folded_graph;
use crate::typed_reduction_placement::CapturingPlacement;
use runmat_accelerate::placement::{PlacementEventKind, PlacementSession, PlacementVariant};
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView,
};
use runmat_execution::{
    CandidateCostUnavailable, CandidateObservationEvidence, CandidatePreparationState,
    ExecutionCandidateKind, PlacementExecutionAuthorityRevision,
};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_value::{NumericDType, Value};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

fn compile_borrowed_x(source: &str) -> runmat_vm::Bytecode {
    let variables = HashMap::from([("x".to_string(), 0)]);
    let ast = runmat_parser::parse(source).expect("parse borrowed extrema source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&variables))
        .expect("lower borrowed extrema source");
    let mir =
        runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower borrowed extrema MIR");
    runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile borrowed extrema bytecode")
}

fn slot(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(slot, candidate)| (candidate == name).then_some(*slot))
        .expect("named borrowed input or output")
}

fn nan_input() -> Vec<f32> {
    let mut input = (1..=128 * 128)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    input[0] = -0.0;
    input[1] = 0.0;
    input[128] = f32::from_bits(0x7fc0_1234);
    for value in &mut input[256..384] {
        *value = f32::from_bits(0x7fc0_1234);
    }
    input[384] = -0.0;
    input[385] = 0.0;
    input[386] = f32::from_bits(0x7fc0_1234);
    input
}

fn expected_nan_min(omit: bool) -> Vec<f64> {
    let mut expected = (0..128)
        .map(|column| (2 * (128 * column + 1)) as f64)
        .collect::<Vec<_>>();
    expected[0] = -0.0;
    expected[1] = if omit { 260.0 } else { f64::NAN };
    expected[2] = f64::NAN;
    expected[3] = if omit { -0.0 } else { f64::NAN };
    expected
}

fn assert_f64_semantics(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "output {index}: expected NaN, got {actual}"
            );
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "output {index}: value or signed-zero bit differs"
            );
        }
    }
}

pub(crate) async fn assert_borrowed_nan_min(provider: &'static dyn AccelProvider, omit: bool) {
    let mode = if omit { "omitnan" } else { "includenan" };
    let source = format!("y = min(x + x, [], '{mode}'); z = gather(y);");
    let bytecode = compile_borrowed_x(&source);
    assert_compiled_folded_graph(&bytecode, 1);
    let input = nan_input();
    let shape = [128, 128];
    let borrowed = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&input),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .expect("upload exact Single source")
        .with_provenance(GpuHandleProvenance::Explicit)
        .expect("explicit borrowed input provenance");

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
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    vars[slot(&bytecode, "x")] = Value::GpuTensor(borrowed.clone());
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
    .expect("typed folded minimum with explicit NaN mode");
    let vars = &completion.values;
    let output_value = vars
        .get(slot(&bytecode, "z"))
        .expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!("folded result was not gathered");
    };
    assert_eq!(output.shape(), [1, 128]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_f64_semantics(&output.materialize_f64(), &expected_nan_min(omit));
    runtime.scope(async {}).await;

    {
        let selections = placement.reduction_selection.borrow();
        assert_eq!(
            selections.len(),
            1,
            "{mode}: exactly one physical selection"
        );
        let selected = &selections[0];
        assert_eq!(selected.signature.operation, "fusion.reduction");
        assert_eq!(selected.signature.region, None);
        assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
        assert!(matches!(
            selected.signature.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ));
        assert!(matches!(
            selected.preparation,
            CandidatePreparationState::Cold | CandidatePreparationState::Warm
        ));
        assert!(runmat_accelerate::placement_report()
            .traces
            .iter()
            .any(|trace| {
                !prior_traces.contains(&trace.correlation)
                    && trace.operation == selected.signature.operation
                    && trace.events.iter().any(|event| {
                        event.kind == PlacementEventKind::Selected
                            && event.variant == Some(PlacementVariant::ProviderFusion)
                    })
                    && trace.events.iter().any(|event| {
                        event.kind == PlacementEventKind::Queue
                            && event.variant == Some(PlacementVariant::ProviderFusion)
                    })
            }));
        let receipts = placement.delivered.lock().unwrap();
        assert_eq!(receipts.len(), 1, "{mode}: exact FullCall receipt");
        let receipt = &receipts[0];
        assert_eq!(receipt.signature, selected.signature);
        assert_eq!(receipt.candidate, selected.candidate);
        assert_eq!(receipt.kind, selected.kind);
        assert!(receipt.succeeded);
        assert!(receipt.total_elapsed_ns > 0);
        assert!(matches!(
            session.observed_candidate_duration(
                &selected.signature,
                &selected.candidate,
                selected.kind
            ),
            CandidateObservationEvidence::Unavailable(
                CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0,
                }
            )
        ));
    }

    let downloaded = provider
        .download_numeric(&borrowed)
        .await
        .expect("original borrowed input remains readable");
    assert_eq!(downloaded.shape, shape);
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("original input lost Single class");
    };
    assert_eq!(actual.len(), input.len());
    assert!(actual
        .iter()
        .zip(&input)
        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    drop(output_value);
    drop(completion);
    assert_eq!(
        borrowed
            .release()
            .expect("retire original after all borrows"),
        runmat_accelerate_api::allocation::AllocationReleaseOutcome::Retired
    );
}

/// Empty floating extrema are deliberately not certified by the min owner's
/// fused-reduction contract. Verify the public VM result without claiming a
/// typed provider FullCall receipt for this case.
pub(crate) async fn assert_empty_min_ordinary_fallback() {
    let bytecode = compile_borrowed_x("y = min(x + x); z = gather(y);");
    assert_compiled_folded_graph(&bytecode, 1);
    let session = Rc::new(PlacementSession::default());
    let placement = Rc::new(CapturingPlacement {
        session,
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    vars[slot(&bytecode, "x")] =
        Value::Tensor(runmat_value::Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap());
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
    .expect("empty minimum follows ordinary domain dispatch");
    let vars = &completion.values;
    let output_value = vars
        .get(slot(&bytecode, "z"))
        .expect("completed output slot");
    let Value::Tensor(output) = &*output_value else {
        panic!("empty minimum did not publish a numeric tensor");
    };
    assert_eq!(output.shape(), [0, 3]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert!(output.materialize_f64().is_empty());
    runtime.scope(async {}).await;
    assert!(
        placement.reduction_selection.borrow().is_empty(),
        "empty minimum has no owner-certified typed recipe"
    );
    assert!(placement
        .delivered
        .lock()
        .unwrap()
        .iter()
        .all(|feedback| feedback.signature.operation != "fusion.reduction"));
}
