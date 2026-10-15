//! One compiled general-chain execution witness shared by native and WebGPU hosts.
//! This intentionally does not qualify the original mixed-Double POWER expression.
use super::test_helpers;
use super::typed_reduction_placement::{selected_provider_capability_digest, CapturingPlacement};
use runmat_accelerate::placement::{PlacementEventKind, PlacementSession, PlacementVariant};
use runmat_accelerate::FusionKind;
use runmat_execution::{ExecutionCandidateKind, PlacementExecutionAuthorityRevision};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_value::{NumericDType, Value};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub(crate) async fn assert_general_chain_full_call(decoded_function: bool) {
    let expression = "exp(-x).*sin(4*x)+tanh(x).*tanh(x)";
    let source = if decoded_function {
        format!("function y = general_chain(x)\ny = {expression};\nend\nx = gpuArray(single(ones(1, 16384))); y = general_chain(x); z = gather(y);")
    } else {
        format!("x = gpuArray(single(ones(1, 16384))); y = {expression}; z = gather(y);")
    };
    let mut bytecode =
        test_helpers::compile_source(&source).expect("compile authentic general chain");
    let function_view;
    let graph_bytecode = if decoded_function {
        let encoded =
            runmat_vm::encode_interpreter_program_v2(&bytecode.function_registry()).unwrap();
        let decoded = runmat_vm::decode_interpreter_program_v2(&encoded).unwrap();
        let id = decoded
            .resolve_name("general_chain")
            .expect("decoded function identity");
        function_view = {
            let mut view = runmat_vm::Bytecode::for_function(
                decoded.get(id).unwrap(),
                decoded.clone(),
                bytecode.layout.clone().unwrap(),
            );
            view.layout = None;
            view
        };
        bytecode.bound_functions = decoded.clone().into_functions();
        bytecode.function_registry = decoded;
        &function_view
    } else {
        &bytecode
    };
    let (graph, groups) = graph_bytecode
        .reconciled_runtime_fusion_graph()
        .expect("checked general-chain graph");
    let signals = graph_bytecode.instructions.iter().enumerate().filter_map(|(pc, instruction)| {
        matches!(instruction, runmat_vm::Instr::CallBuiltinMulti(name, 1, 1) if name == "exp" || name == "sin" || name == "tanh").then_some(pc)
    }).collect::<Vec<_>>();
    assert_eq!(signals.len(), 4);
    let chain = groups
        .iter()
        .find(|group| {
            group.kind == FusionKind::ElementwiseChain
                && signals
                    .iter()
                    .all(|pc| group.span.start <= *pc && *pc <= group.span.end)
        })
        .expect("one complete compiler-owned general chain");
    assert!(chain.nodes.len() >= 8);
    assert!(chain.nodes.iter().all(|id| graph.node(*id).is_some()));

    let placement = Rc::new(CapturingPlacement {
        session: Rc::new(PlacementSession::default()),
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
    let before = runmat_accelerate::placement_report()
        .traces
        .iter()
        .map(|trace| trace.correlation)
        .collect::<HashSet<_>>();
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        runmat_vm::interpret_with_vars_in_context(&bytecode, &mut vars, None, runtime.clone())
            .await
            .expect("ordinary VM general-chain invocation");
    let published = completion
        .with_owned(|completion| runmat_runtime::execution::RootedValues::new(completion.values))
        .unwrap();
    let resident_slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "y").then_some(*slot))
        .unwrap();
    let Value::GpuTensor(resident_output) = &published[resident_slot] else {
        panic!("completed general chain must publish a provider-resident result before gather");
    };
    let expected_provider = selected_provider_capability_digest(resident_output.device_id());
    let slot = bytecode
        .var_names
        .iter()
        .find_map(|(slot, name)| (name == "z").then_some(*slot))
        .unwrap();
    let Value::Tensor(output) = &published[slot] else {
        panic!("published gathered numeric tensor required");
    };
    assert_eq!(output.shape(), [1, 16384]);
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    let expected = (-1f64).exp() * 4f64.sin() + 1f64.tanh().powi(2);
    assert!(output
        .materialize_f64()
        .iter()
        .all(|actual| (actual - expected).abs() < 2e-6));
    runtime.scope(async {}).await;
    let feedback = placement.delivered.lock().unwrap();
    let completed = feedback
        .iter()
        .filter(|feedback| {
            feedback.signature.operation == "fusion.elementwise"
                && feedback.kind == ExecutionCandidateKind::ProviderFusion
        })
        .collect::<Vec<_>>();
    // Setup contains no multi-signal expression; the sole elementwise region
    // is the complete general chain authenticated above.
    let traces = runmat_accelerate::placement_report();
    let chain_traces = traces
        .traces
        .iter()
        .filter(|trace| {
            !before.contains(&trace.correlation)
                && trace.operation == "fusion.elementwise"
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Selected
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
                && trace.events.iter().any(|event| {
                    event.kind == PlacementEventKind::Queue
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chain_traces.len(),
        1,
        "one compiled general chain must queue on the actual provider"
    );
    assert_eq!(
        completed.len(),
        1,
        "one actual provider fenced receipt; FullCall authority is checked below"
    );
    for receipt in completed {
        assert!(receipt.succeeded);
        assert!(receipt.total_elapsed_ns > 0);
        eprintln!("general_chain decoded_function={decoded_function} selected_trace={:?} observed_revision={:?}", chain_traces[0].correlation, receipt.signature.revision);
        // A Provider-local preparation/body/cleanup receipt is not acceptance
        // of the required VM publication boundary. Keep this obligation strict.
        assert!(matches!(receipt.signature.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)),
            "general-chain FullCall receipt required; provider-local compute alone is insufficient; actual output capability={expected_provider:?}; observed revision={:?}", receipt.signature.revision);
        assert_eq!(receipt.signature.region, None);
        assert_eq!(
            receipt.signature.provider_provenance,
            Some(expected_provider)
        );
    }
}
