//! Reuse the real checked-publication hook after an actual general-chain dispatch.
use super::*;
use runmat_accelerate::placement::{PlacementEventKind, PlacementVariant};

#[test]
#[ignore = "requires an actual WGPU adapter; unavailable hardware fails when explicitly run"]
fn general_chain_reject_cancel_and_drop_at_publication_never_issue_full_call_receipts() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        for fault_kind in 0..4 {
            runmat_accelerate::placement::reset();
            let caught = fault_kind == 3;
            let body = "y = exp(-x).*sin(4*x)+tanh(x).*tanh(x);";
            let source = if caught {
                format!("x = gpuArray(single(ones(1,16384))); try; {body} catch e; caught = e.identifier; end;")
            } else {
                format!("x = gpuArray(single(ones(1,16384))); {body} z = gather(y);")
            };
            let bytecode = compile_source_raw(&source);
            let (_, groups) = bytecode.reconciled_runtime_fusion_graph().unwrap();
            assert_eq!(
                groups
                    .iter()
                    .filter(|group| group.kind.is_elementwise() && group.nodes.len() >= 8)
                    .count(),
                1
            );
            let caught_slot = bytecode
                .var_names
                .iter()
                .find_map(|(slot, name)| (name == "caught").then_some(*slot));
            let (_session, capture) = placement();
            let cancellation = CancellationSource::new();
            let runtime = runtime(capture.clone(), Some(&cancellation));
            let reached = Rc::new(Cell::new(false));
            let fault = match fault_kind {
                1 => VmFusionPublicationFault::Cancel(
                    VmFusionPublicationFaultRoute::AnySelected,
                    cancellation.clone(),
                    reached.clone(),
                ),
                2 => VmFusionPublicationFault::Suspend(
                    VmFusionPublicationFaultRoute::AnySelected,
                    reached.clone(),
                ),
                _ => VmFusionPublicationFault::Reject(
                    VmFusionPublicationFaultRoute::AnySelected,
                    reached.clone(),
                ),
            };
            let vars =
                crate::InterpreterVars::from_owned(vec![Value::Num(0.0); bytecode.var_count]);
            if fault_kind == 2 {
                let mut pending = Box::pin(scoped_faultable_run(
                    bytecode,
                    vars,
                    runtime.clone(),
                    Some(fault),
                ));
                block_on(std::future::poll_fn(|context| {
                    match pending.as_mut().poll(context) {
                        Poll::Pending if reached.get() => Poll::Ready(()),
                        Poll::Pending => Poll::Pending,
                        Poll::Ready(_) => {
                            panic!("publication fault must suspend before completion")
                        }
                    }
                }));
                drop(pending);
            } else {
                let result = block_on(scoped_faultable_run(
                    bytecode,
                    vars,
                    runtime.clone(),
                    Some(fault),
                ));
                if caught {
                    let crate::InterpreterOutcome::Completed(completion) =
                        result.expect("ordinary catch handles the actual checked-store failure");
                    let vars = &completion.values;
                    assert_eq!(
                        *vars.get(caught_slot.unwrap()).unwrap(),
                        Value::CharArray(runmat_value::CharArray::new_row(
                            "RunMat:TransientSequenceNotStorable"
                        ))
                    );
                } else {
                    let error = match result {
                        Err(error) => error,
                        Ok(_) => panic!("fault must terminate invocation"),
                    };
                    assert_eq!(
                        error.identifier(),
                        Some(if fault_kind == 1 {
                            "RunMat:ExecutionCancelled"
                        } else {
                            "RunMat:TransientSequenceNotStorable"
                        })
                    );
                }
            }
            assert!(
                reached.get(),
                "fault must reach the installed FullCall publication boundary"
            );
            block_on(runtime.scope(async {}));
            assert!(
                capture
                    .delivered
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|feedback| feedback.signature.operation != "fusion.elementwise"),
                "failed/cancelled/dropped publication cannot issue a FullCall receipt"
            );
            let report = runmat_accelerate::placement_report();
            let selected = report
                .traces
                .iter()
                .filter(|trace| {
                    trace.operation == "fusion.elementwise"
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
                selected.len(),
                1,
                "exactly one actual general-chain submission; no replay after failed publication"
            );
        }
    });
}
