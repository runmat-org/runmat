//! Stack-backed ready-route requests. The existing managed placement owner
//! funds retained feedback; this request creates no heap metadata or payload.
use super::*;

pub(super) fn select_managed_ready_route(
    service: Rc<dyn RuntimePlacementService>,
    resources: runmat_runtime::context::RuntimeParallelResources,
    runtime: &RuntimeContext,
    invocation: ReadyInvocation<'_>,
    ready: &[Option<PublishedEntry>; 2],
) -> Result<Option<ReadyRouteSelection>, RuntimeError> {
    if !service.managed_whole_work_available() {
        return Ok(None);
    }
    let Some(revision) = service.whole_work_host_service_revision_ref(
        Some(&invocation.unit.revision().program_revision),
        Digest::sha256(b"runmat.core.ready-interpreter-native.v1"),
    ) else {
        return Ok(None);
    };
    let Some(runtime_facts) = ready_runtime_facts(invocation, false)? else {
        return Ok(None);
    };
    let signature = PlacementSignatureRef {
        region: None,
        operation: "core.ready-invocation",
        runtime_facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let mut candidates = std::array::from_fn::<_, 3, _>(|_| {
        candidate(
            ExecutionCandidateKind::SharedRuntime,
            interpreter_facts(invocation.unit),
        )
    });
    let mut count = 1;
    for entry in ready.iter().flatten() {
        candidates[count] = candidate(native_kind(entry.tier), native_facts(entry));
        count += 1;
    }
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: resources.cpu_millicores_available,
        memory_available_bytes: resources.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: Vec::new(),
        epoch: resources.epoch,
    };
    // Execution scratch/retained dimensions deliberately remain Unknown.
    // Funding request metadata does not certify an arbitrary function's peak.
    let selected =
        match service.try_select_whole_work_borrowed_managed(WholeWorkPlacementRequestRef {
            signature,
            constraint: WholeWorkConstraint::AnyLegal,
            candidates: &candidates[..count],
            resources: &resources,
            deterministic: false,
            require_transactional_results: false,
        }) {
            Ok(Some(selected)) => selected,
            Ok(None) => return Ok(None),
            Err(error) => return Err(error),
        };
    if !selected.observation_is_current() {
        return Ok(None);
    }
    let selected_route = &selected.selected;
    if selected_route.node.as_str() != signature.operation
        || signature.permit_identity().ok() != Some(selected.identity)
    {
        return Err(runmat_runtime::build_runtime_error(
            "managed placement changed its admitted invocation identity",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .build());
    }
    let native = if selected_route.candidate == candidates[0].identity_digest()
        && selected_route.kind == candidates[0].kind
    {
        None
    } else {
        Some(
            ready
                .iter()
                .position(|entry| {
                    entry.as_ref().is_some_and(|entry| {
                        let route = candidate(native_kind(entry.tier), native_facts(entry));
                        route.identity_digest() == selected_route.candidate
                            && route.kind == selected_route.kind
                    })
                })
                .ok_or_else(|| {
                    runmat_runtime::build_runtime_error(
                        "placement selected a route outside the admitted ready alternatives",
                    )
                    .with_identifier("RunMat:Placement:InvalidDecision")
                    .build()
                })?,
        )
    };
    Ok(Some(ReadyRouteSelection {
        service,
        selected: ReadyObservation::Managed(selected),
        native,
        started: std::time::Instant::now(),
        preparation: None,
    }))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct BorrowedProbe {
        owner: Rc<runmat_accelerate::placement::PlacementSession>,
        expected_facts: Digest,
        calls: Cell<usize>,
    }

    impl RuntimePlacementService for BorrowedProbe {
        fn managed_whole_work_available(&self) -> bool {
            true
        }

        fn whole_work_host_service_revision_ref<'a>(
            &self,
            program: Option<&'a runmat_execution::ProgramRevision>,
            host: Digest,
        ) -> Option<runmat_execution::PlacementRevisionRef<'a>> {
            self.owner
                .whole_work_host_service_revision_ref(program, host)
        }

        fn try_select_whole_work_borrowed_managed(
            &self,
            request: WholeWorkPlacementRequestRef<'_>,
        ) -> Result<Option<RuntimeManagedWholeWorkSelection>, RuntimeError> {
            self.calls.set(self.calls.get() + 1);
            assert_eq!(request.signature.runtime_facts, self.expected_facts);
            assert_eq!(request.candidates.len(), 1);
            assert!(request.candidates.iter().all(|candidate| matches!(
                candidate.resources.scratch_bytes,
                runmat_execution::ResourceByteBound::Unavailable(_)
            )));
            // No execution or fabricated finite-demand certificate is needed
            // to witness the borrowed metadata reaching its real selector.
            Ok(None)
        }

        fn plan(
            &self,
            _: runmat_execution::PlacementPlanRequest,
        ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
            panic!("profileless finite metadata must not use owned graph fallback")
        }
        fn observe(&self, _: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError> {
            Ok(())
        }
        fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
    }

    #[test]
    fn profileless_managed_metadata_reaches_selector_with_canonical_facts_and_unknown_demand() {
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let unit = futures::executor::block_on(session.compile_executable_unit(
            crate::ExecutableSource::new(
                "borrowed-facts@1",
                "borrowedFacts.m",
                "function y=borrowedFacts(x)\ny=x;\nend\n",
            ),
            None,
        ))
        .unwrap();
        let values = [runmat_value::Value::Num(2.0)];
        let invocation = ReadyInvocation {
            unit: &unit,
            preferred_function: Some("borrowedFacts"),
            arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&values),
            requested_outputs: 1,
            profile: None,
        };
        let owned: Vec<_> = values
            .iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect();
        let expected_facts = Digest::sha256_serialized(&(
            "runmat.core.ready-facts.v1",
            &unit.revision().source_digest,
            invocation.preferred_function,
            invocation.requested_outputs,
            owned,
        ))
        .unwrap();
        let probe = Rc::new(BorrowedProbe {
            owner: Rc::new(runmat_accelerate::placement::PlacementSession::default()),
            expected_facts,
            calls: Cell::new(0),
        });
        assert!(select_ready_route(
            ReadyRouteAdmission {
                service: probe.clone(),
                resources: Default::default(),
                managed: true
            },
            session.runtime_context(),
            invocation,
            &[None, None],
        )
        .unwrap()
        .is_none());
        assert_eq!(probe.calls.get(), 1);
        for reason in [
            runmat_execution::CancellationReason::User,
            runmat_execution::CancellationReason::Deadline,
        ] {
            let source = runmat_execution::cancellation::CancellationSource::new();
            let runtime = session
                .runtime_context()
                .clone()
                .with_cancellation_token(source.token());
            source.cancel(reason);
            assert!(ReadyRouteAdmission::prepare(&runtime).is_err());
            for managed in [false, true] {
                let error = select_ready_route(
                    ReadyRouteAdmission {
                        service: probe.clone(),
                        resources: Default::default(),
                        managed,
                    },
                    &runtime,
                    invocation,
                    &[None, None],
                )
                .err()
                .expect("stopped call fails before facts or selector");
                assert!(
                    runmat_runtime::runtime_error::error_has_control_cancellation_origin(&error)
                );
                let stopped = std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<ReadyRouteStopped>()
                    .unwrap();
                assert_eq!(stopped.0, reason);
                assert_eq!(probe.calls.get(), 1);
            }
        }
    }
}
