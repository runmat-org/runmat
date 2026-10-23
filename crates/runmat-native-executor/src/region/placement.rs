use runmat_execution::*;
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;

use super::plan::OptimizedRegionPlan;

pub(crate) enum RegionPlacement {
    Unavailable,
    Generic(RegionObservation),
    Vectorized(RegionObservation),
}

/// One selected execution, completed explicitly or failed when its scope exits.
pub(crate) struct RegionObservation {
    runtime: RuntimeContext,
    started: runmat_time::Instant,
    owner: Option<RegionObservationOwner>,
}

enum RegionObservationOwner {
    Original {
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        observation:
            Box<dyn runmat_runtime::numeric_region::service::NativeOriginalRegionObservation>,
    },
    WholeWork(runmat_runtime::context::RuntimeWholeWorkSelection),
    ManagedWholeWork(runmat_runtime::context::RuntimeManagedWholeWorkSelection),
    Composed {
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        observation:
            Box<dyn runmat_runtime::numeric_region::service::NativeNumericRegionCpuObservation>,
    },
}

impl RegionObservation {
    pub(crate) fn runtime(&self) -> &RuntimeContext {
        &self.runtime
    }
    pub(crate) fn original(
        runtime: RuntimeContext,
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        observation: Option<
            Box<dyn runmat_runtime::numeric_region::service::NativeOriginalRegionObservation>,
        >,
    ) -> Self {
        Self {
            runtime,
            started: runmat_time::Instant::now(),
            owner: observation
                .map(|observation| RegionObservationOwner::Original { right, observation }),
        }
    }

    pub(crate) fn original_audit(
        &self,
    ) -> Option<runmat_runtime::numeric_region::service::SourceRegionRouteAudit> {
        match self.owner.as_ref() {
            Some(RegionObservationOwner::Original { observation, .. }) => observation.audit(),
            _ => None,
        }
    }

    pub(crate) fn finish_original(
        &mut self,
    ) -> Option<runmat_runtime::execution::RuntimeContinuationFuture<'static, ()>> {
        if !matches!(self.owner, Some(RegionObservationOwner::Original { .. })) {
            return None;
        }
        let Some(RegionObservationOwner::Original { right, observation }) = self.owner.take()
        else {
            unreachable!()
        };
        // Completion measurement is optional after the source publications.
        // Denial/cancellation withholds the sample, never replays source work.
        observation.complete(self.runtime.clone(), right).ok()
    }
    pub(crate) fn composed(
        runtime: RuntimeContext,
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        observation: Option<
            Box<dyn runmat_runtime::numeric_region::service::NativeNumericRegionCpuObservation>,
        >,
    ) -> Self {
        Self {
            runtime,
            started: runmat_time::Instant::now(),
            owner: observation
                .map(|observation| RegionObservationOwner::Composed { right, observation }),
        }
    }
    pub fn complete(mut self) {
        self.report(true);
    }

    fn report(&mut self, succeeded: bool) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        let outcome = if succeeded {
            WholeWorkObservationOutcome::Success
        } else {
            // Cancellation/abandonment cannot poison candidate failure rates.
            if self.runtime.is_cancelled() {
                WholeWorkObservationOutcome::Cancelled
            } else {
                WholeWorkObservationOutcome::Abandoned
            }
        };
        match owner {
            RegionObservationOwner::Original { .. } => (),
            RegionObservationOwner::Composed { right, observation } => {
                observation.finish(&self.runtime, right, succeeded);
            }
            RegionObservationOwner::WholeWork(selected) => {
                let Some(service) = self.runtime.service_ports().placement() else {
                    return;
                };
                let _ = service.observe_selected_whole_work(
                    &selected,
                    runmat_time::duration_ns_saturating(self.started.elapsed()).max(1),
                    outcome,
                );
            }
            RegionObservationOwner::ManagedWholeWork(selected) => {
                let Some(service) = self.runtime.service_ports().placement() else {
                    return;
                };
                let _ = service.observe_managed_whole_work(
                    &selected,
                    runmat_time::duration_ns_saturating(self.started.elapsed()).max(1),
                    outcome,
                );
            }
        }
    }
}

impl Drop for RegionObservation {
    fn drop(&mut self) {
        if matches!(
            self.owner.as_ref(),
            Some(RegionObservationOwner::Composed { .. } | RegionObservationOwner::Original { .. })
        ) {
            // Dropped continuation != observed operational failure. Releasing
            // its real permit/receipt is sufficient; no zero/partial sample.
            self.owner.take();
            return;
        }
        self.report(false);
    }
}

pub(crate) fn choose_region(
    runtime: &RuntimeContext,
    plan: &OptimizedRegionPlan,
    inputs: &[&Value],
    workload: runmat_runtime::numeric_region::NumericRegionWorkload,
) -> crate::NativeExecutorResult<RegionPlacement> {
    let started = runmat_time::Instant::now();
    // This standalone evaluator remains host dense-F64 only. In particular,
    // neither explicit nor automatic resident values enter this CPU route.
    if runmat_runtime::numeric_region::workload(inputs) != Some(workload) {
        return Ok(RegionPlacement::Unavailable);
    }
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(RegionPlacement::Unavailable);
    };
    let Some(signature) = signature_ref(runtime, plan, inputs) else {
        return Ok(RegionPlacement::Unavailable);
    };
    use runmat_runtime::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationPlan, HostAllocationReceipt,
    };
    // Candidate identity Strings are the only new owned temporary geometry.
    // Facts/program revision stay borrowed for finite managed selection.
    let bytes = runmat_runtime::numeric_region::service::NativeNumericRegionCpuCandidates::identity_storage_bytes(plan.region);
    let allocation = |source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Placement:NativeRegion",
            "native region candidate metadata",
            source,
        )
    };
    let metadata = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(bytes, 0).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let Some(cpu) = runmat_runtime::numeric_region::service::NativeNumericRegionCpuCandidates::for_verified_program(
        plan.region, plan.arithmetic_operations, workload, plan.program.outputs.len()) else { return Ok(RegionPlacement::Unavailable); };
    let vectorized = cpu.vectorized;
    let unknown = ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound);
    let candidates = [
        WholeWorkCandidate {
            kind: ExecutionCandidateKind::OriginalNativeRegion,
            execution_location: CandidateExecutionLocation::NativeSource,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Unknown,
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&(
                "native-unbound-source-v1",
                plan.region,
                signature.runtime_facts,
            ))
            .expect("typed native scope serializes"),
            cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoObservations),
            resources: CandidateResourceDemand {
                retained_bytes: unknown,
                scratch_bytes: unknown,
                ..Default::default()
            },
            transactional_results: false,
        },
        WholeWorkCandidate {
            kind: vectorized.descriptor.kind,
            execution_location: CandidateExecutionLocation::Host,
            preparation: vectorized.descriptor.preparation,
            output_residency: CandidateOutputResidency::Host,
            presentation: CandidateOutputPresentation::Direct,
            route_facts: Digest::sha256_serialized(&vectorized.descriptor)
                .expect("verified native CPU descriptor serializes"),
            cost: CandidateCostEvidence::Estimated(vectorized.descriptor.cost),
            resources: vectorized.resources,
            transactional_results: true,
        },
    ];
    let parallel = runtime
        .service_ports()
        .parallel()
        .map(|service| service.placement_resources())
        .unwrap_or_default();
    let resources = PlacementResourceSnapshot {
        cpu_millicores_available: parallel.cpu_millicores_available,
        memory_available_bytes: parallel.memory_available_bytes,
        cancellation_requested: runtime.is_cancelled(),
        providers: Vec::new(),
        epoch: parallel.epoch,
    };
    let constraint = WholeWorkConstraint::AnyLegal;
    let request = WholeWorkPlacementRequestRef {
        signature,
        constraint,
        candidates: &candidates,
        resources: &resources,
        deterministic: false,
        require_transactional_results: false,
    };
    let expected = candidates[1].identity_digest();
    let checkpoint = || -> Result<(), runmat_runtime::RuntimeError> {
        if let Some(metadata) = &metadata {
            metadata.checkpoint(runtime).map_err(allocation)?;
        }
        if runtime.is_cancelled() {
            return Err(allocation(
                runmat_runtime::context::host_allocation::HostAllocationError::Cancelled,
            ));
        }
        Ok(())
    };
    let validate_selected =
        |candidate: &WholeWorkCandidate| -> Result<(), runmat_runtime::RuntimeError> {
            single_candidate_resources_admitted(
                CandidateResourceFacts {
                    kind: candidate.kind,
                    location: candidate.execution_location,
                    output_residency: candidate.output_residency,
                    presentation: &candidate.presentation,
                    demand: candidate.resources,
                    transactional_results: candidate.transactional_results,
                },
                &resources,
                false,
            )
            .map_err(|source| {
                runmat_runtime::runtime_error::semantic_error_with_source(
                    "RunMat:Placement:InvalidNativeRegion",
                    "selected native route has no admitted resource envelope",
                    source,
                )
            })
        };
    let owner = if resources.memory_available_bytes.is_some() {
        let Some(selected) = service.try_select_whole_work_borrowed_managed(request)? else {
            return Ok(RegionPlacement::Unavailable);
        };
        if selected.identity
            != signature.permit_identity().map_err(|source| {
                runmat_runtime::runtime_error::semantic_error_with_source(
                    "RunMat:Placement:InvalidNativeRegion",
                    "invalid native selection scope",
                    source,
                )
            })?
        {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Placement:InvalidNativeRegion",
                "native managed selection changed scope",
            )
            .into());
        }
        if selected.selected.kind == ExecutionCandidateKind::OriginalNativeRegion
            && selected.selected.candidate == candidates[0].identity_digest()
        {
            checkpoint()?;
            validate_selected(&candidates[0])?;
            return Ok(RegionPlacement::Generic(RegionObservation {
                runtime: runtime.clone(),
                started,
                owner: None,
            }));
        }
        if selected.selected.kind != ExecutionCandidateKind::VectorizedNativeCpu
            || selected.selected.candidate != expected
        {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Placement:InvalidNativeRegion",
                "native selection changed candidate",
            )
            .into());
        }
        RegionObservationOwner::ManagedWholeWork(selected)
    } else {
        let request = WholeWorkPlacementRequest {
            signature: signature.to_owned(),
            constraint,
            candidates: candidates.to_vec(),
            resources: resources.clone(),
            deterministic: false,
            require_transactional_results: false,
        };
        let Some(selected) = service.try_select_whole_work(request)? else {
            return Ok(RegionPlacement::Unavailable);
        };
        if selected.decision.signature.cache_key() != signature.cache_key() {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Placement:InvalidNativeRegion",
                "native selection changed scope",
            )
            .into());
        }
        let identity = selected
            .decision
            .selected
            .candidate
            .parse::<Digest>()
            .map_err(|source| {
                runmat_runtime::runtime_error::semantic_error_with_source(
                    "RunMat:Placement:InvalidNativeRegion",
                    "native candidate identity is not canonical",
                    source,
                )
            })?;
        if selected.decision.selected.kind == ExecutionCandidateKind::OriginalNativeRegion
            && identity == candidates[0].identity_digest()
        {
            checkpoint()?;
            validate_selected(&candidates[0])?;
            return Ok(RegionPlacement::Generic(RegionObservation {
                runtime: runtime.clone(),
                started,
                owner: None,
            }));
        }
        if selected.decision.selected.kind != ExecutionCandidateKind::VectorizedNativeCpu
            || identity != expected
        {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Placement:InvalidNativeRegion",
                "native selection changed candidate",
            )
            .into());
        }
        RegionObservationOwner::WholeWork(selected)
    };
    checkpoint()?;
    validate_selected(&candidates[1])?;
    Ok(RegionPlacement::Vectorized(RegionObservation {
        runtime: runtime.clone(),
        started,
        owner: Some(owner),
    }))
}

fn signature_ref<'a>(
    runtime: &'a RuntimeContext,
    plan: &OptimizedRegionPlan,
    inputs: &'a [&'a Value],
) -> Option<PlacementSignatureRef<'a>> {
    let facts = runmat_runtime::value_fact::BorrowedValueFacts::from_references(inputs)?;
    Some(PlacementSignatureRef {
        region: Some(plan.region),
        operation: "native.numeric_region",
        runtime_facts: Digest::sha256_serialized(&facts).ok()?,
        provider_provenance: None,
        graph_output_authority: None,
        revision: PlacementRevisionRef {
            program: runtime.program_revision(),
            catalog: Digest::sha256(b"runmat-placement-catalog-v1"),
            compiler: Digest::sha256(b"runmat-native-source-vector-region-v2"),
            authority: PlacementExecutionAuthorityRevision::HostService(Digest::sha256(
                b"native-source-owner",
            )),
            policy: Digest::sha256(b"shared-placement-v1"),
        },
    })
}

#[cfg(test)]
fn signature(
    runtime: &RuntimeContext,
    plan: &OptimizedRegionPlan,
    inputs: &[&Value],
) -> PlacementSignature {
    signature_ref(runtime, plan, inputs)
        .expect("supported numeric facts")
        .to_owned()
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
