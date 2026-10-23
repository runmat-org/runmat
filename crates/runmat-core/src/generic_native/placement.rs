//! Complete-call selection between the original interpreter and all matching
//! ready native products. Tiering remains the preparation/hotness gate.

use std::rc::Rc;

use runmat_execution::placement::{
    PreparationPaybackEvidence, PreparationPaybackUnavailable, WholeWorkRouteIdentity,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, DigestStream, ExecutionCandidateKind,
    PlacementResourceSnapshot, PlacementSignature, PlacementSignatureRef, WholeWorkCandidate,
    WholeWorkConstraint, WholeWorkObservationOutcome, WholeWorkPlacementRequest,
    WholeWorkPlacementRequestRef,
};
use runmat_jit::entry::{PublishedEntry, PublishedTier};
use runmat_runtime::context::{
    host_allocation::{host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext, RuntimeManagedWholeWorkSelection, RuntimePlacementService,
    RuntimeWholeWorkSelection,
};
use runmat_runtime::RuntimeError;

use crate::ExecutableUnit;

mod managed;
use managed::select_managed_ready_route;

/// Borrowed facts for one complete invocation, shared by route selection and
/// final publication authentication. No argument or workspace payload is copied.
#[derive(Clone, Copy)]
pub(crate) struct ReadyInvocation<'a> {
    pub unit: &'a ExecutableUnit,
    pub preferred_function: Option<&'a str>,
    pub arguments: runmat_runtime::call::FunctionArgumentView<'a>,
    pub requested_outputs: usize,
    pub profile: Option<Digest>,
}

pub(crate) struct ReadyRouteSelection {
    service: Rc<dyn RuntimePlacementService>,
    selected: ReadyObservation,
    native: Option<usize>,
    started: std::time::Instant,
    preparation: Option<(Option<u64>, runmat_execution::SelectedExecutionCandidate)>,
}

enum ReadyObservation {
    Ordinary(RuntimeWholeWorkSelection),
    Managed(RuntimeManagedWholeWorkSelection),
}

/// Cheap, immutable authority for one ready-route comparison. Preparing it
/// before inspecting publications keeps disabled and finite-memory calls from
/// allocating a dependency snapshot solely for an ineligible comparison.
pub(crate) struct ReadyRouteAdmission {
    service: Rc<dyn RuntimePlacementService>,
    resources: runmat_runtime::context::RuntimeParallelResources,
    managed: bool,
}

impl ReadyRouteAdmission {
    pub(crate) fn prepare(runtime: &RuntimeContext) -> Result<Option<Self>, RuntimeError> {
        ready_route_checkpoint(runtime)?;
        let Some(service) = runtime.service_ports().placement().cloned() else {
            return Ok(None);
        };
        let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|source| {
            host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Placement:Admission",
                "ready-route placement allocation policy is unavailable",
                source,
            )
        })?;
        let resources = runtime
            .service_ports()
            .parallel()
            .map_or_else(Default::default, |parallel| parallel.placement_resources());
        Ok(Some(Self {
            service,
            resources,
            managed: policy == HostAllocationPolicy::Managed,
        }))
    }
}

impl ReadyRouteSelection {
    pub(crate) fn native(&self) -> Option<usize> {
        self.native
    }

    pub(crate) fn observe(self, outcome: WholeWorkObservationOutcome) {
        let elapsed_ns = runmat_time::duration_ns_saturating(self.started.elapsed());
        let recorded = match &self.selected {
            ReadyObservation::Ordinary(selected) => self
                .service
                .observe_selected_whole_work(selected, elapsed_ns, outcome),
            ReadyObservation::Managed(selected) => self
                .service
                .observe_managed_whole_work(selected, elapsed_ns, outcome),
        };
        if let Err(error) = recorded {
            log::warn!("ready native placement feedback was not recorded: {error}");
        }
        if let (Some((paid, baseline)), ReadyObservation::Ordinary(selected)) =
            (&self.preparation, &self.selected)
        {
            // Both samples belong to this complete-call signature. Actual
            // compilation evidence is not added to a warm execution quote.
            let payback = PreparationPaybackEvidence::from_observations(
                *paid,
                self.service.observed_candidate_duration(
                    &selected.decision.signature,
                    &baseline.candidate,
                    baseline.kind,
                ),
                self.service.observed_candidate_duration(
                    &selected.decision.signature,
                    &selected.decision.selected.candidate,
                    selected.decision.selected.kind,
                ),
            );
            log::debug!(target: "runmat_core::native_tiering", "ready native preparation payback: {payback:?}");
        }
    }
}

/// `None` means there is no admitted policy selection; it does not invent a
/// cost or override the existing tiering decision. The portable owned request
/// is deliberately ordinary-only until finite request metadata is admitted.
pub(crate) fn select_ready_route(
    admission: ReadyRouteAdmission,
    runtime: &RuntimeContext,
    invocation: ReadyInvocation<'_>,
    ready: &[Option<PublishedEntry>; 2],
) -> Result<Option<ReadyRouteSelection>, RuntimeError> {
    ready_route_checkpoint(runtime)?;
    let ReadyRouteAdmission {
        service,
        resources,
        managed,
    } = admission;
    if managed {
        return select_managed_ready_route(service, resources, runtime, invocation, ready);
    }
    let Some(signature) = ready_signature(service.as_ref(), invocation)? else {
        return Ok(None);
    };
    let unit = invocation.unit;
    let interpreter = candidate(
        ExecutionCandidateKind::SharedRuntime,
        interpreter_facts(unit),
    );
    let expected_interpreter = interpreter.selected(&signature.operation);
    let mut candidates = Vec::with_capacity(3);
    candidates.push(interpreter);
    let expected_native = ready.each_ref().map(|entry| {
        entry.as_ref().map(|entry| {
            let native = candidate(native_kind(entry.tier), native_facts(entry));
            let identity = native.selected(&signature.operation);
            candidates.push(native);
            identity
        })
    });
    let selected = match service.try_select_whole_work(WholeWorkPlacementRequest {
        signature,
        constraint: WholeWorkConstraint::AnyLegal,
        candidates,
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: resources.cpu_millicores_available,
            memory_available_bytes: resources.memory_available_bytes,
            cancellation_requested: runtime.is_cancelled(),
            providers: Vec::new(),
            epoch: resources.epoch,
        },
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
    let native = if selected.decision.selected == expected_interpreter {
        None
    } else if let Some(index) = expected_native
        .iter()
        .position(|expected| expected.as_ref() == Some(&selected.decision.selected))
    {
        Some(index)
    } else {
        return Err(runmat_runtime::build_runtime_error(
            "placement selected a route outside the admitted ready alternatives",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .build());
    };
    let preparation = native
        .and_then(|index| ready[index].as_ref())
        .map(|entry| (entry.compilation.total_ns(), expected_interpreter));
    Ok(Some(ReadyRouteSelection {
        service,
        selected: ReadyObservation::Ordinary(selected),
        native,
        started: std::time::Instant::now(),
        preparation,
    }))
}

/// Keep the exact originating stop reason with Runtime's existing typed
/// execution-control cause. A deadline is not a route-capability decline.
#[derive(Debug)]
struct ReadyRouteStopped(runmat_execution::CancellationReason);

impl std::fmt::Display for ReadyRouteStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ready-route execution stopped: {:?}", self.0)
    }
}

impl std::error::Error for ReadyRouteStopped {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&runmat_runtime::execution::ExecutionServiceError::Cancelled)
    }
}

fn ready_route_checkpoint(runtime: &RuntimeContext) -> Result<(), RuntimeError> {
    let Some(reason) = runtime.cancellation().stop_reason() else {
        return Ok(());
    };
    let mut error =
        runmat_runtime::execution::ExecutionServiceError::Cancelled.into_runtime_error();
    let suffix = if reason == runmat_execution::CancellationReason::Deadline {
        "ExecutionDeadline"
    } else {
        "ExecutionCancelled"
    };
    error.identifier = Some(format!("{}:{suffix}", runtime.error_namespace()));
    error.source = Some(Box::new(ReadyRouteStopped(reason)));
    Err(error)
}

fn ready_signature(
    service: &dyn RuntimePlacementService,
    invocation: ReadyInvocation<'_>,
) -> Result<Option<PlacementSignature>, RuntimeError> {
    let unit = invocation.unit;
    let Some(revision) = service.whole_work_host_service_revision(
        Some(unit.revision().program_revision.clone()),
        Digest::sha256(b"runmat.core.ready-interpreter-native.v1"),
    ) else {
        return Ok(None);
    };

    let Some(runtime_facts) = ready_runtime_facts(invocation, true)? else {
        return Ok(None);
    };
    Ok(Some(PlacementSignature {
        region: None,
        operation: "core.ready-invocation".to_owned(),
        runtime_facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    }))
}

/// Preserve the existing profile/facts digest byte encoding in both request
/// forms. Only the ordinary owner may construct a recursive owned fact tree.
fn ready_runtime_facts(
    invocation: ReadyInvocation<'_>,
    allow_owned: bool,
) -> Result<Option<Digest>, RuntimeError> {
    let ReadyInvocation {
        unit,
        preferred_function,
        arguments,
        requested_outputs,
        profile,
    } = invocation;
    let encoded = if let Some(profile) = profile {
        Digest::sha256_serialized(&(
            "runmat.core.ready-profile.v1",
            &unit.revision().source_digest,
            preferred_function,
            requested_outputs,
            profile,
        ))
    } else if let Some(facts) = runmat_runtime::value_fact::BorrowedValueFacts::new(arguments) {
        Digest::sha256_serialized(&(
            "runmat.core.ready-facts.v1",
            &unit.revision().source_digest,
            preferred_function,
            requested_outputs,
            facts,
        ))
    } else if allow_owned {
        let facts = arguments
            .iter()
            .map(runmat_runtime::value_fact::value_fact)
            .collect::<Vec<_>>();
        Digest::sha256_serialized(&(
            "runmat.core.ready-facts.v1",
            &unit.revision().source_digest,
            preferred_function,
            requested_outputs,
            facts,
        ))
    } else {
        return Ok(None);
    };
    encoded.map(Some).map_err(|error| {
        runmat_runtime::build_runtime_error("ready-route facts could not be encoded")
            .with_identifier("RunMat:Placement:InvalidFacts")
            .with_source(error)
            .build()
    })
}

/// Statistics-only quotation for an identical, budget-retired recipe. Neither
/// a historical executable nor an observation/selection permit is recreated.
pub(crate) fn repreparation_evidence(
    runtime: &RuntimeContext,
    invocation: ReadyInvocation<'_>,
    quote: &runmat_jit::entry::NativePreparationQuote,
) -> Result<PreparationPaybackEvidence, RuntimeError> {
    ready_route_checkpoint(runtime)?;
    let unavailable = || {
        PreparationPaybackEvidence::Unavailable(PreparationPaybackUnavailable::Baseline(
            CandidateCostUnavailable::NoObservations,
        ))
    };
    let Some(service) = runtime.service_ports().placement() else {
        return Ok(unavailable());
    };
    // This statistics request uses owned signature/fact metadata. Do not
    // smuggle it through the managed allocation-free request protocol.
    if HostAllocationReceipt::policy(Some(runtime)).map_err(|source| {
        host_allocation_runtime_error(
            Some(runtime),
            "RunMat:Placement:Admission",
            "preparation evidence allocation policy is unavailable",
            source,
        )
    })? != HostAllocationPolicy::Unbounded
    {
        return Ok(unavailable());
    }
    let Some(signature) = ready_signature(service.as_ref(), invocation)? else {
        return Ok(unavailable());
    };
    let baseline = candidate(
        ExecutionCandidateKind::SharedRuntime,
        interpreter_facts(invocation.unit),
    )
    .selected(&signature.operation);
    let prepared_kind = native_kind(quote.tier);
    let retired_identity = WholeWorkRouteIdentity {
        kind: prepared_kind,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Unknown,
        presentation: &CandidateOutputPresentation::Direct,
        route_facts: native_facts_for(
            quote.publication,
            quote.entrypoint,
            quote.tier,
            &quote.dependencies,
        ),
    }
    .digest()
    .to_string();
    Ok(PreparationPaybackEvidence::from_observations(
        quote.compilation.total_ns(),
        service.observed_candidate_duration(&signature, &baseline.candidate, baseline.kind),
        service.observed_candidate_duration(&signature, &retired_identity, prepared_kind),
    ))
}

fn candidate(kind: ExecutionCandidateKind, route_facts: Digest) -> WholeWorkCandidate {
    WholeWorkCandidate {
        kind,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        output_residency: CandidateOutputResidency::Unknown,
        presentation: CandidateOutputPresentation::Direct,
        route_facts,
        cost: CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate),
        resources: CandidateResourceDemand {
            cpu_millicores: 1_000,
            ..Default::default()
        },
        transactional_results: false,
    }
}

fn native_kind(tier: PublishedTier) -> ExecutionCandidateKind {
    match tier {
        PublishedTier::Generic => ExecutionCandidateKind::GenericNativeCpu,
        PublishedTier::Specialized { .. } => ExecutionCandidateKind::SpecializedNativeCpu,
    }
}

fn interpreter_facts(unit: &ExecutableUnit) -> Digest {
    let mut stream = DigestStream::new();
    stream.update(b"runmat.core.interpreted-executable.v1\0");
    stream.update(unit.revision().source_digest.as_bytes());
    stream.finish()
}

fn native_facts(ready: &PublishedEntry) -> Digest {
    native_facts_for(
        ready.publication,
        ready.entrypoint,
        ready.tier,
        &ready.dependencies,
    )
}

fn native_facts_for(
    publication: u64,
    entrypoint: runmat_types::FunctionId,
    tier: PublishedTier,
    dependencies: &runmat_jit::invalidation::DependencySnapshot,
) -> Digest {
    let mut stream = DigestStream::new();
    stream.update(b"runmat.core.published-native-executable.v1\0");
    stream.update(&publication.to_le_bytes());
    stream.update(&entrypoint.0.to_le_bytes());
    match tier {
        PublishedTier::Generic => stream.update(&[0]),
        PublishedTier::Specialized { profile } => {
            stream.update(&[1]);
            stream.update(profile.bytes());
        }
    }
    for (key, generation) in dependencies.iter() {
        use runmat_jit::invalidation::DependencyKey;
        match key {
            DependencyKey::Program(name) => {
                stream.update(&[0]);
                framed(&mut stream, name.as_bytes());
            }
            DependencyKey::Catalog(name) => {
                stream.update(&[1]);
                framed(&mut stream, name.as_bytes());
            }
            DependencyKey::SessionFunction(name) => {
                stream.update(&[2]);
                framed(&mut stream, name.as_bytes());
            }
            DependencyKey::SessionCatalog => stream.update(&[3]),
            DependencyKey::Provider(device) => {
                stream.update(&[4]);
                stream.update(&device.to_le_bytes());
            }
        }
        stream.update(&generation.get().to_le_bytes());
    }
    stream.finish()
}

fn framed(stream: &mut DigestStream, bytes: &[u8]) {
    stream.update(&(bytes.len() as u64).to_le_bytes());
    stream.update(bytes);
}
