//! Complete-call selection between the original interpreter and an already
//! published native product. Tiering remains the preparation/hotness gate.

use std::rc::Rc;

use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidateExecutionLocation,
    CandidateOutputPresentation, CandidateOutputResidency, CandidatePreparationState,
    CandidateResourceDemand, Digest, DigestStream, ExecutionCandidateKind,
    PlacementResourceSnapshot, PlacementSignature, WholeWorkCandidate, WholeWorkConstraint,
    WholeWorkObservationOutcome, WholeWorkPlacementRequest,
};
use runmat_jit::entry::{PublishedEntry, PublishedTier};
use runmat_runtime::context::{
    host_allocation::{host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext, RuntimePlacementService, RuntimeWholeWorkSelection,
};
use runmat_runtime::RuntimeError;

use crate::ExecutableUnit;

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
    selected: RuntimeWholeWorkSelection,
    native: bool,
    started: std::time::Instant,
}

/// Cheap, immutable authority for one ready-route comparison. Preparing it
/// before inspecting publications keeps disabled and finite-memory calls from
/// allocating a dependency snapshot solely for an ineligible comparison.
pub(crate) struct ReadyRouteAdmission {
    service: Rc<dyn RuntimePlacementService>,
    resources: runmat_runtime::context::RuntimeParallelResources,
}

impl ReadyRouteAdmission {
    pub(crate) fn prepare(runtime: &RuntimeContext) -> Result<Option<Self>, RuntimeError> {
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
        if policy != HostAllocationPolicy::Unbounded {
            return Ok(None);
        }
        let resources = runtime
            .service_ports()
            .parallel()
            .map_or_else(Default::default, |parallel| parallel.placement_resources());
        if resources.memory_available_bytes.is_some() || resources.cpu_millicores_available < 1_000
        {
            return Ok(None);
        }
        Ok(Some(Self { service, resources }))
    }
}

impl ReadyRouteSelection {
    pub(crate) fn native(&self) -> bool {
        self.native
    }

    pub(crate) fn observe(self, outcome: WholeWorkObservationOutcome) {
        let elapsed_ns = runmat_time::duration_ns_saturating(self.started.elapsed());
        if let Err(error) =
            self.service
                .observe_selected_whole_work(&self.selected, elapsed_ns, outcome)
        {
            log::warn!("ready native placement feedback was not recorded: {error}");
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
    ready: &PublishedEntry,
) -> Result<Option<ReadyRouteSelection>, RuntimeError> {
    let ReadyInvocation {
        unit,
        preferred_function,
        arguments,
        requested_outputs,
        profile,
    } = invocation;
    let ReadyRouteAdmission { service, resources } = admission;
    let Some(revision) = service.whole_work_host_service_revision(
        Some(unit.revision().program_revision.clone()),
        Digest::sha256(b"runmat.core.ready-interpreter-native.v1"),
    ) else {
        return Ok(None);
    };

    let runtime_facts = if let Some(profile) = profile {
        Digest::sha256_serialized(&(
            "runmat.core.ready-profile.v1",
            &unit.revision().source_digest,
            preferred_function,
            requested_outputs,
            profile,
        ))
    } else {
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
    }
    .map_err(|error| {
        runmat_runtime::build_runtime_error("ready-route facts could not be encoded")
            .with_identifier("RunMat:Placement:InvalidFacts")
            .with_source(error)
            .build()
    })?;
    let signature = PlacementSignature {
        region: None,
        operation: "core.ready-invocation".to_owned(),
        runtime_facts,
        provider_provenance: None,
        graph_output_authority: None,
        revision,
    };
    let interpreter = candidate(
        ExecutionCandidateKind::SharedRuntime,
        interpreter_facts(unit),
    );
    let native = candidate(native_kind(ready.tier), native_facts(ready));
    let expected_interpreter = interpreter.selected(&signature.operation);
    let expected_native = native.selected(&signature.operation);
    let selected = service.select_whole_work(WholeWorkPlacementRequest {
        signature,
        constraint: WholeWorkConstraint::AnyLegal,
        candidates: vec![interpreter, native],
        resources: PlacementResourceSnapshot {
            cpu_millicores_available: resources.cpu_millicores_available,
            memory_available_bytes: resources.memory_available_bytes,
            cancellation_requested: runtime.is_cancelled(),
            providers: Vec::new(),
            epoch: resources.epoch,
        },
        deterministic: false,
        require_transactional_results: false,
    })?;
    if !selected.observation_is_current() {
        return Ok(None);
    }
    let native = if selected.decision.selected == expected_native {
        true
    } else if selected.decision.selected == expected_interpreter {
        false
    } else {
        return Err(runmat_runtime::build_runtime_error(
            "placement selected a route outside the admitted ready alternatives",
        )
        .with_identifier("RunMat:Placement:InvalidDecision")
        .build());
    };
    Ok(Some(ReadyRouteSelection {
        service,
        selected,
        native,
        started: std::time::Instant::now(),
    }))
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
    let mut stream = DigestStream::new();
    stream.update(b"runmat.core.published-native-executable.v1\0");
    stream.update(&ready.publication.to_le_bytes());
    stream.update(&ready.entrypoint.0.to_le_bytes());
    match ready.tier {
        PublishedTier::Generic => stream.update(&[0]),
        PublishedTier::Specialized { profile } => {
            stream.update(&[1]);
            stream.update(profile.bytes());
        }
    }
    for (key, generation) in ready.dependencies.iter() {
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
