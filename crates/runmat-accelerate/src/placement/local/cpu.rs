//! One host-route carrier; numerical graph descriptors require real estimates.
use runmat_execution::{
    CandidateCostEvidence, CandidateExecutionLocation, CandidateOutputPresentation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand, Digest,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkCandidate,
};
use runmat_types::{RegionGuardId, RegionId};

pub(crate) struct LocalCpuCandidate {
    pub(super) identity: String,
    pub(super) region: Option<RegionId>,
    pub(super) guards: Vec<RegionGuardId>,
    pub(super) route: WholeWorkCandidate,
    pub(super) native_alternative: Option<runmat_execution::PlacementGraphCandidate>,
    pub(super) native_host_alternative: Option<WholeWorkCandidate>,
    pub(super) native_origin:
        Option<runmat_runtime::numeric_region::service::NativeNumericRegionSelectionAuthority>,
}

impl LocalCpuCandidate {
    pub(crate) fn native_region(
        candidates: Option<
            &runmat_runtime::numeric_region::service::NativeNumericRegionCpuCandidates,
        >,
        provider: &dyn runmat_accelerate_api::AccelProvider,
        right: &runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        source_host: Option<&crate::fusion_exec::host_readback::CpuHostResources>,
    ) -> (Self, super::multi::LocalCpuAdmission) {
        let (region, site) = right.identity();
        let identity = format!("native.original.f{}.r{}", region.function.0, region.ordinal);
        let tail = provider.region_queue_tail_resources();
        let native_host_alternative = source_host.map(|host| WholeWorkCandidate {
            kind: ExecutionCandidateKind::SpecializedNativeCpu,
            execution_location: CandidateExecutionLocation::Host,
            preparation: CandidatePreparationState::Ready,
            output_residency: CandidateOutputResidency::Host,
            presentation: host.presentation.clone(),
            // Scalar arithmetic alone is not the complete source-host route.
            // Canonical dispatch, admitted inputs/outputs and optional readback
            // belong to this same full-call observation, not the separate
            // generic numeric evaluator's compiler prior.
            cost: CandidateCostEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::NoObservations,
            ),
            resources: host.resources,
            route_facts: Digest::sha256_serialized(&(
                "native-canonical-source-host-v3",
                region,
                site.function,
                site.block,
                site.position,
                site.phase.0,
                site.ordinal,
                host.download_bytes,
                host.readback_count,
                host.scratch_bytes,
                &host.resources,
            ))
            .expect("typed native source-host facts serialize"),
            transactional_results: false,
        });
        let original_admission = if tail.is_some() {
            super::multi::LocalCpuAdmission::NativeRegionAtUse
        } else {
            // Ordinary source execution remains available through decline.
            // An unmeasured comparative route cannot explore forever without
            // its physical completion/audit capability.
            super::multi::LocalCpuAdmission::Unavailable
        };
        (Self {
            identity,
            region: Some(region),
            guards: Vec::new(),
            native_alternative: candidates.map(|candidates| candidates.vectorized.clone()),
            native_host_alternative,
            native_origin: Some(right.selection_authority()),
            route: WholeWorkCandidate {
                kind: ExecutionCandidateKind::OriginalNativeRegion,
                execution_location: CandidateExecutionLocation::NativeRegion { device_id: provider.device_id() },
                preparation: CandidatePreparationState::Ready,
                output_residency: CandidateOutputResidency::Unknown,
                presentation: CandidateOutputPresentation::Direct,
                route_facts: Digest::sha256_serialized(&("native-original-source-v1", region, site.function, site.block, site.position, site.phase.0, site.ordinal)).expect("native site identity serializes"),
                cost: CandidateCostEvidence::Unavailable(runmat_execution::CandidateCostUnavailable::NoObservations),
                resources: CandidateResourceDemand {
                    // No tail means no admitted comparative original route.
                    queue_slots: tail.map_or(0, |tail| tail.queue_slots),
                    scratch_bytes: tail.map(|tail| ResourceByteBound::Known(tail.provider_scratch_bytes)).unwrap_or(ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound)),
                    provider_host_workspace: tail.and_then(|tail| tail.completion_control_host_bytes)
                        .and_then(|bytes| runmat_runtime::context::host_allocation::HostAllocationPlan::retained_metadata_bytes(bytes).ok())
                        .map(ResourceByteBound::Known),
                    ..Default::default()
                },
                transactional_results: false,
            },
        }, original_admission)
    }
    /// Source-ordered original VM instructions, not a host-only fallback.
    /// The installed-region owner supplies `source_region`; placement never
    /// derives it from a display label or an input tensor alone.
    pub(crate) fn original_vm_region(
        source_region: Digest,
        device_id: u32,
        tail: runmat_accelerate_api::ProviderRegionTailResources,
        cost: CandidateCostEvidence,
    ) -> Self {
        let identity = "vm.original.region".to_owned();
        Self {
            identity,
            native_alternative: None,
            native_host_alternative: None,
            native_origin: None,
            region: None,
            guards: Vec::new(),
            route: WholeWorkCandidate {
                kind: ExecutionCandidateKind::OriginalVmRegion,
                execution_location: CandidateExecutionLocation::VmRegion { device_id },
                preparation: CandidatePreparationState::Ready,
                output_residency: CandidateOutputResidency::Unknown,
                presentation: CandidateOutputPresentation::Direct,
                route_facts: source_region,
                cost,
                resources: CandidateResourceDemand {
                    // One provider-owned tail submission orders every nested
                    // ordinary WGPU submission before cost feedback. Browser
                    // also carries its exact admitted map-fence bytes.
                    queue_slots: tail.queue_slots,
                    scratch_bytes: ResourceByteBound::Known(tail.provider_scratch_bytes),
                    provider_host_workspace: Some(tail.completion_control_host_bytes
                        .and_then(|bytes| runmat_runtime::context::host_allocation::HostAllocationPlan::retained_metadata_bytes(bytes).ok())
                        .map(ResourceByteBound::Known)
                        .unwrap_or(ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound))),
                    ..CandidateResourceDemand::default()
                },
                transactional_results: false,
            },
        }
    }
    pub(crate) fn with_presentation(mut self, presentation: CandidateOutputPresentation) -> Self {
        self.route.presentation = presentation;
        self
    }
    pub(crate) fn host(identity: String, cost: CandidateCostEvidence) -> Self {
        let guards = Vec::<RegionGuardId>::new();
        let route_facts =
            Digest::sha256_serialized(&(&identity, &guards)).expect("host route facts serialize");
        Self {
            identity,
            native_alternative: None,
            native_host_alternative: None,
            native_origin: None,
            region: None,
            guards,
            route: WholeWorkCandidate {
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                output_residency: CandidateOutputResidency::Host,
                presentation: CandidateOutputPresentation::Direct,
                route_facts,
                cost,
                resources: legacy_resources(cost),
                transactional_results: true,
            },
        }
    }

    pub(super) fn into_priced(self) -> Option<ExecutionCandidateDescriptor> {
        Some(ExecutionCandidateDescriptor {
            identity: self.identity,
            region: self.region,
            guards: self.guards,
            kind: self.route.kind,
            execution_location: self.route.execution_location,
            preparation: self.route.preparation,
            output_residency: self.route.output_residency,
            presentation: self.route.presentation,
            cost: self.route.cost.estimate()?,
        })
    }
}

impl From<ExecutionCandidateDescriptor> for LocalCpuCandidate {
    fn from(cpu: ExecutionCandidateDescriptor) -> Self {
        let cost = CandidateCostEvidence::Estimated(cpu.cost);
        Self {
            native_alternative: None,
            native_host_alternative: None,
            native_origin: None,
            route: WholeWorkCandidate {
                kind: cpu.kind,
                execution_location: cpu.execution_location,
                preparation: cpu.preparation,
                output_residency: cpu.output_residency,
                presentation: cpu.presentation,
                route_facts: Digest::sha256_serialized(&(&cpu.identity, &cpu.guards))
                    .expect("host route facts serialize"),
                cost,
                resources: legacy_resources(cost),
                transactional_results: true,
            },
            identity: cpu.identity,
            region: cpu.region,
            guards: cpu.guards,
        }
    }
}

fn legacy_resources(cost: CandidateCostEvidence) -> CandidateResourceDemand {
    CandidateResourceDemand {
        cpu_millicores: 1_000,
        retained_bytes: ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoRetainedValueBound,
        ),
        scratch_bytes: cost.estimate().and_then(|cost| cost.scratch_bytes).map_or(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            Into::into,
        ),
        queue_slots: 0,
        provider_host_workspace: None,
        provider_input_stages: None,
        peak_live: None,
    }
}
