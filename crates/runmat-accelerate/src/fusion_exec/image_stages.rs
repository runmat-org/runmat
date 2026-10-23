//! Resident image stages selected from an exact source transcript. The VM
//! joins ordered stores; the physical owner admits the move-owned recipes.
//! Whole-program cost remains unpriced until genuine observation exists.
use crate::fusion::image_reduction::{
    ImageMeanAndPowerSource, ImageSelectedBinaryCalls, ImageSelectedMomentBindings,
    ImageSelectedRealCalls, ImageSelectedSigmaOperations, ImageSourceInvocation,
};
use anyhow::anyhow;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ImageNormalizeDescriptor, NumericElementType,
    ProviderColdNdMean, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionResult, ProviderImageMeanCall, ProviderImageMeanClass, ProviderImageMeanHook,
    ProviderImageOutputProgram, ProviderImageStageWorkload, ProviderPinnedNdMean,
    ProviderPrecision,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};
use runmat_value::Value;

/// Request-local source arithmetic authority. The canonical resolver has
/// bound every external leaf, the output programs retain the selected
/// operator/control/binding identities, and coverage enumerates every graph
/// operation in source order. Physical feasibility remains a separate gate.
#[derive(Clone, Copy)]
pub(crate) enum ImageArithmeticProof {
    Selected {
        semantic_identity: runmat_execution::Digest,
    },
    #[cfg(test)]
    Fixture,
}

impl ImageArithmeticProof {
    pub(crate) fn for_selected(invocation: &ImageSourceInvocation) -> Option<Self> {
        (invocation.selected_publications_bound() && invocation.selected_stage_covers_source())
            .then_some(Self::Selected {
                semantic_identity: invocation.semantic_identity(),
            })
    }
}

#[cfg(test)]
impl ImageArithmeticProof {
    fn fixture() -> Self {
        Self::Fixture
    }
}

#[derive(Debug)]
pub(crate) struct ImageMean(pub GpuTensorHandle);
#[derive(Debug)]
pub(crate) struct ImageSigma(pub GpuTensorHandle);
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct ImageResult(pub GpuTensorHandle);

#[derive(Debug)]
pub(crate) struct ImageOutputCheckpoint {
    pub source_output: u64,
    pub handle: GpuTensorHandle,
}

struct OutputProgress {
    mean: GpuTensorHandle,
    sigma: GpuTensorHandle,
    values: Vec<Option<GpuTensorHandle>>,
    positions: Vec<(u64, usize)>,
    next_call: usize,
}

struct OutputProgressLayout {
    values: Vec<Option<GpuTensorHandle>>,
    positions: Vec<(u64, usize)>,
}

impl OutputProgress {
    fn layout(
        calls: &[runmat_accelerate_api::ProviderImageOutputCall],
    ) -> ProviderExecutionResult<OutputProgressLayout> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(calls.len())
            .map_err(|error| ProviderExecutionError::from(anyhow!(error)).after_effect())?;
        values.resize_with(calls.len(), || None);
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(calls.len())
            .map_err(|error| ProviderExecutionError::from(anyhow!(error)).after_effect())?;
        positions.extend(
            calls
                .iter()
                .enumerate()
                .map(|(index, call)| (call.source_output, index)),
        );
        positions.sort_unstable_by_key(|(id, _)| *id);
        if positions.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image output IDs are not unique"
            ))
            .after_effect());
        }
        Ok(OutputProgressLayout { values, positions })
    }

    fn new(mean: GpuTensorHandle, sigma: GpuTensorHandle, layout: OutputProgressLayout) -> Self {
        let OutputProgressLayout { values, positions } = layout;
        Self {
            mean,
            sigma,
            values,
            positions,
            next_call: 0,
        }
    }

    fn get(&self, id: u64) -> Option<&GpuTensorHandle> {
        let index = self
            .positions
            .binary_search_by_key(&id, |(key, _)| *key)
            .ok()?;
        self.values.get(self.positions[index].1)?.as_ref()
    }

    fn retained_values(&self) -> impl Iterator<Item = &GpuTensorHandle> {
        self.values.iter().filter_map(Option::as_ref)
    }

    fn into_values(self) -> impl Iterator<Item = GpuTensorHandle> {
        self.values.into_iter().flatten()
    }
}

enum State {
    BeforeMean,
    /// Installed before the first await so dropping a stage future cannot
    /// leave a replayable checkpoint after an unknown provider submission.
    InFlight,
    MeanPublished(GpuTensorHandle),
    SigmaPublished {
        mean: GpuTensorHandle,
        sigma: GpuTensorHandle,
    },
    OutputReady(OutputProgress),
    OutputAwaitAck {
        progress: OutputProgress,
        source_output: u64,
    },
    Complete,
    Failed,
}

mod host_memory;
mod host_source;
mod moment_publications;
mod nd_resources;
mod observed;
mod ordered_mean;
#[cfg(feature = "wgpu")]
mod original_route;
mod output_hooks;
mod placement;
mod power;
mod prepared;
mod resources;
mod selected_binary;
mod selected_metadata;
mod selected_plan;
mod selected_real_output;
mod sigma_operations;
mod stages;
mod vm;
pub(crate) use host_source::HostSingleImageSource;
#[cfg(feature = "wgpu")]
pub use original_route::ImageVmOriginalRoute;
pub(crate) use prepared::PreparedImageStages;
pub(crate) use prepared::{admit_region_audit, admit_region_tail, RegionTailAdmission};
pub use vm::{
    ImageVmCheckpoint, ImageVmPreparation, ImageVmSelectedRoute, ImageVmSourceFallbackRoute,
    ImageVmStages,
};

/// Owns only its own references. A VM-published moment is a separate retained
/// handle and remains live if this session is abandoned at a later checkpoint.
pub(crate) struct ImageNormalizeStageSession {
    provider: &'static dyn AccelProvider,
    input: GpuTensorHandle,
    desc: ImageNormalizeDescriptor,
    program: ProviderImageStageWorkload,
    selected_power_plans: Option<power::ImagePowerPlans>,
    selected_mean_plans: Option<ordered_mean::ImageOrderedMeanPlans>,
    selected_binary_plans: Option<selected_binary::SelectedBinaryPlans>,
    selected_real_plans: Option<selected_real_output::SelectedRealPlans>,
    reserved_mu: Option<ProviderPinnedNdMean>,
    cold_mu: Option<ProviderColdNdMean>,
    retained_view_permit: Option<runmat_runtime::context::host_allocation::HostRetainedViewPermit>,
    provider_metadata_permit:
        Option<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>>,
    state: State,
}

// A selected program and its sealed source binding enter the session together.
struct ImageStageBeginRequest {
    source: ImageMeanAndPowerSource,
    output_program: Option<ProviderImageOutputProgram>,
    frozen: Option<ProviderImageStageWorkload>,
    bindings: Option<ImageSelectedMomentBindings>,
    sigma_operations: Option<ImageSelectedSigmaOperations>,
    binary_calls: Option<ImageSelectedBinaryCalls>,
    real_calls: Option<ImageSelectedRealCalls>,
}

impl ImageNormalizeStageSession {
    fn cold_cache_view_rank(&self) -> Option<usize> {
        self.cold_mu.as_ref().map(|cold| cold.output_shape().len())
    }

    fn cold_cache_metadata_payload_bytes(&self) -> Option<u64> {
        self.cold_mu
            .as_ref()
            .map(|cold| cold.requested_cache_metadata_payload_bytes())
    }

    fn fund_cold_cache_views(
        &mut self,
        permit: &runmat_runtime::context::host_allocation::HostRetainedViewPermit,
    ) -> Result<(), runmat_runtime::context::host_allocation::HostAllocationError> {
        let Some(cold) = self.cold_mu.take() else {
            return Ok(());
        };
        let Some(bytes) = GpuTensorHandle::view_host_upper_bytes(cold.output_shape().len()) else {
            self.cold_mu = Some(cold);
            return Err(runmat_runtime::context::host_allocation::HostAllocationError::Overflow);
        };
        let first = match permit.claim_view(bytes) {
            Ok(grant) => grant,
            Err(error) => {
                self.cold_mu = Some(cold);
                return Err(error);
            }
        };
        let second = match permit.claim_view(bytes) {
            Ok(grant) => grant,
            Err(error) => {
                self.cold_mu = Some(cold);
                return Err(error);
            }
        };
        let payload = cold.requested_cache_metadata_payload_bytes();
        let metadata = match permit.claim_metadata(payload) {
            Ok(claim) => claim,
            Err(error) => {
                self.cold_mu = Some(cold);
                return Err(error);
            }
        };
        self.cold_mu = Some(
            cold.with_cache_view_grants([first, second])
                .with_cache_metadata_claim(metadata),
        );
        Ok(())
    }
    fn set_retained_view_permit(
        &mut self,
        permit: runmat_runtime::context::host_allocation::HostRetainedViewPermit,
    ) {
        self.retained_view_permit = Some(permit);
    }

    fn set_provider_metadata_permit(
        &mut self,
        permit: Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) {
        self.provider_metadata_permit = Some(permit);
    }
    /// Physical requests from every selected provider recipe. No estimate is
    /// returned for a legacy hook or a missing proposal. Existing input and
    /// provider cache allocations are not falsely charged as fresh outputs.
    fn selected_physical_upper_bound(&self) -> Option<resources::ImageStagePhysicalUpperBound> {
        let mut bound = resources::ImageStagePhysicalUpperBound::default();
        bound.combine(self.selected_mean_plans.as_ref()?.physical_upper_bound()?)?;
        bound.combine(
            self.selected_binary_plans
                .as_ref()?
                .physical_upper_bound()?,
        )?;
        bound.combine(self.selected_real_plans.as_ref()?.physical_upper_bound()?)?;
        bound.combine(self.selected_power_plans.as_ref()?.physical_upper_bound()?)?;
        Some(bound)
    }

    #[cfg(test)]
    pub(crate) fn begin(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        desc: ImageNormalizeDescriptor,
        source: ImageMeanAndPowerSource,
        proof: ImageArithmeticProof,
    ) -> ProviderExecutionResult<Option<Self>> {
        Self::begin_with_output_program(
            provider,
            input,
            desc,
            ImageStageBeginRequest {
                source,
                output_program: None,
                frozen: None,
                bindings: None,
                sigma_operations: None,
                binary_calls: None,
                real_calls: None,
            },
            proof,
        )
    }

    /// A selected source-bound output request enters only with its matching
    /// arithmetic proof. The physical candidate is admitted separately.
    pub(crate) fn begin_selected(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        invocation: &ImageSourceInvocation,
        proof: ImageArithmeticProof,
    ) -> ProviderExecutionResult<Option<Self>> {
        let authorized = match proof {
            ImageArithmeticProof::Selected { semantic_identity } => {
                semantic_identity == invocation.semantic_identity()
            }
            #[cfg(test)]
            ImageArithmeticProof::Fixture => true,
        };
        if !authorized {
            return Ok(None);
        }
        if !invocation.selected_stage_covers_source() {
            return Ok(None);
        }
        let Some((desc, source)) = invocation.selected_staged_mean_power(input) else {
            return Ok(None);
        };
        let Some(output_program) = invocation.requested_output_program().cloned() else {
            return Ok(None);
        };
        let Some(frozen) = invocation.requested_staged_workload().cloned() else {
            return Ok(None);
        };
        if frozen.selected_powers.is_none() {
            return Ok(None);
        }
        let Some(bindings) = invocation.selected_moment_bindings() else {
            return Ok(None);
        };
        let Some(sigma_operations) = invocation.selected_sigma_operations() else {
            return Ok(None);
        };
        let Some(binary_calls) = invocation.selected_binary_calls() else {
            return Ok(None);
        };
        if binary_calls.semantic_identity() != invocation.semantic_identity() {
            return Ok(None);
        }
        let Some(real_calls) = invocation.selected_real_output_calls() else {
            return Ok(None);
        };
        if real_calls.semantic_identity() != invocation.semantic_identity() {
            return Ok(None);
        }
        Self::begin_with_output_program(
            provider,
            input,
            desc.clone(),
            ImageStageBeginRequest {
                source: source.clone(),
                output_program: Some(output_program),
                frozen: Some(frozen),
                bindings: Some(bindings),
                sigma_operations: Some(sigma_operations),
                binary_calls: Some(binary_calls),
                real_calls: Some(real_calls),
            },
            proof,
        )
    }

    #[cfg(test)]
    fn begin_selected_fixture(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        desc: ImageNormalizeDescriptor,
        source: ImageMeanAndPowerSource,
        output_program: ProviderImageOutputProgram,
        proof: ImageArithmeticProof,
    ) -> ProviderExecutionResult<Option<Self>> {
        Self::begin_with_output_program(
            provider,
            input,
            desc,
            ImageStageBeginRequest {
                source,
                output_program: Some(output_program),
                frozen: None,
                bindings: None,
                sigma_operations: None,
                binary_calls: None,
                real_calls: None,
            },
            proof,
        )
    }

    fn begin_with_output_program(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        desc: ImageNormalizeDescriptor,
        request: ImageStageBeginRequest,
        _proof: ImageArithmeticProof,
    ) -> ProviderExecutionResult<Option<Self>> {
        let ImageStageBeginRequest {
            source,
            output_program,
            frozen,
            bindings,
            sigma_operations,
            binary_calls,
            real_calls,
        } = request;
        input
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let valid_owner = input.device_id() == provider.device_id()
            && exact_provider_for_handle(input).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            });
        if !valid_owner {
            return Err(anyhow!("image normalize stage input has no live selected owner").into());
        }
        let shape = input.shape();
        let selected_powers = frozen
            .as_ref()
            .and_then(|workload| workload.selected_powers.clone());
        let supported = input.physical().element_type() == NumericElementType::F32
            && input.physical().storage() == GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_logical(input)
            && !runmat_accelerate_api::handle_is_transposed(input)
            && gpu_class_metadata_matches(input, Some(ProviderPrecision::F32), None, false)
            && shape == [desc.batch, desc.height, desc.width]
            && shape.len() == 3
            && shape.iter().all(|dimension| *dimension > 0)
            && shape
                .iter()
                .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
                .is_some()
            && desc.epsilon.is_finite()
            && desc.epsilon > 0.0
            && (selected_powers.is_some() || (desc.epsilon as f32) as f64 == desc.epsilon)
            && (output_program.is_some()
                || (desc.gain.is_none() && desc.bias.is_none() && !desc.clamp_zero));
        let Some(mut program) = (if let Some(powers) = selected_powers {
            let Some(selected_epsilon) = frozen.as_ref().map(|workload| workload.epsilon) else {
                return Ok(None);
            };
            source.staged_workload_selected(&desc, powers, selected_epsilon)
        } else {
            source.staged_workload(&desc)
        }) else {
            return Ok(None);
        };
        if let Some(output_program) = output_program {
            if !output_program.validate()
                || output_program.calls.iter().any(|call| {
                    call.output_shape != program.input_shape
                        || (program.selected_powers.is_none()
                            && call.output_precision != ProviderPrecision::F32)
                        || (call.hook == runmat_accelerate_api::ProviderImageOutputHook::ElemPow
                            && program.selected_powers.is_none()
                            && !matches!(
                            call.operands.get(1),
                            Some(
                                runmat_accelerate_api::ProviderImageOutputOperand::ScalarF32Bits(_)
                            )
                        ))
                })
            {
                return Ok(None);
            }
            program.output_program = Some(output_program);
        }
        if !supported {
            return Ok(None);
        }
        if let Some(frozen) = frozen {
            let mut expected = program;
            expected.moment_program = frozen.moment_program.clone();
            if expected != frozen || !frozen.validate() || frozen.moment_program.is_none() {
                return Ok(None);
            }
            program = frozen;
        }
        if program.selected_powers.is_some() && program.output_program.is_none() {
            return Ok(None);
        }
        let selected_power_plans = power::plan_selected(provider, &program)?;
        if program.selected_powers.is_some() && selected_power_plans.is_none() {
            return Ok(None);
        }
        let mut selected_mean_plans = if let Some(powers) = selected_power_plans.as_ref() {
            ordered_mean::plan_selected(provider, input, &program, powers, bindings.as_ref())?
        } else {
            None
        };
        if program.selected_powers.is_some() && selected_mean_plans.is_none() {
            return Ok(None);
        }
        if program.selected_powers.is_some() {
            let Some((publication, source, bindings, moments)) = selected_mean_plans
                .as_mut()
                .and_then(|plans| plans.sigma.as_mut())
                .zip(sigma_operations.as_ref())
                .zip(bindings.as_ref())
                .zip(program.moment_program.as_ref())
                .map(|(((publication, source), bindings), moments)| {
                    (publication, source, bindings, moments)
                })
            else {
                return Ok(None);
            };
            let Some(planned) =
                sigma_operations::plan(provider, source, publication, moments, bindings)?
            else {
                return Ok(None);
            };
            if !planned.matches_program(&program) {
                return Ok(None);
            }
            publication.operations = Some(planned);
        }
        let selected_binary_plans = match (
            program.selected_powers.as_ref(),
            selected_power_plans.as_ref(),
            selected_mean_plans.as_ref(),
            binary_calls.as_ref(),
        ) {
            (Some(_), Some(powers), Some(means), Some(calls)) => {
                selected_binary::plan(provider, input, calls, powers, means, &program)?
            }
            (None, _, _, None) => None,
            _ => return Ok(None),
        };
        if program.selected_powers.is_some() && selected_binary_plans.is_none() {
            return Ok(None);
        }
        let selected_real_plans = match (
            program.selected_powers.as_ref(),
            selected_binary_plans.as_ref(),
            real_calls.as_ref(),
            selected_power_plans.as_ref(),
        ) {
            (Some(_), Some(binary), Some(calls), Some(powers)) => {
                selected_real_output::plan(provider, calls, binary, powers, &program)?
            }
            (None, _, None, None) => None,
            _ => return Ok(None),
        };
        if program.selected_powers.is_some() && selected_real_plans.is_none() {
            return Ok(None);
        }
        // Only the exact one-hook N-D source form may consume this cache
        // reservation. Nested dimension means have distinct arithmetic and
        // their ordinary rank-three WGPU route is not qualified here.
        let (reserved_mu, cold_mu) = match (
            selected_mean_plans.as_ref(),
            self::first_nd_hook(&program.mu),
        ) {
            (Some(_), _) => (None, None),
            (None, Some(dims)) => {
                let warm = provider
                    .reserve_cached_mean_nd(input, &dims)
                    .map_err(ProviderExecutionError::from)?;
                let cold = if warm.is_none() {
                    provider
                        .prepare_cold_mean_nd(input, &dims)
                        .map_err(ProviderExecutionError::from)?
                } else {
                    None
                };
                (warm, cold)
            }
            (None, None) => (None, None),
        };
        let expected_mean = [desc.batch, 1, 1];
        if let Some(warm) = reserved_mu.as_ref() {
            let dims = first_nd_hook(&program.mu)
                .ok_or_else(|| anyhow!("selected warm mean has no source N-D hook"))?;
            nd_resources::warm_mean(provider, input, warm, &dims, &expected_mean)?;
        }
        if let Some(cold) = cold_mu.as_ref() {
            let dims = first_nd_hook(&program.mu)
                .ok_or_else(|| anyhow!("selected cold mean has no source N-D hook"))?;
            nd_resources::cold_mean(provider, input, cold, &dims, &expected_mean)?;
        }
        crate::ensure_residency_hooks();
        Ok(Some(Self {
            provider,
            input: input.clone(),
            desc,
            program,
            selected_power_plans,
            selected_mean_plans,
            selected_binary_plans,
            selected_real_plans,
            reserved_mu,
            cold_mu,
            retained_view_permit: None,
            provider_metadata_permit: None,
            state: State::BeforeMean,
        }))
    }

    fn moment_shape(&self) -> [usize; 3] {
        [self.desc.batch, 1, 1]
    }

    fn admit_next(
        &self,
        output: GpuTensorHandle,
        expected: &[usize],
        scratch: &[GpuTensorHandle],
        published: &[&GpuTensorHandle],
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let protected = scratch
            .iter()
            .chain(published.iter().copied())
            .collect::<Vec<_>>();
        super::image_output::admit_stage(output, self.provider, &self.input, expected, &protected)
    }

    async fn reduce_source_mean(
        &self,
        input: &GpuTensorHandle,
        calls: &[ProviderImageMeanCall],
        scratch: &mut Vec<GpuTensorHandle>,
        published: &[&GpuTensorHandle],
        first_stage: bool,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let publications = self.program.moment_program.as_ref().map(|program| {
            if first_stage {
                program.mu.as_slice()
            } else {
                program.variance.as_slice()
            }
        });
        let ordered = self.selected_mean_plans.as_ref().map(|plans| {
            if first_stage {
                plans.mu.as_slice()
            } else {
                plans.variance.as_slice()
            }
        });
        if self.program.selected_powers.is_some() && ordered.is_none() {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected ordered mean plan is missing"
            ))
            .after_effect());
        }
        let mut current = input.clone();
        for (index, call) in calls.iter().enumerate() {
            let expected_class = match call.output_class {
                ProviderImageMeanClass::Default => {
                    runmat_accelerate_api::handle_precision(&current)
                }
                ProviderImageMeanClass::Native => runmat_accelerate_api::handle_precision(&current),
            };
            let next = if let Some(ordered) = ordered {
                let planned = ordered.get(index).ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("selected mean call plan is missing"))
                        .after_effect()
                })?;
                let publication = publications
                    .and_then(|publications| publications.get(index))
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!(
                            "selected mean source publication is missing"
                        ))
                        .after_effect()
                    })?;
                ordered_mean::execute_call_admitted(
                    self.provider,
                    ordered_mean::OrderedMeanExecutionContext {
                        current: &current,
                        image_input: &self.input,
                        scratch,
                        published,
                        prior_effect: !first_stage || index > 0,
                    },
                    call,
                    publication,
                    planned,
                    self.provider_metadata_permit.as_ref().map(|permit| {
                        permit.as_ref()
                            as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
                    }),
                )?
            } else {
                match call.hook {
                    ProviderImageMeanHook::ReduceNd(axes) => {
                        let axes = axes.map(usize::from);
                        self.provider.reduce_mean_nd(&current, &axes).await
                    }
                    ProviderImageMeanHook::ReduceDim(axis) => {
                        self.provider
                            .reduce_mean_dim(&current, usize::from(axis))
                            .await
                    }
                }
                .map_err(|source| {
                    let failure = source.with_unproven_effects();
                    if first_stage && index == 0 {
                        failure
                    } else {
                        failure.after_effect()
                    }
                })?
                .ok_or_else(|| {
                    let failure = ProviderExecutionError::from(
                        runmat_accelerate_api::unsupported_provider_operation(
                            "selected image mean stage unavailable",
                        ),
                    );
                    if first_stage && index == 0 {
                        failure
                    } else {
                        failure.after_effect()
                    }
                })?
            };
            let expected = call
                .shape
                .map(|n| usize::try_from(n).expect("validated source shape"));
            let admitted = self.admit_reduction(
                next,
                &current,
                &expected,
                expected_class,
                scratch,
                published,
            )?;
            if runmat_accelerate_api::handle_precision(&admitted) != expected_class {
                return Err(super::outputs::reject_outputs(
                    vec![admitted],
                    "image mean provider returned the wrong selected output class",
                ));
            }
            let admitted = if let Some(publications) = publications {
                let publication = publications.get(index).ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("selected mean publication is absent"))
                        .after_effect()
                })?;
                let selected_casts = ordered
                    .and_then(|plans| plans.get(index))
                    .map(|plan| plan.casts.as_slice());
                self.publish_moment(admitted, publication, scratch, published, selected_casts)
                    .await?
            } else {
                admitted
            };
            if index + 1 == calls.len() {
                return Ok(admitted);
            }
            current = admitted.clone();
            scratch.push(admitted);
        }
        Err(anyhow!("image mean source has no direct provider hook").into())
    }

    fn finish(
        result: ProviderExecutionResult<GpuTensorHandle>,
        scratch: Vec<GpuTensorHandle>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let mut cleanup = scratch
            .into_iter()
            .filter_map(|handle| handle.release().err())
            .collect::<Vec<_>>();
        if cleanup.is_empty() {
            result
        } else {
            let primary = match result {
                Ok(unpublished) => {
                    cleanup.extend(unpublished.release().err());
                    None
                }
                Err(error) => Some(error),
            };
            Err(
                ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                    primary, cleanup,
                ))
                .after_effect(),
            )
        }
    }

    fn provenance(&self, mut handle: GpuTensorHandle) -> ProviderExecutionResult<GpuTensorHandle> {
        let stamped = (|| -> ProviderExecutionResult<GpuTensorHandle> {
            if let Some(permit) = &self.retained_view_permit {
                let bytes = GpuTensorHandle::view_host_upper_bytes(handle.shape().len())
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!("image view extent overflows"))
                            .after_effect()
                    })?;
                let claim = permit
                    .claim_view(bytes)
                    .map_err(|error| ProviderExecutionError::from_source(error).after_effect())?;
                let provenance = if runmat_accelerate_api::handle_is_explicit(&self.input) {
                    runmat_accelerate_api::GpuHandleProvenance::Explicit
                } else {
                    runmat_accelerate_api::GpuHandleProvenance::Automatic
                };
                handle
                    .with_view_retained_grant(
                        handle.view().clone().with_provenance(provenance),
                        claim,
                    )
                    .map_err(|error| ProviderExecutionError::from_source(error).after_effect())
            } else {
                super::mark_fusion_output(&mut handle, &[Value::GpuTensor(self.input.clone())])
                    .map(|()| handle.clone())
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())
            }
        })();
        let published = match stamped {
            Ok(published) => published,
            Err(primary) => {
                return match handle.release() {
                    Ok(_) => Err(primary),
                    Err(cleanup) => Err(ProviderExecutionError::from_source(
                        ProviderExecutionCompletionError::new(Some(primary), vec![cleanup]),
                    )
                    .after_effect()),
                }
            }
        };
        crate::fusion_residency::mark(&published);
        Ok(published)
    }
}

fn first_nd_hook(calls: &[ProviderImageMeanCall]) -> Option<[usize; 2]> {
    match calls {
        [ProviderImageMeanCall {
            hook: ProviderImageMeanHook::ReduceNd(axes),
            ..
        }] => Some(axes.map(usize::from)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "image_stages/numerical_tests.rs"]
mod numerical_tests;
#[cfg(test)]
mod tests;
