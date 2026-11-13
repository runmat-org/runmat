//! Opaque selected stage continuation. Only a source-bound physical candidate
//! chosen by the canonical placement owner can reach its first provider hook.

use super::super::{FusionPlacementSelection, FusionProviderCompletion};
use super::selected_metadata::{self, PreparedInput};
use super::*;
use crate::fusion::image_reduction::{ImageSourcePublication, ImageSourceTranscript};
use crate::fusion::semantics::preparation_resources::{ImageSourceGeometry, PreparationBoundError};
use crate::fusion::{FusionGroupPlan, FusionPattern};
use crate::graph::AccelGraph;
use crate::placement::{FusionPlacementObserver, LocalPlacementOutcome};
use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use runmat_runtime::context::VmFusionCompletionRight;
use std::sync::Arc;

pub struct ImageVmCheckpoint {
    pub source_output: u64,
    pub handle: GpuTensorHandle,
}

pub struct ImageVmStages {
    prepared: Box<PreparedImageStages>,
    selection: FusionPlacementSelection,
    runtime: runmat_runtime::context::RuntimeContext,
    completion: Option<VmFusionCompletionRight>,
    upload_submissions: u32,
    _source_preparation: Option<HostAllocationReceipt>,
    _stage_preparation: Option<HostAllocationReceipt>,
    _provider_metadata_preparation: Option<HostAllocationReceipt>,
    _cold_cache_preparation: Option<HostAllocationReceipt>,
    tail_control: Option<super::prepared::RegionTailAdmission>,
}

/// Source-bound inputs and invocation rights for one selected VM region.
pub struct ImageVmPreparation<'a> {
    pub graph: &'a AccelGraph,
    pub plan: &'a FusionGroupPlan,
    pub source: Arc<ImageSourceTranscript>,
    pub inputs: &'a [Value],
    pub publications: &'a [ImageSourcePublication],
    pub runtime: &'a runmat_runtime::context::RuntimeContext,
    pub placement: Option<crate::placement::PlacementCorrelationId>,
    pub decline: Option<&'a runmat_runtime::context::VmFusionDeclineRight>,
    pub completion: Option<VmFusionCompletionRight>,
    pub started: runmat_time::Instant,
}

pub enum ImageVmSelectedRoute {
    Declined,
    Provider(Box<ImageVmStages>),
    #[cfg(feature = "wgpu")]
    Original(Box<super::ImageVmOriginalRoute>),
}

/// Feature-neutral classification for a VM that executes the original source
/// span without retaining the optional original-route observation.
pub enum ImageVmSourceFallbackRoute {
    Declined,
    Provider(Box<ImageVmStages>),
    OriginalSource,
}

impl ImageVmSelectedRoute {
    pub fn into_source_fallback_route(self) -> ImageVmSourceFallbackRoute {
        match self {
            Self::Declined => ImageVmSourceFallbackRoute::Declined,
            Self::Provider(stage) => ImageVmSourceFallbackRoute::Provider(stage),
            #[cfg(feature = "wgpu")]
            Self::Original(_) => ImageVmSourceFallbackRoute::OriginalSource,
        }
    }
}

impl ImageVmStages {
    fn decline_after_probe(
        observed: bool,
        reason: &'static str,
    ) -> ProviderExecutionResult<ImageVmSelectedRoute> {
        log::debug!("selected image declined after binding: {reason}; probe_observed={observed}");
        if observed {
            Err(ProviderExecutionError::from(anyhow!(reason)).with_unproven_effects())
        } else {
            Ok(ImageVmSelectedRoute::Declined)
        }
    }

    /// A structural miss declines before effects; an operational canonical
    /// resolver or physical-owner failure retains its typed cause.
    pub async fn prepare(
        request: ImageVmPreparation<'_>,
    ) -> ProviderExecutionResult<ImageVmSelectedRoute> {
        let ImageVmPreparation {
            graph,
            plan,
            source,
            inputs,
            publications,
            runtime,
            placement,
            decline,
            completion,
            #[cfg(feature = "wgpu")]
            started,
            #[cfg(not(feature = "wgpu"))]
                started: _,
        } = request;
        let Some(FusionPattern::ImageNormalize(pattern)) = plan.group.pattern.as_ref() else {
            log::debug!("selected image declined: no image pattern");
            return Ok(ImageVmSelectedRoute::Declined);
        };
        let Some(position) = plan.inputs.iter().position(|id| *id == pattern.input) else {
            log::debug!("selected image declined: no input position");
            return Ok(ImageVmSelectedRoute::Declined);
        };
        let Some(input_value) = inputs.get(position) else {
            log::debug!("selected image declined: no input value");
            return Ok(ImageVmSelectedRoute::Declined);
        };
        let finite = runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        let provider = match input_value {
            Value::GpuTensor(input) => exact_provider_for_handle(input).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "selected image input has no registered exact provider"
                ))
            })?,
            Value::Tensor(tensor)
                if tensor.numeric_dtype() == runmat_value::NumericDType::F32
                    && tensor.as_f32_slice().is_some() =>
            {
                let selected = if finite {
                    runmat_accelerate_api::provider_without_registry_allocation()
                } else {
                    runmat_accelerate_api::provider()
                };
                let Some(provider) = selected else {
                    log::debug!("selected image declined: no host provider");
                    return Ok(ImageVmSelectedRoute::Declined);
                };
                provider
            }
            _ => {
                log::debug!("selected image declined: unsupported input carrier");
                return Ok(ImageVmSelectedRoute::Declined);
            }
        };
        if finite && provider.capability_snapshot_ref().is_none() {
            log::debug!("selected image declined: no finite capability snapshot");
            return Ok(ImageVmSelectedRoute::Declined);
        }
        let (source_preparation, stage_preparation) = if finite {
            let geometry = match ImageSourceGeometry::admit(plan, inputs, &source) {
                Ok(geometry) => geometry,
                Err(PreparationBoundError::UnsupportedDeclarationOrCarrier) => {
                    log::debug!("selected image declined: finite source geometry");
                    return Ok(ImageVmSelectedRoute::Declined);
                }
                Err(error) => return Err(ProviderExecutionError::from_source(error)),
            };
            let source_bytes = geometry
                .resolution_bytes()
                .map_err(ProviderExecutionError::from_source)?;
            let stage_bound = super::host_memory::requested_bytes(
                geometry
                    .stage_cardinality()
                    .map_err(ProviderExecutionError::from_source)?,
            )
            .map_err(ProviderExecutionError::from_source)?;
            let mut stage_bytes = stage_bound.scratch_bytes;
            if let Value::Tensor(tensor) = input_value {
                let host_bytes =
                    super::host_memory::host_source_requested_bytes(tensor.shape().len())
                        .map_err(ProviderExecutionError::from_source)?;
                stage_bytes = stage_bytes.checked_add(host_bytes).ok_or_else(|| {
                    ProviderExecutionError::from_source(PreparationBoundError::Overflow)
                })?;
            }
            let source_plan = HostAllocationPlan::new(source_bytes, 0)
                .map_err(ProviderExecutionError::from_source)?;
            let stage_plan = HostAllocationPlan::new(stage_bytes, stage_bound.retained_view_bytes)
                .map_err(ProviderExecutionError::from_source)?;
            let source_receipt = HostAllocationReceipt::admit(Some(runtime), source_plan)
                .map_err(ProviderExecutionError::from_source)?;
            let stage_receipt = HostAllocationReceipt::admit(Some(runtime), stage_plan)
                .map_err(ProviderExecutionError::from_source)?;
            if source_receipt.is_none() || stage_receipt.is_none() {
                return Err(ProviderExecutionError::from(anyhow!(
                    "finite image preparation authority changed before allocation"
                )));
            }
            (source_receipt, stage_receipt)
        } else {
            (None, None)
        };
        let invocation = match ImageSourceInvocation::bind_selected(
            source,
            graph,
            plan,
            inputs,
            ProviderPrecision::F32,
            placement,
            finite,
        )
        .await
        {
            Ok(invocation) => invocation,
            Err(
                crate::fusion::image_reduction::ImageSourceAdmissionFailure::CanonicalExecution(
                    error,
                ),
            ) => return Err(error),
            Err(reason) => {
                log::debug!("selected image declined: source binding {reason:?}");
                return Ok(ImageVmSelectedRoute::Declined);
            }
        };
        let probe_observed = invocation.observed_provider_probe();
        let Some(invocation) = invocation.bind_publications(graph, publications) else {
            return Self::decline_after_probe(
                probe_observed,
                "selected image publication join failed after input-owner proof",
            );
        };
        let mut prepared = match input_value {
            Value::GpuTensor(input) => {
                let Some(prepared) = PreparedImageStages::prepare(provider, input, &invocation)
                    .map_err(|error| {
                        if probe_observed {
                            error.with_unproven_effects()
                        } else {
                            error
                        }
                    })?
                else {
                    return Self::decline_after_probe(
                        probe_observed,
                        "selected image physical preparation declined after input-owner proof",
                    );
                };
                PreparedInput::Resident(Box::new(prepared))
            }
            Value::Tensor(tensor) => {
                let Some(host) = HostSingleImageSource::plan(provider, tensor, &invocation)
                    .map_err(|error| {
                        if probe_observed {
                            error.with_unproven_effects()
                        } else {
                            error
                        }
                    })?
                else {
                    return Self::decline_after_probe(
                        probe_observed,
                        "selected host image physical preparation declined after input-owner proof",
                    );
                };
                PreparedInput::Host(Box::new(host))
            }
            _ => return Ok(ImageVmSelectedRoute::Declined),
        };
        let provider_metadata_quote = if finite {
            let Some(quote) = selected_metadata::quote(provider, &mut prepared)? else {
                return Self::decline_after_probe(
                    probe_observed,
                    "selected image provider lacks finite registration or recipe evidence",
                );
            };
            Some(quote)
        } else {
            None
        };
        let checked_preparation = |receipt: &HostAllocationReceipt| {
            receipt.checkpoint(runtime).map_err(|error| {
                let failure = ProviderExecutionError::from_source(error);
                if probe_observed {
                    failure.with_unproven_effects()
                } else {
                    failure
                }
            })
        };
        if let Some(receipt) = &source_preparation {
            checked_preparation(receipt)?;
        }
        if let Some(receipt) = &stage_preparation {
            checked_preparation(receipt)?;
        }
        let current_finite = runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        if current_finite != finite {
            let failure = ProviderExecutionError::from(anyhow!(
                "selected image host allocation policy changed during preparation"
            ));
            return Err(if probe_observed {
                failure.with_unproven_effects()
            } else {
                failure
            });
        }
        let (selected, contract) = match &mut prepared {
            PreparedInput::Resident(stages) => {
                stages.select(input_value, runtime, decline, completion.as_ref(), finite)
            }
            PreparedInput::Host(host) => {
                host.select(input_value, runtime, decline, completion.as_ref(), finite)
            }
        }
        .map_err(|error| {
            if probe_observed {
                error.with_unproven_effects()
            } else {
                error
            }
        })?;
        if let Some(receipt) = &source_preparation {
            checked_preparation(receipt)?;
        }
        if let Some(receipt) = &stage_preparation {
            checked_preparation(receipt)?;
        }
        match selected {
            LocalPlacementOutcome::Selected {
                selected,
                provider_index: Some(0),
                feedback,
                provider_observation,
                ..
            } if selected.kind.is_provider() => {
                let physical = provider_observation
                    .then(|| match &prepared {
                        PreparedInput::Resident(stages) => stages.physical_work(),
                        PreparedInput::Host(host) => host.stage_physical_work(),
                    })
                    .flatten();
                let mut selection = FusionPlacementSelection::new_vm_image(
                    FusionPlacementObserver::new(placement),
                    (contract == crate::placement::TotalCandidateObservationContract::FullCall)
                        .then_some(feedback)
                        .flatten()
                        .map(|feedback| *feedback),
                    provider,
                    physical,
                );
                selection.require_vm_selected_stage_execution();
                // The candidate quote above affects placement, but no
                // provider-only headroom is reserved for a CPU choice. Once
                // selected, reserve it before a HostSingle upload or any
                // provider stage can emit its first effect.
                let tail_control = if completion.is_some() {
                    super::prepared::admit_region_tail(runtime, provider)?
                } else {
                    None
                };
                let mut provider_metadata_admission =
                    selected_metadata::admit_selected(provider, runtime, provider_metadata_quote)?;
                let (mut prepared, upload_submissions) = match prepared {
                    PreparedInput::Resident(prepared) => (prepared, 0),
                    PreparedInput::Host(mut host) => {
                        if runtime.is_cancelled() {
                            selection.fail_vm_image(
                                runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled,
                            );
                            return Err(ProviderExecutionError::from_source(
                                runmat_runtime::build_runtime_error(
                                    "selected image host upload cancelled before effect",
                                )
                                .with_identifier("RunMat:Placement:Cancelled")
                                .build(),
                            ));
                        }
                        let Value::Tensor(tensor) = input_value else {
                            return Err(ProviderExecutionError::from(anyhow!(
                                "selected image host source changed after placement"
                            )));
                        };
                        let semantic_identity = host.semantic_identity();
                        let resources = host.plan.physical_upper_bound().ok_or_else(|| {
                            ProviderExecutionError::from(anyhow!(
                                "selected host image physical bound changed"
                            ))
                        })?;
                        let placement_preparation = host.take_placement_preparation();
                        selection.require_vm_selected_image_upload(provider.device_id());
                        let uploaded =
                            runmat_accelerate_api::observation::with_provider_observation(
                                selection.recorder(),
                                || {
                                    (*host).upload_selected(
                                        tensor,
                                        provider_metadata_admission.permit(),
                                    )
                                },
                            );
                        if uploaded.is_err() {
                            selection.fail_vm_image(
                                runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                            );
                        }
                        let session = uploaded?.ok_or_else(|| {
                            selection.fail_vm_image(
                                runmat_accelerate_api::observation::ProviderObservationFailure::ExecutionUnproven,
                            );
                            ProviderExecutionError::from(anyhow!(
                                "selected host image upload did not bind its recipe"
                            ))
                        })?;
                        if runtime.is_cancelled() {
                            selection.fail_vm_image(
                                runmat_accelerate_api::observation::ProviderObservationFailure::Cancelled,
                            );
                            return Err(ProviderExecutionError::from_source(
                                runmat_runtime::build_runtime_error(
                                    "selected image host upload cancelled after effect",
                                )
                                .with_identifier("RunMat:Placement:Cancelled")
                                .build(),
                            )
                            .after_effect());
                        }
                        (
                            Box::new(PreparedImageStages::from_selected_host_upload(
                                session,
                                semantic_identity,
                                resources,
                                placement_preparation,
                            )?),
                            u32::from(selection.recorder().is_some()),
                        )
                    }
                };
                let checkpoint_after_selection = |receipt: &HostAllocationReceipt| {
                    receipt.checkpoint(runtime).map_err(|error| {
                        let failure = ProviderExecutionError::from_source(error);
                        if matches!(input_value, Value::Tensor(_)) {
                            failure.after_effect()
                        } else if probe_observed {
                            failure.with_unproven_effects()
                        } else {
                            failure
                        }
                    })
                };
                if let Some(receipt) = &source_preparation {
                    checkpoint_after_selection(receipt)?;
                }
                if let Some(receipt) = &stage_preparation {
                    checkpoint_after_selection(receipt)?;
                    let permit = receipt.retained_view_permit(runtime).map_err(|error| {
                        let failure = ProviderExecutionError::from_source(error);
                        if upload_submissions != 0 {
                            failure.after_effect()
                        } else {
                            failure
                        }
                    })?;
                    prepared.session_mut().set_retained_view_permit(permit);
                }
                if let Some(permit) = provider_metadata_admission.take_permit() {
                    prepared.session_mut().set_provider_metadata_permit(permit);
                }
                // HostSingle uses the selected ordered-mean plan, whose
                // session has no `cold_mu`; only an already resident source
                // can reach this separate cache preparation. Thus a finite
                // HostSingle upload cannot precede a rejecting cold-cache
                // reservation.
                let cold_cache_preparation = if finite {
                    selected_metadata::fund_cold_cache(runtime, &mut prepared, upload_submissions)?
                } else {
                    None
                };
                Ok(ImageVmSelectedRoute::Provider(Box::new(Self {
                    prepared,
                    selection,
                    runtime: runtime.clone(),
                    completion,
                    upload_submissions,
                    _source_preparation: source_preparation,
                    _stage_preparation: stage_preparation,
                    _provider_metadata_preparation: provider_metadata_admission.into_receipt(),
                    _cold_cache_preparation: cold_cache_preparation,
                    tail_control,
                })))
            }
            #[cfg(feature = "wgpu")]
            LocalPlacementOutcome::Selected {
                selected,
                provider_index: None,
                feedback,
                ..
            } if selected.kind == runmat_execution::ExecutionCandidateKind::OriginalVmRegion => {
                if finite {
                    return Err(ProviderExecutionError::from(anyhow!(
                        "finite image placement selected an unbounded original VM route"
                    )));
                }
                let Some(right) = completion else {
                    return Ok(ImageVmSelectedRoute::Declined);
                };
                let Some(route) =
                    super::ImageVmOriginalRoute::new(runtime, provider, feedback, right, started)
                else {
                    return Ok(ImageVmSelectedRoute::Declined);
                };
                Ok(ImageVmSelectedRoute::Original(Box::new(route)))
            }
            LocalPlacementOutcome::ProviderRejected { .. } => Self::decline_after_probe(
                probe_observed,
                "selected image provider was rejected after input-owner proof",
            ),
            _ => Err(ProviderExecutionError::from(anyhow!(
                "selected image placement returned a different physical route"
            ))),
        }
    }

    fn revalidate(&self) -> ProviderExecutionResult<()> {
        if self.prepared.selected_owner_is_live()? {
            Ok(())
        } else {
            Err(ProviderExecutionError::from(anyhow!(
                "selected image provider owner or device liveness changed"
            ))
            .after_effect())
        }
    }

    pub async fn mean(&mut self) -> ProviderExecutionResult<GpuTensorHandle> {
        if !self
            .prepared
            .matches_pre_effect_snapshot(self.prepared.session.provider)?
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image provider resource snapshot changed before dispatch"
            )));
        }
        super::observed::observe_stage(
            &self.runtime,
            self.selection.recorder(),
            self.prepared.session_mut().mean(),
            false,
        )
        .await
        .map(|value| value.0)
    }

    pub async fn sigma(&mut self) -> ProviderExecutionResult<GpuTensorHandle> {
        self.revalidate()
            .map_err(ProviderExecutionError::after_effect)?;
        super::observed::observe_stage(
            &self.runtime,
            self.selection.recorder(),
            self.prepared.session_mut().sigma(),
            true,
        )
        .await
        .map(|value| value.0)
    }

    pub async fn next_output_checkpoint(&mut self) -> ProviderExecutionResult<ImageVmCheckpoint> {
        self.revalidate()
            .map_err(ProviderExecutionError::after_effect)?;
        super::observed::observe_stage(
            &self.runtime,
            self.selection.recorder(),
            self.prepared.session_mut().next_output_checkpoint(),
            true,
        )
        .await
        .map(|checkpoint| ImageVmCheckpoint {
            source_output: checkpoint.source_output,
            handle: checkpoint.handle,
        })
    }

    pub fn acknowledge_output_checkpoint(
        &mut self,
        source_output: u64,
    ) -> ProviderExecutionResult<()> {
        self.prepared
            .session_mut()
            .acknowledge_output_checkpoint(source_output)
    }

    /// Called only after every selected source checkpoint has been published.
    pub async fn into_vm_completion_after_tail(
        self: Box<Self>,
    ) -> Option<FusionProviderCompletion> {
        if !matches!(self.prepared.session.state, State::Complete)
            || !self.prepared.selected_owner_is_live().ok()?
        {
            return None;
        }
        if let Some(work) = self.prepared.physical_work() {
            self.selection.confirm_vm_image_submissions(
                work.queue_submissions
                    .checked_add(self.upload_submissions)?,
            );
        }
        let completion = self.selection.into_vm_completion(self.completion?)?;
        let tail_control = self.tail_control?;
        let provider = self.prepared.session.provider;
        if self.runtime.is_cancelled()
            || provider
                .observe_region_queue_tail(provider.device_id(), tail_control.claim)
                .await
                .is_err()
            || self.runtime.is_cancelled()
        {
            return None;
        }
        Some(completion)
    }
}
