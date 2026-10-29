//! Exact checked WGSL stages; no legacy reduction uniforms or shader parsing.
use super::tag::{ReductionPipelineKeys, StageTag};
use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use crate::wgsl::host_buffers::ClaimedVec;
use runmat_accelerate_api::{reduction::ProviderReductionRecipe, WgpuEmission};
use runmat_execution::resource::ProviderHostMetadataAdmission;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    Single,
    First,
    Final,
}
impl Stage {
    fn source(self, recipe: &ProviderReductionRecipe) -> &str {
        match self {
            Stage::Single => &recipe.sources().single,
            Stage::First => &recipe.sources().pass1,
            Stage::Final => &recipe.sources().pass2,
        }
    }

    fn workgroups(self, recipe: &ProviderReductionRecipe) -> [u32; 3] {
        let geometry = &recipe.descriptor().geometry;
        match self {
            Stage::Single => geometry.single.workgroups,
            Stage::First => geometry.pass1.workgroups,
            Stage::Final => geometry.pass2.workgroups,
        }
    }
}

pub(super) struct StageInvocation<'a> {
    pub(super) recipe: &'a ProviderReductionRecipe,
    pub(super) stage: Stage,
    pub(super) resources: SubmittedReductionResources,
    pub(super) host_metadata: Option<&'a dyn ProviderHostMetadataAdmission>,
}

impl WgpuProvider {
    fn typed_reduction_pipeline_key(
        &self,
        stage: Stage,
        recipe: &ProviderReductionRecipe,
    ) -> Result<u64> {
        let source = stage.source(recipe);
        let tag = StageTag::new(stage, recipe.identity())?;
        Ok(self.compute_pipeline_hash_bytes(
            source.as_bytes(),
            tag.as_str()?,
            Some(recipe.descriptor().config.workgroup_size),
        ))
    }

    pub(crate) fn typed_reduction_pipeline_keys(
        &self,
        selection: &runmat_accelerate_api::reduction::ReductionSelection<'_>,
    ) -> Option<ReductionPipelineKeys> {
        let recipe = selection.recipe();
        match selection.strategy() {
            runmat_accelerate_api::reduction::ReductionStrategy::SinglePass => {
                Some(ReductionPipelineKeys {
                    keys: [
                        self.typed_reduction_pipeline_key(Stage::Single, recipe)
                            .ok()?,
                        0,
                    ],
                    len: 1,
                })
            }
            runmat_accelerate_api::reduction::ReductionStrategy::TwoPass => {
                Some(ReductionPipelineKeys {
                    keys: [
                        self.typed_reduction_pipeline_key(Stage::First, recipe)
                            .ok()?,
                        self.typed_reduction_pipeline_key(Stage::Final, recipe)
                            .ok()?,
                    ],
                    len: 2,
                })
            }
        }
    }

    /// Every stage has precisely N read-only input bindings and one read-write
    /// output binding. Final has N=1 (the private partial), not the producer's
    /// original inputs. The cache key includes stage and immutable recipe ID.
    pub(super) fn emit_reduction_stage<O: ReductionObserver>(
        &self,
        emission: &WgpuEmission<'_>,
        pending: &mut PendingSubmissions,
        request: StageInvocation<'_>,
        submit_attempted: &mut bool,
        observer: &mut O,
    ) -> Result<u32> {
        let StageInvocation {
            recipe,
            stage,
            mut resources,
            host_metadata,
        } = request;
        // A recorded operation may already fund its canonical completion.
        // Otherwise this selected invocation owns the exact callback/payload
        // backing, independently of recorder presence or caller handle drops.
        let recorder = runmat_accelerate_api::observation::current_provider_observation();
        if recorder
            .as_ref()
            .and_then(|recorder| recorder.completion_host_metadata_claim())
            .is_none()
        {
            let completion_bytes = crate::backend::wgpu::dispatch::observation::typed_submission_dispatch_host_control_upper_bytes(recorder.is_some())
                .ok_or_else(|| anyhow::anyhow!("typed reduction completion control size overflow"))?;
            #[cfg(not(target_arch = "wasm32"))]
            let completion_bytes = completion_bytes.checked_add(
                crate::backend::wgpu::provider::backend::observation_progress::reservation_host_upper_bytes()
                    .ok_or_else(|| anyhow::anyhow!("typed reduction progress control size overflow"))?)
                .ok_or_else(|| anyhow::anyhow!("typed reduction completion control size overflow"))?;
            resources.retain_completion_claim(crate::wgsl::host_buffers::claim_bytes(
                host_metadata,
                usize::try_from(completion_bytes)?,
            )?);
        }
        let input_count = resources.input_count();
        let count = input_count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("typed reduction binding count overflow"))?;
        let source = stage.source(recipe);
        let tag = StageTag::new(stage, recipe.identity())?;
        let key = self.compute_pipeline_hash_bytes(
            source.as_bytes(),
            tag.as_str()?,
            Some(recipe.descriptor().config.workgroup_size),
        );
        let pipeline = self.get_or_create_pipeline_with(
            emission,
            host_metadata,
            key,
            "typed-reduction-pipeline",
            Some(source.as_bytes()),
            Some(tag.as_str()?),
            Some(recipe.descriptor().config.workgroup_size),
            || {
                // Warm cache hits do not construct transient layouts or compile
                // another module. Every first-party Vec backing precedes effects.
                let mut layout = ClaimedVec::with_capacity(count, host_metadata)?;
                for slot in 0..count {
                    layout.push(
                        wgpu::BindGroupLayoutEntry {
                            binding: u32::try_from(slot)?,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: slot < input_count,
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        host_metadata,
                    )?;
                }
                let bgl =
                    emission
                        .device()
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("typed-reduction-layout"),
                            entries: layout.as_slice(),
                        });
                let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
                    emission,
                    "typed-reduction-pipeline-layout",
                    &bgl,
                );
                let module = crate::backend::wgpu::shader_compilation::create_shader_module(
                    emission,
                    "typed-reduction-source",
                    source,
                    host_metadata,
                )?;
                Ok(emission.device().create_compute_pipeline(
                    &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                        label: Some("typed-reduction-pipeline"), layout: Some(&pl),
                        module: &module, entry_point: "main",
                    },
                ))
            },
        )?;
        let mut bindings = ClaimedVec::with_capacity(count, host_metadata)?;
        for slot in 0..input_count {
            bindings.push(
                wgpu::BindGroupEntry {
                    binding: u32::try_from(slot)?,
                    resource: resources.input_buffer(slot)?.as_ref().as_entire_binding(),
                },
                host_metadata,
            )?;
        }
        bindings.push(
            wgpu::BindGroupEntry {
                binding: u32::try_from(input_count)?,
                resource: resources.output_buffer().as_ref().as_entire_binding(),
            },
            host_metadata,
        )?;
        let bgl = pipeline.get_bind_group_layout(0);
        let bg = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("typed-reduction-bind"),
                layout: &bgl,
                entries: bindings.as_slice(),
            });
        // Bindings and physical owners are derived from the SAME frame. The
        // bind group has captured its WGPU refs; transfer exact lease/raw
        // ownership before encoding/submit, independently of any recorder.
        drop(bindings);
        #[cfg(test)]
        observer.prepare_pending(pending);
        pending.retain_submission_resources(observer.retaining(resources)?)?;
        let mut enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("typed-reduction-encoder"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("typed-reduction-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let [x, y, z] = stage.workgroups(recipe);
            pass.dispatch_workgroups(x, y, z);
        }
        *submit_attempted = true;
        self.submit(emission, pending, enc)
    }
}
