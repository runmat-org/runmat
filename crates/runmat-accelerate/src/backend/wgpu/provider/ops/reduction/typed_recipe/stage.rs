//! Exact checked WGSL stages; no legacy reduction uniforms or shader parsing.
use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use runmat_accelerate_api::{reduction::ProviderReductionRecipe, WgpuBufferOwner, WgpuEmission};

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
    pub(super) inputs: &'a [WgpuBufferOwner],
    pub(super) output: &'a WgpuBufferOwner,
}

impl WgpuProvider {
    fn typed_reduction_pipeline_key(&self, stage: Stage, recipe: &ProviderReductionRecipe) -> u64 {
        let source = stage.source(recipe);
        let tag = format!("typed-reduction-{stage:?}-{:?}", recipe.identity());
        self.compute_pipeline_hash_bytes(
            source.as_bytes(),
            &tag,
            Some(recipe.descriptor().config.workgroup_size),
        )
    }

    pub(crate) fn typed_reduction_pipeline_keys(
        &self,
        selection: &runmat_accelerate_api::reduction::ReductionSelection<'_>,
    ) -> Vec<u64> {
        let recipe = selection.recipe();
        match selection.strategy() {
            runmat_accelerate_api::reduction::ReductionStrategy::SinglePass => {
                vec![self.typed_reduction_pipeline_key(Stage::Single, recipe)]
            }
            runmat_accelerate_api::reduction::ReductionStrategy::TwoPass => vec![
                self.typed_reduction_pipeline_key(Stage::First, recipe),
                self.typed_reduction_pipeline_key(Stage::Final, recipe),
            ],
        }
    }

    /// Every stage has precisely N read-only input bindings and one read-write
    /// output binding. Final has N=1 (the private partial), not the producer's
    /// original inputs. The cache key includes stage and immutable recipe ID.
    pub(super) fn emit_reduction_stage(
        &self,
        emission: &WgpuEmission<'_>,
        pending: &mut PendingSubmissions,
        request: StageInvocation<'_>,
        submit_attempted: &mut bool,
    ) -> Result<u32> {
        let StageInvocation {
            recipe,
            stage,
            inputs,
            output,
        } = request;
        let mut layout = Vec::with_capacity(inputs.len() + 1);
        let mut bindings = Vec::with_capacity(inputs.len() + 1);
        for (slot, buffer) in inputs.iter().enumerate() {
            layout.push(wgpu::BindGroupLayoutEntry {
                binding: slot as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            bindings.push(wgpu::BindGroupEntry {
                binding: slot as u32,
                resource: buffer.as_ref().as_entire_binding(),
            });
        }
        let output_slot = inputs.len() as u32;
        layout.push(wgpu::BindGroupLayoutEntry {
            binding: output_slot,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        bindings.push(wgpu::BindGroupEntry {
            binding: output_slot,
            resource: output.as_ref().as_entire_binding(),
        });
        let bgl = emission
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("typed-reduction-layout"),
                entries: &layout,
            });
        let pl = crate::backend::wgpu::pipelines::create_pipeline_layout(
            emission,
            "typed-reduction-pipeline-layout",
            &bgl,
        );
        let source = stage.source(recipe);
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            emission,
            "typed-reduction-source",
            source,
        );
        let tag = format!("typed-reduction-{stage:?}-{:?}", recipe.identity());
        let key = self.typed_reduction_pipeline_key(stage, recipe);
        let pipeline = self.get_or_create_pipeline(
            emission,
            key,
            &pl,
            &module,
            "typed-reduction-pipeline",
            Some(source.as_bytes()),
            Some(&tag),
            Some(recipe.descriptor().config.workgroup_size),
        )?;
        let bg = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("typed-reduction-bind"),
                layout: &bgl,
                entries: &bindings,
            });
        let mut enc = emission
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("typed-reduction-encoder"),
            });
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
