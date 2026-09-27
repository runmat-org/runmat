//! Exact source/layout/workgroup identity shared by query and execution.
use crate::backend::wgpu::{cache::key, types::NumericPrecision};

pub(super) struct PipelineRecipe {
    pub shader: String,
    pub layout: String,
    pub workgroup: u32,
}

pub(super) enum FusionPipelineLayout {
    Single,
    Multiple(usize),
}

impl PipelineRecipe {
    pub fn fusion(
        shader: &str,
        inputs: usize,
        outputs: FusionPipelineLayout,
        workgroup: u32,
    ) -> Self {
        Self {
            shader: shader.replace("@WG@", &workgroup.to_string()),
            layout: match outputs {
                FusionPipelineLayout::Single => format!("runmat-fusion-layout-{inputs}"),
                FusionPipelineLayout::Multiple(outputs) => {
                    format!("runmat-fusion-multi-layout-{inputs}-{outputs}")
                }
            },
            workgroup,
        }
    }

    pub fn floating_cast(
        source: NumericPrecision,
        target: NumericPrecision,
        workgroup: u32,
    ) -> Self {
        let scalar = |precision| match precision {
            NumericPrecision::F32 => "f32",
            NumericPrecision::F64 => "f64",
        };
        let source_type = scalar(source);
        let target_type = scalar(target);
        Self {
            shader: format!(
                "struct Params {{ len: u32, offset: u32, padding: vec2<u32> }};
                 @group(0) @binding(0) var<storage, read> input: array<{source_type}>;
                 @group(0) @binding(1) var<storage, read_write> output: array<{target_type}>;
                 @group(0) @binding(2) var<uniform> params: Params;
                 @compute @workgroup_size({workgroup})
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                     if (id.x >= params.len) {{ return; }}
                     let index = params.offset + id.x;
                     output[index] = {target_type}(input[index]);
                 }}"
            ),
            layout: "runmat-floating-cast-bgl".into(),
            workgroup,
        }
    }

    pub fn key(&self) -> u64 {
        key::compute_pipeline_hash_bytes(self.shader.as_bytes(), &self.layout, Some(self.workgroup))
    }
}

#[cfg(test)]
#[path = "pipeline_recipe/tests.rs"]
mod tests;
