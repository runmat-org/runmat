//! Exact source/layout/workgroup identity shared by query and execution.
use crate::backend::wgpu::{cache::key, types::NumericPrecision};

const DIMENSION_N_SUM_F64_SHADER: &str = r#"
struct Doubles { data: array<f64> };
struct Params { len: u32, offset: u32, stride_before: u32, reduce_len: u32 };
@group(0) @binding(0) var<storage, read> input: Doubles;
@group(0) @binding(1) var<storage, read_write> output: Doubles;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.len) { return; }
    let index = params.offset + id.x;
    let before = params.stride_before;
    let base = (index % before) + (index / before) * before * params.reduce_len;
    var sum = f64(0.0);
    for (var k = 0u; k < params.reduce_len; k = k + 1u) {
        sum = sum + input.data[base + k * before];
    }
    output.data[index] = sum;
}
"#;

pub(super) const INTEGER_SUM_CAST_LAYOUT_TAG: &str = "runmat-integer-sum-floating-cast-bgl";
pub(super) const DIMENSION_N_SUM_F64_LAYOUT_TAG: &str = "runmat-nd-sum-f64-bgl";
pub(super) const INTEGER_CAST_LAYOUT_TAG: &str = "runmat-integer-cast-bgl";
pub(super) const FLOATING_CAST_LAYOUT_TAG: &str = "runmat-floating-cast-bgl";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PipelineRecipe {
    pub shader: String,
    pub layout: String,
    pub workgroup: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            layout: FLOATING_CAST_LAYOUT_TAG.into(),
            workgroup,
        }
    }

    pub fn integer_cast(source: NumericPrecision, workgroup: u32) -> Self {
        Self {
            shader: crate::backend::wgpu::shaders::integer::cast_shader(source, workgroup),
            layout: INTEGER_CAST_LAYOUT_TAG.into(),
            workgroup,
        }
    }

    pub fn integer_sum_cast(workgroup: u32) -> Self {
        Self {
            shader: crate::backend::wgpu::shaders::integer::floating_sum_cast_shader(workgroup),
            layout: INTEGER_SUM_CAST_LAYOUT_TAG.into(),
            workgroup,
        }
    }

    pub fn dimension_n_sum_f64(workgroup: u32) -> Self {
        Self {
            shader: DIMENSION_N_SUM_F64_SHADER.replace("@WG@", &workgroup.to_string()),
            layout: DIMENSION_N_SUM_F64_LAYOUT_TAG.into(),
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
