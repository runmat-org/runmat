pub const SHADER: &str = r#"// Image shader with direct data->viewport mapping

struct DirectUniforms {
    data_min: vec2<f32>,
    data_max: vec2<f32>,
    viewport_min: vec2<f32>,
    viewport_max: vec2<f32>,
    viewport_px: vec2<f32>,
    log_flags: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> uDirect: DirectUniforms;

@group(1) @binding(0)
var imgSampler: sampler;

@group(1) @binding(1)
var imgTex: texture_2d<f32>;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn transform_axis(value: f32, is_log: u32) -> f32 {
    if (is_log != 0u) {
        if (value <= 0.0) {
            return 1e30;
        }
        return log(value) / log(10.0);
    }
    return value;
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
    var out: VSOut;
    let data_min = vec2<f32>(
        transform_axis(uDirect.data_min.x, uDirect.log_flags.x),
        transform_axis(uDirect.data_min.y, uDirect.log_flags.y)
    );
    let data_max = vec2<f32>(
        transform_axis(uDirect.data_max.x, uDirect.log_flags.x),
        transform_axis(uDirect.data_max.y, uDirect.log_flags.y)
    );
    let position = vec2<f32>(
        transform_axis(input.position.x, uDirect.log_flags.x),
        transform_axis(input.position.y, uDirect.log_flags.y)
    );
    let data_range = data_max - data_min;
    let viewport_range = uDirect.viewport_max - uDirect.viewport_min;
    let normalized = (position - data_min) / data_range;
    let ndc = uDirect.viewport_min + normalized * viewport_range;
    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = input.uv;
    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let color = textureSample(imgTex, imgSampler, input.uv);
    return color;
}
"#;
