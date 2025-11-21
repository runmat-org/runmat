pub const STOCHASTIC_EVOLUTION_SHADER_F32: &str = r#"
struct Tensor {
    data: array<f32>,
};

struct StochasticEvolutionParams {
    offset: u32,
    chunk: u32,
    len: u32,
    steps: u32,
    key0: u32,
    key1: u32,
    _pad0: u32,
    _pad1: u32,
    drift: f32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: StochasticEvolutionParams;

const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const INV_U32: f32 = 1.0 / 4294967296.0;
const TWO_PI: f32 = 6.2831855;
const MIN_UNIFORM: f32 = 1.0e-8;
const ALMOST_ONE: f32 = 0.99999994;

fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a_hi = a >> 16u;
    let a_lo = a & 0xFFFFu;
    let b_hi = b >> 16u;
    let b_lo = b & 0xFFFFu;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
}

fn philox_round(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let hi0 = mul_hi_u32(PHILOX_M0, counter.x);
    let lo0 = PHILOX_M0 * counter.x;
    let hi1 = mul_hi_u32(PHILOX_M1, counter.z);
    let lo1 = PHILOX_M1 * counter.z;
    return vec4<u32>(
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0,
    );
}

fn philox(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        ctr = philox_round(ctr, k);
        k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    }
    return ctr;
}

fn uniform_from_bits(bits: u32) -> f32 {
    let sample = (f32(bits) + 0.5) * INV_U32;
    return clamp(sample, MIN_UNIFORM, ALMOST_ONE);
}

fn normal_from_bits(bits0: u32, bits1: u32) -> f32 {
    let u1 = uniform_from_bits(bits0);
    let u2 = uniform_from_bits(bits1);
    let radius = sqrt(-2.0 * log(u1));
    let angle = TWO_PI * u2;
    return radius * cos(angle);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    if global_idx >= params.len {
        return;
    }

    var ctr = vec4<u32>(global_idx, 0u, 0u, 0u);
    let key = vec2<u32>(params.key0, params.key1);
    var steps: u32 = params.steps;
    var current = Input.data[global_idx];

    var iter: u32 = 0u;
    loop {
        if iter >= steps {
            break;
        }
        ctr.y = iter;
        let rnd = philox(ctr, key);
        let noise = normal_from_bits(rnd.x, rnd.y);
        let delta = params.drift + params.scale * noise;
        current = current * exp(delta);
        iter = iter + 1u;
    }

    Output.data[global_idx] = current;
}
"#;

pub const STOCHASTIC_EVOLUTION_SHADER_F64: &str = r#"
struct Tensor {
    data: array<f64>,
};

struct StochasticEvolutionParams {
    offset: u32,
    chunk: u32,
    len: u32,
    steps: u32,
    key0: u32,
    key1: u32,
    _pad0: u32,
    drift: f64,
    scale: f64,
}

@group(0) @binding(0) var<storage, read> Input: Tensor;
@group(0) @binding(1) var<storage, read_write> Output: Tensor;
@group(0) @binding(2) var<uniform> params: StochasticEvolutionParams;

const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;
const PHILOX_W0: u32 = 0x9E3779B9u;
const PHILOX_W1: u32 = 0xBB67AE85u;
const INV_U32: f32 = 1.0 / 4294967296.0;
const TWO_PI: f32 = 6.2831855;
const MIN_UNIFORM: f32 = 1.0e-8;
const ALMOST_ONE: f32 = 0.99999994;

fn mul_hi_u32(a: u32, b: u32) -> u32 {
    let a_hi = a >> 16u;
    let a_lo = a & 0xFFFFu;
    let b_hi = b >> 16u;
    let b_lo = b & 0xFFFFu;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
    return p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
}

fn philox_round(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let hi0 = mul_hi_u32(PHILOX_M0, counter.x);
    let lo0 = PHILOX_M0 * counter.x;
    let hi1 = mul_hi_u32(PHILOX_M1, counter.z);
    let lo1 = PHILOX_M1 * counter.z;
    return vec4<u32>(
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0,
    );
}

fn philox(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
        ctr = philox_round(ctr, k);
        k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    }
    return ctr;
}

fn uniform_from_bits(bits: u32) -> f32 {
    let sample = (f32(bits) + 0.5) * INV_U32;
    return clamp(sample, MIN_UNIFORM, ALMOST_ONE);
}

fn normal_from_bits(bits0: u32, bits1: u32) -> f32 {
    let u1 = uniform_from_bits(bits0);
    let u2 = uniform_from_bits(bits1);
    let radius = sqrt(-2.0 * log(u1));
    let angle = TWO_PI * u2;
    return radius * cos(angle);
}

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.chunk {
        return;
    }
    let global_idx = params.offset + idx;
    if global_idx >= params.len {
        return;
    }

    var ctr = vec4<u32>(global_idx, 0u, 0u, 0u);
    let key = vec2<u32>(params.key0, params.key1);
    var steps: u32 = params.steps;
    var current = Input.data[global_idx];

    var iter: u32 = 0u;
    loop {
        if iter >= steps {
            break;
        }
        ctr.y = iter;
        let rnd = philox(ctr, key);
        let noise_f32 = normal_from_bits(rnd.x, rnd.y);
        let noise = f64(noise_f32);
        let delta = params.drift + params.scale * noise;
        current = current * exp(delta);
        iter = iter + 1u;
    }

    Output.data[global_idx] = current;
}
"#;
