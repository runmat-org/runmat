//! Certified Double-scalar POWER on binary32 storage, without float arithmetic.
//! Not a binary64 emulator. A binary32 square has at most 48 significant bits;
//! the exact product is also the binary64 product, then rounded once to F32.
use runmat_accelerate_api::ExactSingleBasePower;

pub(crate) fn shader_for_layout(operation: ExactSingleBasePower, broadcast: bool) -> String {
    let entry = if broadcast { BROADCAST_ENTRY } else { ENTRY };
    format!(
        "{}\n{}",
        crate::kernel_math::power_exact::functions(),
        entry.replace(
            "apply(",
            &format!(
                "{}(",
                crate::kernel_math::power_exact::entrypoint(operation)
            )
        )
    )
}

const ENTRY: &str = r#"
struct Tensor { data: array<u32> };
struct Params { len: u32, offset: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.len { return; }
    let idx = params.offset + gid.x;
    Out.data[idx] = apply(A.data[idx]);
}
"#;

const BROADCAST_ENTRY: &str = r#"
const MAX_RANK: u32 = 128u;
struct PackedValue { value: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
alias PackedArray = array<PackedValue, MAX_RANK>;
struct Tensor { data: array<u32> };
struct Params {
    len: u32, offset: u32, rank: u32, _pad: u32,
    out_shape: PackedArray, in0_shape: PackedArray, in0_stride: PackedArray,
};
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.len { return; }
    let idx = params.offset + gid.x;
    var remaining = idx;
    var source = 0u;
    for (var d = 0u; d < params.rank; d = d + 1u) {
        let coordinate = remaining % params.out_shape[d].value;
        remaining = remaining / params.out_shape[d].value;
        if params.in0_shape[d].value != 1u {
            source = source + coordinate * params.in0_stride[d].value;
        }
    }
    Out.data[idx] = apply(A.data[source]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[test]
    fn exact_scalar_power_uses_only_baseline_integer_words() {
        for operation in [
            ExactSingleBasePower::Zero,
            ExactSingleBasePower::Identity,
            ExactSingleBasePower::Square,
        ] {
            for broadcast in [false, true] {
                let source = shader_for_layout(operation, broadcast);
                assert!(!source.contains("f32"));
                assert!(!source.contains("f64"));
                assert!(!source.contains("u64"));
                validate_generated(
                    "exact-single-base-power",
                    &source,
                    64,
                    8,
                    RequiredCaps::None,
                );
            }
        }
    }
}
