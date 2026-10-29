//! GPU-free Naga validation for production WGSL from fusion and WGPU.

#[derive(Clone, Copy, Debug)]
pub(crate) enum RequiredCaps {
    None,
    Float64,
    Float64Int64,
}

impl RequiredCaps {
    fn allowed(self) -> naga::valid::Capabilities {
        use naga::valid::Capabilities as Caps;
        match self {
            Self::None => Caps::empty(),
            Self::Float64 => Caps::FLOAT64,
            Self::Float64Int64 => Caps::FLOAT64 | Caps::SHADER_INT64,
        }
    }

    fn missing_one(self) -> [Option<naga::valid::Capabilities>; 2] {
        use naga::valid::Capabilities as Caps;
        match self {
            Self::None => [None, None],
            Self::Float64 => [Some(Caps::empty()), None],
            Self::Float64Int64 => [Some(Caps::FLOAT64), Some(Caps::SHADER_INT64)],
        }
    }
}

pub(crate) fn validate_generated(
    name: &str,
    source: &str,
    workgroup: u32,
    tile: u32,
    required: RequiredCaps,
) {
    let source = super::substitute_tokens(source, workgroup, tile, None).unwrap();
    assert!(!source.contains("@IEEE_"), "{name}: unbound IEEE marker");
    assert!(!source.contains("@WG@"), "{name}: unbound workgroup marker");
    assert!(!source.contains("@MT@"), "{name}: unbound tile marker");

    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
    let flags = naga::valid::ValidationFlags::all();
    naga::valid::Validator::new(flags, required.allowed())
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} ({required:?}): {error:?}\n{source}"));

    for insufficient in required.missing_one().into_iter().flatten() {
        assert!(
            naga::valid::Validator::new(flags, insufficient)
                .validate(&module)
                .is_err(),
            "{name}: unexpectedly valid with insufficient {insufficient:?} for {required:?}"
        );
    }

    // Frontend acceptance alone permits binary64 calls whose corresponding
    // GLSL extended instructions have no legal binary64 form. Exercise the
    // same typed preparation and controlled emission as production.
    let prepared = crate::wgsl::f64_math::prepare(&source, None)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if prepared.requires_f64_math() && !module.entry_points.is_empty() {
        crate::backend::wgpu::shader_compilation::spirv::compile(prepared.source(), None)
            .unwrap_or_else(|error| panic!("{name}: controlled SPIR-V: {error:#}"));
    }
    #[cfg(not(all(feature = "wgpu", not(target_arch = "wasm32"))))]
    let _ = prepared;
}

#[test]
fn native_f32_and_f64_sources_validate_without_a_backend() {
    let f32 = "@group(0) @binding(0) var<storage, read_write> out: array<f32>; @compute @workgroup_size(@WG@) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = 1.0f; }";
    let f64 = "@group(0) @binding(0) var<storage, read_write> out: array<f64>; @compute @workgroup_size(@WG@) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = f64(1.0); }";
    validate_generated("pure-f32", f32, 64, 8, RequiredCaps::None);
    validate_generated("pure-f64", f64, 64, 8, RequiredCaps::Float64);
}
