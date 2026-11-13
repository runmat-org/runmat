//! Persisted WGSL is untrusted input, even when its content hash matches.
use std::fmt;
use wgpu::naga;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum CachedShaderAdmissionError {
    InvalidSource(String),
    IncompatibleAdapter(String),
}

impl fmt::Display for CachedShaderAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource(detail) => write!(f, "invalid cached WGSL: {detail}"),
            Self::IncompatibleAdapter(detail) => {
                write!(
                    f,
                    "cached WGSL exceeds enabled adapter capabilities: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for CachedShaderAdmissionError {}

pub(super) fn validate(
    source: &str,
    features: wgpu::Features,
) -> Result<(), CachedShaderAdmissionError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| CachedShaderAdmissionError::InvalidSource(error.emit_to_string(source)))?;
    // Validate source semantics independently of this adapter first. A valid
    // shader needing another device's optional capability is not corruption.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| CachedShaderAdmissionError::InvalidSource(error.to_string()))?;
    // Warmup is optional: admit only the baseline compute language and the
    // numeric extensions explicitly enabled on this device. Other extensions
    // can still be prepared through their owning runtime recipe.
    let mut capabilities = naga::valid::Capabilities::empty();
    if features.contains(wgpu::Features::SHADER_F64) {
        capabilities |= naga::valid::Capabilities::FLOAT64;
    }
    if features.contains(wgpu::Features::SHADER_INT64) {
        capabilities |= naga::valid::Capabilities::SHADER_INT64;
    }
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|error| CachedShaderAdmissionError::IncompatibleAdapter(error.to_string()))?;
    let prepared = crate::wgsl::f64_math::prepare(source, None)
        .map_err(|error| CachedShaderAdmissionError::InvalidSource(error.to_string()))?;
    let controlled_features = super::super::shader_compilation::capabilities::REQUIRED_FEATURES;
    if prepared.requires_f64_math() && !features.contains(controlled_features) {
        return Err(CachedShaderAdmissionError::IncompatibleAdapter(
            "binary64 shader requires the controlled arithmetic feature contract".into(),
        ));
    }
    // A valid frontend module is not sufficient: warmup must pass the same
    // legalization, bounds checking and controlled-emission audit as a cold
    // runtime compilation. This performs no device allocation/publication.
    super::super::shader_compilation::prepare(source, features, None)
        .map_err(|error| CachedShaderAdmissionError::InvalidSource(error.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_shader_admission_checks_language_and_enabled_numeric_features() {
        let single = "@compute @workgroup_size(1) fn main() { let x = f32(1.0); }";
        assert!(validate(single, wgpu::Features::empty()).is_ok());
        // An unused constant is folded out by the frontend and has no runtime
        // numeric capability requirement. Retain an observable binary64 type.
        let double = "@group(0) @binding(0) var<storage, read_write> out: array<f64>; @compute @workgroup_size(1) fn main() { out[0] = f64(1.0); }";
        assert!(matches!(
            validate(double, wgpu::Features::empty()),
            Err(CachedShaderAdmissionError::IncompatibleAdapter(_))
        ));
        assert!(matches!(
            validate(double, wgpu::Features::SHADER_F64),
            Err(CachedShaderAdmissionError::IncompatibleAdapter(_))
        ));
        #[cfg(not(target_arch = "wasm32"))]
        assert!(validate(
            double,
            wgpu::Features::SHADER_F64
                | wgpu::Features::SHADER_INT64
                | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
        )
        .is_ok());
        for invalid in [
            "@compute @workgroup_size(@WG@) fn main() {}",
            "@compute @workgroup_size(1) fn main() { let x = 1.0 / 0.0; }",
            "@compute @workgroup_size(1) fn main() { let x = 0.0 / 0.0; }",
            "@compute @workgroup_size(1) fn main() { let x = 3.402823669209385e38f; }",
            "@compute @workgroup_size(0) fn main() {}",
        ] {
            assert!(
                matches!(
                    validate(invalid, wgpu::Features::empty()),
                    Err(CachedShaderAdmissionError::InvalidSource(_))
                ),
                "{invalid}"
            );
        }
    }
}
