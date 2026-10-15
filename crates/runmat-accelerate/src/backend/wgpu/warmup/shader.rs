//! Persisted WGSL is untrusted input, even when its content hash matches.
use anyhow::{anyhow, Result};
use wgpu::naga;

pub(super) fn validate(source: &str, features: wgpu::Features) -> Result<()> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| anyhow!("cached WGSL parse failed: {}", error.emit_to_string(source)))?;
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
        .map_err(|error| anyhow!("cached WGSL validation failed: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_shader_admission_checks_language_and_enabled_numeric_features() {
        let single = "@compute @workgroup_size(1) fn main() { let x = f32(1.0); }";
        assert!(validate(single, wgpu::Features::empty()).is_ok());
        let double = "@compute @workgroup_size(1) fn main() { let x = f64(1.0); }";
        assert!(validate(double, wgpu::Features::empty()).is_err());
        assert!(validate(double, wgpu::Features::SHADER_F64).is_ok());
        for invalid in [
            "@compute @workgroup_size(1) fn main() { let x = 1.0 / 0.0; }",
            "@compute @workgroup_size(1) fn main() { let x = 0.0 / 0.0; }",
            "@compute @workgroup_size(1) fn main() { let x = 3.402823669209385e38f; }",
            "@compute @workgroup_size(0) fn main() {}",
        ] {
            assert!(
                validate(invalid, wgpu::Features::empty()).is_err(),
                "{invalid}"
            );
        }
    }
}
