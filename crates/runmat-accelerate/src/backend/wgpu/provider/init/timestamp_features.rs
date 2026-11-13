//! Reliable optional instrumentation is admitted before device creation.

pub(super) fn requested(backend: wgpu::Backend, advertised: wgpu::Features) -> wgpu::Features {
    if !advertised.contains(wgpu::Features::TIMESTAMP_QUERY) {
        return wgpu::Features::empty();
    }
    let supported = match backend {
        // Encoder-level writes can prevent Metal submissions from completing,
        // even when the adapter advertises support. Descriptor-based compute
        // timestamps remain usable. Do not synthesize the missing outer interval
        // from pass sums or host waits. Requalify before enabling this extension.
        // Upstream: gfx-rs/wgpu-native#624; local direct/selected timing controls.
        wgpu::Backend::Metal => wgpu::Features::TIMESTAMP_QUERY,
        _ => wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    };
    advertised & supported
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_negotiation_preserves_compute_without_admitting_unreliable_encoder_writes() {
        let timestamps =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        for backend in [
            wgpu::Backend::Metal,
            wgpu::Backend::Vulkan,
            wgpu::Backend::Dx12,
            wgpu::Backend::Gl,
            wgpu::Backend::BrowserWebGpu,
        ] {
            assert!(requested(backend, wgpu::Features::empty()).is_empty());
            assert!(requested(backend, wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS).is_empty());
            assert_eq!(
                requested(backend, wgpu::Features::TIMESTAMP_QUERY),
                wgpu::Features::TIMESTAMP_QUERY
            );
            let expected = if backend == wgpu::Backend::Metal {
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                timestamps
            };
            assert_eq!(
                requested(backend, timestamps | wgpu::Features::SHADER_F64),
                expected
            );
        }
    }
}
