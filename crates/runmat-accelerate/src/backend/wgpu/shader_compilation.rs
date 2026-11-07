//! One fallible, device-independent preparation boundary for every GPU shader.
//!
//! Original WGSL and the compiler revision identify cache entries. RunMat's
//! token/rewrite/control buffers retain their explicit admission claims;
//! Naga's own compiler/output storage and WGPU/driver-private allocations are
//! excluded, not inferred from source length. Preparation finishes before
//! shader creation; uncaptured device errors are not converted into Results.

use crate::wgsl::host_buffers::ClaimedString;
#[cfg(not(target_arch = "wasm32"))]
use crate::wgsl::host_buffers::ClaimedVec;
use runmat_execution::resource::ProviderHostMetadataAdmission;
use std::borrow::Cow;

pub(crate) mod capabilities;
#[cfg(not(target_arch = "wasm32"))]
mod controls;
#[cfg(not(target_arch = "wasm32"))]
mod unsigned_conversion;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod spirv;

pub(crate) struct PreparedModule {
    source: PreparedSource,
    required_features: wgpu::Features,
}

enum PreparedSource {
    Wgsl(ClaimedString),
    #[cfg(not(target_arch = "wasm32"))]
    ControlledSpirV(ClaimedVec<u32>),
}

pub(crate) fn prepare(
    source: &str,
    features: wgpu::Features,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> anyhow::Result<PreparedModule> {
    let source = crate::wgsl::substitute_tokens(
        source,
        super::config::effective_workgroup_size(),
        super::config::effective_matmul_tile(),
        metadata,
    )?;
    let prepared = crate::wgsl::f64_math::prepare(source.as_str(), metadata)?;
    if !prepared.requires_f64_math() {
        return Ok(PreparedModule {
            source: PreparedSource::Wgsl(prepared.into_source()),
            required_features: wgpu::Features::empty(),
        });
    }
    let required = capabilities::REQUIRED_FEATURES;
    anyhow::ensure!(
        features.contains(required),
        "portable binary64 shader requires F64, INT64 and controlled SPIR-V support"
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(PreparedModule {
            source: PreparedSource::ControlledSpirV(spirv::compile(prepared.source(), metadata)?),
            required_features: required,
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        anyhow::bail!("portable binary64 shaders are unavailable on WebGPU")
    }
}

/// Admission binds the capability check to the exact emission used to create
/// the module. There is deliberately no way to supply a different emission to
/// an admitted receipt's infallible creation method.
pub(crate) struct AdmittedModule<'source, 'emission, 'device> {
    source: &'source PreparedModule,
    emission: &'emission runmat_accelerate_api::WgpuEmission<'device>,
}

impl PreparedModule {
    pub(crate) fn admit<'source, 'emission, 'device>(
        &'source self,
        emission: &'emission runmat_accelerate_api::WgpuEmission<'device>,
    ) -> anyhow::Result<AdmittedModule<'source, 'emission, 'device>> {
        anyhow::ensure!(
            emission
                .device()
                .features()
                .contains(self.required_features),
            "prepared shader requires features unavailable on the emission device"
        );
        Ok(AdmittedModule {
            source: self,
            emission,
        })
    }
}

impl AdmittedModule<'_, '_, '_> {
    pub(crate) fn create(&self, label: &str) -> wgpu::ShaderModule {
        match &self.source.source {
            PreparedSource::Wgsl(source) => {
                self.emission
                    .device()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source.as_str())),
                    })
            }
            #[cfg(not(target_arch = "wasm32"))]
            PreparedSource::ControlledSpirV(words) => {
                // SAFETY: only validated, audited compiler output enters the
                // opaque receipt. Admission checked the capabilities of this
                // exact borrowed device, which cannot be exchanged afterward.
                unsafe {
                    self.emission.device().create_shader_module_spirv(
                        &wgpu::ShaderModuleDescriptorSpirV {
                            label: Some(label),
                            source: Cow::Borrowed(words.as_slice()),
                        },
                    )
                }
            }
        }
    }
}

pub(crate) fn create_prepared(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    source: &PreparedModule,
) -> anyhow::Result<wgpu::ShaderModule> {
    Ok(source.admit(emission)?.create(label))
}

pub(crate) fn create_shader_module(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    source: &str,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
) -> anyhow::Result<wgpu::ShaderModule> {
    let prepared = prepare(source, emission.device().features(), metadata)?;
    create_prepared(emission, label, &prepared)
}
