//! Alternate native F32 kernels for a provider that prefers F64.
//! The provider-native bundles remain the fast path; all bundles compile on
//! first selected use rather than at adapter creation.

use super::*;

pub(crate) struct F32RealFloatingPipelines {
    pub(crate) binary: PipelineBundle,
    pub(crate) binary_broadcast: PipelineBundle,
    pub(crate) binary_max: PipelineBundle,
    pub(crate) binary_broadcast_max: PipelineBundle,
    pub(crate) scalar: PipelineBundle,
    pub(crate) scalar_max: PipelineBundle,
    pub(crate) reduce_nd_mean: PipelineBundle,
    pub(crate) reduce_nd_moments: PipelineBundle,
    pub(crate) tril: PipelineBundle,
    pub(crate) triu: PipelineBundle,
    pub(crate) kron: PipelineBundle,
}

impl F32RealFloatingPipelines {
    pub(crate) fn new(
        access: &runmat_accelerate_api::WgpuDeviceAccess,
        linear_launch: FrozenLinearLaunch,
        linear_tokens: FrozenShaderTokens,
    ) -> Self {
        Self {
            binary: create_pipeline(
                access,
                "runmat-f32-real-binary-layout",
                "runmat-f32-real-binary-shader",
                "runmat-f32-real-binary-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                BINARY_SHADER_F32,
            ).with_binary_launch(linear_launch),
            binary_broadcast: create_pipeline(
                access,
                "runmat-f32-real-binary-broadcast-layout",
                "runmat-f32-real-binary-broadcast-shader",
                "runmat-f32-real-binary-broadcast-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                BINARY_BROADCAST_SHADER_F32,
            ).with_binary_launch(linear_launch),
            binary_max: create_pipeline(
                access,
                "runmat-f32-real-binary-max-layout",
                "runmat-f32-real-binary-max-shader",
                "runmat-f32-real-binary-max-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                crate::backend::wgpu::shaders::real_max::binary_shader(
                    NumericPrecision::F32,
                    false,
                    crate::backend::wgpu::shaders::real_max::RealMaxPolicy::IncludeNaNPreferPositiveZero,
                ),
            ).with_binary_launch(linear_launch),
            binary_broadcast_max: create_pipeline(
                access,
                "runmat-f32-real-binary-max-broadcast-layout",
                "runmat-f32-real-binary-max-broadcast-shader",
                "runmat-f32-real-binary-max-broadcast-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                crate::backend::wgpu::shaders::real_max::binary_shader(
                    NumericPrecision::F32,
                    true,
                    crate::backend::wgpu::shaders::real_max::RealMaxPolicy::IncludeNaNPreferPositiveZero,
                ),
            ).with_binary_launch(linear_launch),
            scalar: create_pipeline(
                access,
                "runmat-f32-real-scalar-layout",
                "runmat-f32-real-scalar-shader",
                "runmat-f32-real-scalar-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                SCALAR_SHADER_F32,
            )
            .with_scalar_launch(NumericPrecision::F32, linear_launch),
            scalar_max: create_pipeline(
                access,
                "runmat-f32-real-scalar-max-layout",
                "runmat-f32-real-scalar-max-shader",
                "runmat-f32-real-scalar-max-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                SCALAR_SHADER_F32,
            )
            .with_scalar_max_launch(NumericPrecision::F32, linear_launch),
            reduce_nd_mean: create_pipeline(
                access,
                "runmat-f32-real-nd-mean-layout",
                "runmat-f32-real-nd-mean-shader",
                "runmat-f32-real-nd-mean-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                REDUCE_ND_MEAN_SHADER_F32,
            ),
            reduce_nd_moments: create_pipeline(
                access,
                "runmat-f32-real-nd-moments-layout",
                "runmat-f32-real-nd-moments-shader",
                "runmat-f32-real-nd-moments-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                REDUCE_ND_MOMENTS_SHADER_F32,
            ),
            tril: create_pipeline(
                access,
                "runmat-f32-real-tril-layout",
                "runmat-f32-real-tril-shader",
                "runmat-f32-real-tril-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                TRIL_SHADER_F32,
            )
            .with_linear_launch(linear_tokens),
            triu: create_pipeline(
                access,
                "runmat-f32-real-triu-layout",
                "runmat-f32-real-triu-shader",
                "runmat-f32-real-triu-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_write_entry(1),
                    uniform_entry(2),
                ],
                TRIU_SHADER_F32,
            )
            .with_linear_launch(linear_tokens),
            kron: create_pipeline(
                access,
                "runmat-f32-real-kron-layout",
                "runmat-f32-real-kron-shader",
                "runmat-f32-real-kron-pipeline",
                vec![
                    storage_read_entry(0),
                    storage_read_entry(1),
                    storage_read_write_entry(2),
                    uniform_entry(3),
                ],
                KRON_SHADER_F32,
            )
            .with_linear_launch(linear_tokens),
        }
    }
}
