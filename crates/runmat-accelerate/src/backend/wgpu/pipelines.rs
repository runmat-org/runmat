use crate::backend::wgpu::bindings::{storage_read_entry, storage_read_write_entry, uniform_entry};
use crate::backend::wgpu::types::NumericPrecision;
use std::borrow::Cow;

// Shader aliases
const BINARY_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::BINARY_SHADER_F64;
const BINARY_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::BINARY_SHADER_F32;
const BINARY_BROADCAST_SHADER_F64: &str =
    crate::backend::wgpu::shaders::elementwise::BINARY_BROADCAST_SHADER_F64;
const BINARY_BROADCAST_SHADER_F32: &str =
    crate::backend::wgpu::shaders::elementwise::BINARY_BROADCAST_SHADER_F32;
const UNARY_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::UNARY_SHADER_F64;
const UNARY_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::UNARY_SHADER_F32;
const SCALAR_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::SCALAR_SHADER_F64;
const SCALAR_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::SCALAR_SHADER_F32;
const TRANSPOSE_SHADER_F64: &str = crate::backend::wgpu::shaders::transpose::TRANSPOSE_SHADER_F64;
const TRANSPOSE_SHADER_F32: &str = crate::backend::wgpu::shaders::transpose::TRANSPOSE_SHADER_F32;
const PERMUTE_SHADER_F64: &str = crate::backend::wgpu::shaders::permute::PERMUTE_SHADER_F64;
const PERMUTE_SHADER_F32: &str = crate::backend::wgpu::shaders::permute::PERMUTE_SHADER_F32;
const FLIP_SHADER_F64: &str = crate::backend::wgpu::shaders::flip::FLIP_SHADER_F64;
const FLIP_SHADER_F32: &str = crate::backend::wgpu::shaders::flip::FLIP_SHADER_F32;
const CIRCSHIFT_SHADER_F64: &str = crate::backend::wgpu::shaders::circshift::CIRCSHIFT_SHADER_F64;
const CIRCSHIFT_SHADER_F32: &str = crate::backend::wgpu::shaders::circshift::CIRCSHIFT_SHADER_F32;
const DIFF_SHADER_F64: &str = crate::backend::wgpu::shaders::diff::DIFF_SHADER_F64;
const DIFF_SHADER_F32: &str = crate::backend::wgpu::shaders::diff::DIFF_SHADER_F32;
const CUMSUM_SHADER_F64: &str = crate::backend::wgpu::shaders::scan::CUMSUM_SHADER_F64;
const CUMSUM_SHADER_F32: &str = crate::backend::wgpu::shaders::scan::CUMSUM_SHADER_F32;
const REPMAT_SHADER_F64: &str = crate::backend::wgpu::shaders::repmat::REPMAT_SHADER_F64;
const REPMAT_SHADER_F32: &str = crate::backend::wgpu::shaders::repmat::REPMAT_SHADER_F32;
const KRON_SHADER_F64: &str = crate::backend::wgpu::shaders::kron::KRON_SHADER_F64;
const KRON_SHADER_F32: &str = crate::backend::wgpu::shaders::kron::KRON_SHADER_F32;
const MATMUL_SHADER_F64: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_F64;
const MATMUL_SHADER_F32: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_F32;
const MATMUL_SHADER_VEC4_F64: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_VEC4_F64;
const MATMUL_SHADER_VEC4_F32: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_VEC4_F32;
const MATMUL_EPILOGUE_SHADER_F64: &str =
    crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F64;
const MATMUL_EPILOGUE_SHADER_F32: &str =
    crate::backend::wgpu::shaders::matmul::MATMUL_EPILOGUE_SHADER_F32;
const MATMUL_SMALLK_SHADER_F64: &str =
    crate::backend::wgpu::shaders::matmul_smallk::MATMUL_SMALLK_SHADER_F64;
const MATMUL_SMALLK_SHADER_F32: &str =
    crate::backend::wgpu::shaders::matmul_smallk::MATMUL_SMALLK_SHADER_F32;
const MATMUL_TALL_SKINNY_SHADER_F64: &str =
    crate::backend::wgpu::shaders::matmul_tall_skinny::MATMUL_TALL_SKINNY_F64;
const MATMUL_TALL_SKINNY_SHADER_F32: &str =
    crate::backend::wgpu::shaders::matmul_tall_skinny::MATMUL_TALL_SKINNY_F32;
const SYRK_SHADER_F64: &str = crate::backend::wgpu::shaders::syrk::SYRK_SHADER_F64;
const SYRK_SHADER_F32: &str = crate::backend::wgpu::shaders::syrk::SYRK_SHADER_F32;
const CENTERED_GRAM_SHADER_F64: &str =
    crate::backend::wgpu::shaders::centered_gram::CENTERED_GRAM_SHADER_F64;
const CENTERED_GRAM_SHADER_F32: &str =
    crate::backend::wgpu::shaders::centered_gram::CENTERED_GRAM_SHADER_F32;
const QR_POWER_ITER_CHOL_SHADER: &str =
    crate::backend::wgpu::shaders::qr_power_iter::QR_POWER_ITER_CHOL_SHADER;
const CONV1D_SHADER_F64: &str = crate::backend::wgpu::shaders::conv::CONV1D_SHADER_F64;
const CONV1D_SHADER_F32: &str = crate::backend::wgpu::shaders::conv::CONV1D_SHADER_F32;
const REDUCE_GLOBAL_SHADER_F64: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_GLOBAL_SHADER_F64;
const REDUCE_GLOBAL_SHADER_F32: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_GLOBAL_SHADER_F32;
const REDUCE_DIM_SHADER_F64: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_SHADER_F64;
const REDUCE_DIM_SHADER_F32: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_SHADER_F32;
const REDUCE_DIM_MINMAX_SHADER_F64: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_DIM_MINMAX_SHADER_F64;
const REDUCE_DIM_MINMAX_SHADER_F32: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_DIM_MINMAX_SHADER_F32;
const EYE_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::EYE_SHADER_F64;
const EYE_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::EYE_SHADER_F32;
const FILL_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::FILL_SHADER_F64;
const FILL_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::FILL_SHADER_F32;
const LINSPACE_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::LINSPACE_SHADER_F64;
const LINSPACE_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::LINSPACE_SHADER_F32;
const RANDOM_INT_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::RANDOM_INT_SHADER_F64;
const RANDOM_INT_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::RANDOM_INT_SHADER_F32;
const RANDOM_UNIFORM_SHADER_F64: &str =
    crate::backend::wgpu::shaders::creation::RANDOM_UNIFORM_SHADER_F64;
const RANDOM_UNIFORM_SHADER_F32: &str =
    crate::backend::wgpu::shaders::creation::RANDOM_UNIFORM_SHADER_F32;
const RANDOM_NORMAL_SHADER_F64: &str =
    crate::backend::wgpu::shaders::creation::RANDOM_NORMAL_SHADER_F64;
const RANDOM_NORMAL_SHADER_F32: &str =
    crate::backend::wgpu::shaders::creation::RANDOM_NORMAL_SHADER_F32;
const RANDPERM_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::RANDPERM_SHADER_F64;
const RANDPERM_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::RANDPERM_SHADER_F32;
const FSPECIAL_SHADER_F64: &str = crate::backend::wgpu::shaders::creation::FSPECIAL_SHADER_F64;
const FSPECIAL_SHADER_F32: &str = crate::backend::wgpu::shaders::creation::FSPECIAL_SHADER_F32;
const POLYVAL_SHADER_F64: &str = crate::backend::wgpu::shaders::polyval::POLYVAL_SHADER_F64;
const POLYVAL_SHADER_F32: &str = crate::backend::wgpu::shaders::polyval::POLYVAL_SHADER_F32;
const POLYDER_SHADER_F64: &str = crate::backend::wgpu::shaders::polyder::POLYDER_SHADER_F64;
const POLYDER_SHADER_F32: &str = crate::backend::wgpu::shaders::polyder::POLYDER_SHADER_F32;
const POLYINT_SHADER_F64: &str = crate::backend::wgpu::shaders::polyint::POLYINT_SHADER_F64;
const POLYINT_SHADER_F32: &str = crate::backend::wgpu::shaders::polyint::POLYINT_SHADER_F32;
const DIAG_FROM_VECTOR_SHADER_F64: &str =
    crate::backend::wgpu::shaders::diag::DIAG_FROM_VECTOR_SHADER_F64;
const DIAG_FROM_VECTOR_SHADER_F32: &str =
    crate::backend::wgpu::shaders::diag::DIAG_FROM_VECTOR_SHADER_F32;
const DIAG_EXTRACT_SHADER_F64: &str = crate::backend::wgpu::shaders::diag::DIAG_EXTRACT_SHADER_F64;
const DIAG_EXTRACT_SHADER_F32: &str = crate::backend::wgpu::shaders::diag::DIAG_EXTRACT_SHADER_F32;
const FIND_SHADER_F64: &str = crate::backend::wgpu::shaders::find::FIND_SHADER_F64;
const FIND_SHADER_F32: &str = crate::backend::wgpu::shaders::find::FIND_SHADER_F32;
const TRIL_SHADER_F64: &str = crate::backend::wgpu::shaders::tril::TRIL_SHADER_F64;
const TRIL_SHADER_F32: &str = crate::backend::wgpu::shaders::tril::TRIL_SHADER_F32;
const TRIU_SHADER_F64: &str = crate::backend::wgpu::shaders::triu::TRIU_SHADER_F64;
const TRIU_SHADER_F32: &str = crate::backend::wgpu::shaders::triu::TRIU_SHADER_F32;
const IMFILTER_SHADER_F64: &str = crate::backend::wgpu::shaders::imfilter::IMFILTER_SHADER_F64;
const IMFILTER_SHADER_F32: &str = crate::backend::wgpu::shaders::imfilter::IMFILTER_SHADER_F32;
const IMAGE_NORMALIZE_SHADER_F64: &str =
    crate::backend::wgpu::shaders::image_normalize::IMAGE_NORMALIZE_SHADER_F64;
const IMAGE_NORMALIZE_SHADER_F32: &str =
    crate::backend::wgpu::shaders::image_normalize::IMAGE_NORMALIZE_SHADER_F32;
const BANDWIDTH_SHADER_F64: &str = crate::backend::wgpu::shaders::bandwidth::BANDWIDTH_SHADER_F64;
const BANDWIDTH_SHADER_F32: &str = crate::backend::wgpu::shaders::bandwidth::BANDWIDTH_SHADER_F32;
const SYMMETRY_SHADER_F64: &str = crate::backend::wgpu::shaders::symmetry::SYMMETRY_SHADER_F64;
const SYMMETRY_SHADER_F32: &str = crate::backend::wgpu::shaders::symmetry::SYMMETRY_SHADER_F32;
const CUMPROD_SHADER_F64: &str = crate::backend::wgpu::shaders::scan::CUMPROD_SHADER_F64;
const CUMPROD_SHADER_F32: &str = crate::backend::wgpu::shaders::scan::CUMPROD_SHADER_F32;
const CUMMIN_SHADER_F64: &str = crate::backend::wgpu::shaders::scan::CUMMIN_SHADER_F64;
const CUMMIN_SHADER_F32: &str = crate::backend::wgpu::shaders::scan::CUMMIN_SHADER_F32;
const CUMMAX_SHADER_F64: &str = crate::backend::wgpu::shaders::scan::CUMMAX_SHADER_F64;
const CUMMAX_SHADER_F32: &str = crate::backend::wgpu::shaders::scan::CUMMAX_SHADER_F32;
const REDUCE_ND_MEAN_SHADER_F64: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_ND_MEAN_SHADER_F64;
const REDUCE_ND_MEAN_SHADER_F32: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_ND_MEAN_SHADER_F32;
const REDUCE_ND_MOMENTS_SHADER_F64: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_ND_MOMENTS_SHADER_F64;
const REDUCE_ND_MOMENTS_SHADER_F32: &str =
    crate::backend::wgpu::shaders::reduction::REDUCE_ND_MOMENTS_SHADER_F32;
const STOCHASTIC_EVOLUTION_SHADER_F64: &str =
    crate::backend::wgpu::shaders::stochastic_evolution::STOCHASTIC_EVOLUTION_SHADER_F64;
const STOCHASTIC_EVOLUTION_SHADER_F32: &str =
    crate::backend::wgpu::shaders::stochastic_evolution::STOCHASTIC_EVOLUTION_SHADER_F32;

pub struct PipelineBundle {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

pub struct WgpuPipelines {
    pub binary: PipelineBundle,
    pub binary_broadcast: PipelineBundle,
    pub unary: PipelineBundle,
    pub scalar: PipelineBundle,
    pub transpose: PipelineBundle,
    pub permute: PipelineBundle,
    pub flip: PipelineBundle,
    pub diff: PipelineBundle,
    pub conv1d: PipelineBundle,
    pub filter: PipelineBundle,
    pub cumsum: PipelineBundle,
    pub cumprod: PipelineBundle,
    pub cummin: PipelineBundle,
    pub cummax: PipelineBundle,
    pub circshift: PipelineBundle,
    pub tril: PipelineBundle,
    pub triu: PipelineBundle,
    pub repmat: PipelineBundle,
    pub kron: PipelineBundle,
    pub matmul: PipelineBundle,
    pub matmul_vec4: PipelineBundle,
    pub matmul_smallk: PipelineBundle,
    pub matmul_tall_skinny: PipelineBundle,
    pub matmul_epilogue: PipelineBundle,
    pub centered_gram: PipelineBundle,
    pub qr_power_iter: PipelineBundle,
    pub syrk: PipelineBundle,
    pub reduce_global: PipelineBundle,
    pub reduce_dim_sum_mean: PipelineBundle,
    pub reduce_dim_minmax: PipelineBundle,
    pub eye: PipelineBundle,
    pub fill: PipelineBundle,
    pub linspace: PipelineBundle,
    pub random_int: PipelineBundle,
    pub random_uniform: PipelineBundle,
    pub random_normal: PipelineBundle,
    pub stochastic_evolution: PipelineBundle,
    pub randperm: PipelineBundle,
    pub fspecial: PipelineBundle,
    pub imfilter: PipelineBundle,
    pub image_normalize: PipelineBundle,
    pub polyval: PipelineBundle,
    pub polyder: PipelineBundle,
    pub polyint: PipelineBundle,
    pub diag_from_vector: PipelineBundle,
    pub diag_extract: PipelineBundle,
    pub find: PipelineBundle,
    pub bandwidth: PipelineBundle,
    pub symmetry: PipelineBundle,
    pub reduce_nd_mean: PipelineBundle,
    pub reduce_nd_moments: PipelineBundle,
}

impl WgpuPipelines {
    pub fn new(device: &wgpu::Device, precision: NumericPrecision) -> Self {
        let binary = create_pipeline(
            device,
            "runmat-binary-layout",
            "runmat-binary-shader",
            "runmat-binary-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => BINARY_SHADER_F64,
                NumericPrecision::F32 => BINARY_SHADER_F32,
            },
        );

        let binary_broadcast = create_pipeline(
            device,
            "runmat-binary-broadcast-layout",
            "runmat-binary-broadcast-shader",
            "runmat-binary-broadcast-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => BINARY_BROADCAST_SHADER_F64,
                NumericPrecision::F32 => BINARY_BROADCAST_SHADER_F32,
            },
        );

        let unary = create_pipeline(
            device,
            "runmat-unary-layout",
            "runmat-unary-shader",
            "runmat-unary-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => UNARY_SHADER_F64,
                NumericPrecision::F32 => UNARY_SHADER_F32,
            },
        );

        let scalar = create_pipeline(
            device,
            "runmat-scalar-layout",
            "runmat-scalar-shader",
            "runmat-scalar-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => SCALAR_SHADER_F64,
                NumericPrecision::F32 => SCALAR_SHADER_F32,
            },
        );

        let transpose = create_pipeline(
            device,
            "runmat-transpose-layout",
            "runmat-transpose-shader",
            "runmat-transpose-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => TRANSPOSE_SHADER_F64,
                NumericPrecision::F32 => TRANSPOSE_SHADER_F32,
            },
        );

        let permute = create_pipeline(
            device,
            "runmat-permute-layout",
            "runmat-permute-shader",
            "runmat-permute-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => PERMUTE_SHADER_F64,
                NumericPrecision::F32 => PERMUTE_SHADER_F32,
            },
        );

        let flip = create_pipeline(
            device,
            "runmat-flip-layout",
            "runmat-flip-shader",
            "runmat-flip-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => FLIP_SHADER_F64,
                NumericPrecision::F32 => FLIP_SHADER_F32,
            },
        );

        let conv1d = create_pipeline(
            device,
            "runmat-conv1d-layout",
            "runmat-conv1d-shader",
            "runmat-conv1d-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => CONV1D_SHADER_F64,
                NumericPrecision::F32 => CONV1D_SHADER_F32,
            },
        );

        let filter = create_pipeline(
            device,
            "runmat-iir-filter-layout",
            "runmat-iir-filter-shader",
            "runmat-iir-filter-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_entry(2),
                storage_read_entry(3),
                storage_read_write_entry(4),
                storage_read_write_entry(5),
                storage_read_write_entry(6),
                uniform_entry(7),
            ],
            match precision {
                NumericPrecision::F64 => crate::backend::wgpu::shaders::filter::FILTER_SHADER_F64,
                NumericPrecision::F32 => crate::backend::wgpu::shaders::filter::FILTER_SHADER_F32,
            },
        );

        let diff = create_pipeline(
            device,
            "runmat-diff-layout",
            "runmat-diff-shader",
            "runmat-diff-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => DIFF_SHADER_F64,
                NumericPrecision::F32 => DIFF_SHADER_F32,
            },
        );

        let cumsum = create_pipeline(
            device,
            "runmat-cumsum-layout",
            "runmat-cumsum-shader",
            "runmat-cumsum-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => CUMSUM_SHADER_F64,
                NumericPrecision::F32 => CUMSUM_SHADER_F32,
            },
        );

        let cumprod = create_pipeline(
            device,
            "runmat-cumprod-layout",
            "runmat-cumprod-shader",
            "runmat-cumprod-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => CUMPROD_SHADER_F64,
                NumericPrecision::F32 => CUMPROD_SHADER_F32,
            },
        );

        let cummin = create_pipeline(
            device,
            "runmat-cummin-layout",
            "runmat-cummin-shader",
            "runmat-cummin-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => CUMMIN_SHADER_F64,
                NumericPrecision::F32 => CUMMIN_SHADER_F32,
            },
        );

        let cummax = create_pipeline(
            device,
            "runmat-cummax-layout",
            "runmat-cummax-shader",
            "runmat-cummax-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => CUMMAX_SHADER_F64,
                NumericPrecision::F32 => CUMMAX_SHADER_F32,
            },
        );

        let tril = create_pipeline(
            device,
            "runmat-tril-layout",
            "runmat-tril-shader",
            "runmat-tril-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => TRIL_SHADER_F64,
                NumericPrecision::F32 => TRIL_SHADER_F32,
            },
        );

        let triu = create_pipeline(
            device,
            "runmat-triu-layout",
            "runmat-triu-shader",
            "runmat-triu-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => TRIU_SHADER_F64,
                NumericPrecision::F32 => TRIU_SHADER_F32,
            },
        );

        let circshift = create_pipeline(
            device,
            "runmat-circshift-layout",
            "runmat-circshift-shader",
            "runmat-circshift-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => CIRCSHIFT_SHADER_F64,
                NumericPrecision::F32 => CIRCSHIFT_SHADER_F32,
            },
        );

        let repmat = create_pipeline(
            device,
            "runmat-repmat-layout",
            "runmat-repmat-shader",
            "runmat-repmat-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REPMAT_SHADER_F64,
                NumericPrecision::F32 => REPMAT_SHADER_F32,
            },
        );

        let kron = create_pipeline(
            device,
            "runmat-kron-layout",
            "runmat-kron-shader",
            "runmat-kron-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => KRON_SHADER_F64,
                NumericPrecision::F32 => KRON_SHADER_F32,
            },
        );

        let matmul = create_pipeline(
            device,
            "runmat-matmul-layout",
            "runmat-matmul-shader",
            "runmat-matmul-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_SHADER_F64,
                NumericPrecision::F32 => MATMUL_SHADER_F32,
            },
        );

        let matmul_vec4 = create_pipeline(
            device,
            "runmat-matmul-vec4-layout",
            "runmat-matmul-vec4-shader",
            "runmat-matmul-vec4-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_SHADER_VEC4_F64,
                NumericPrecision::F32 => MATMUL_SHADER_VEC4_F32,
            },
        );

        let matmul_smallk = create_pipeline(
            device,
            "runmat-matmul-smallk-layout",
            "runmat-matmul-smallk-shader",
            "runmat-matmul-smallk-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_SMALLK_SHADER_F64,
                NumericPrecision::F32 => MATMUL_SMALLK_SHADER_F32,
            },
        );

        let matmul_tall_skinny = create_pipeline(
            device,
            "runmat-matmul-tall-skinny-layout",
            "runmat-matmul-tall-skinny-shader",
            "runmat-matmul-tall-skinny-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_TALL_SKINNY_SHADER_F64,
                NumericPrecision::F32 => MATMUL_TALL_SKINNY_SHADER_F32,
            },
        );

        let matmul_epilogue = create_pipeline(
            device,
            "runmat-matmul-epilogue-layout",
            "runmat-matmul-epilogue-shader",
            "runmat-matmul-epilogue-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
                storage_read_entry(4),
                storage_read_entry(5),
                uniform_entry(6),
            ],
            match precision {
                NumericPrecision::F64 => MATMUL_EPILOGUE_SHADER_F64,
                NumericPrecision::F32 => MATMUL_EPILOGUE_SHADER_F32,
            },
        );

        let centered_gram = create_pipeline(
            device,
            "runmat-centered-gram-layout",
            "runmat-centered-gram-shader",
            "runmat-centered-gram-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => CENTERED_GRAM_SHADER_F64,
                NumericPrecision::F32 => CENTERED_GRAM_SHADER_F32,
            },
        );

        let qr_power_iter = create_pipeline(
            device,
            "runmat-qr-power-layout",
            "runmat-qr-power-shader",
            "runmat-qr-power-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            QR_POWER_ITER_CHOL_SHADER,
        );

        let syrk = create_pipeline(
            device,
            "runmat-syrk-layout",
            "runmat-syrk-shader",
            "runmat-syrk-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => SYRK_SHADER_F64,
                NumericPrecision::F32 => SYRK_SHADER_F32,
            },
        );

        let reduce_global = create_pipeline(
            device,
            "runmat-reduce-global-layout",
            "runmat-reduce-global-shader",
            "runmat-reduce-global-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_GLOBAL_SHADER_F64,
                NumericPrecision::F32 => REDUCE_GLOBAL_SHADER_F32,
            },
        );

        let reduce_dim_sum_mean = create_pipeline(
            device,
            "runmat-reduce-dim-layout",
            "runmat-reduce-dim-shader",
            "runmat-reduce-dim-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_DIM_SHADER_F64,
                NumericPrecision::F32 => REDUCE_DIM_SHADER_F32,
            },
        );

        let reduce_dim_minmax = create_pipeline(
            device,
            "runmat-reduce-dim-minmax-layout",
            "runmat-reduce-dim-minmax-shader",
            "runmat-reduce-dim-minmax-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_DIM_MINMAX_SHADER_F64,
                NumericPrecision::F32 => REDUCE_DIM_MINMAX_SHADER_F32,
            },
        );

        let eye = create_pipeline(
            device,
            "runmat-eye-layout",
            "runmat-eye-shader",
            "runmat-eye-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => EYE_SHADER_F64,
                NumericPrecision::F32 => EYE_SHADER_F32,
            },
        );

        let fill = create_pipeline(
            device,
            "runmat-fill-layout",
            "runmat-fill-shader",
            "runmat-fill-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => FILL_SHADER_F64,
                NumericPrecision::F32 => FILL_SHADER_F32,
            },
        );

        let linspace = create_pipeline(
            device,
            "runmat-linspace-layout",
            "runmat-linspace-shader",
            "runmat-linspace-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => LINSPACE_SHADER_F64,
                NumericPrecision::F32 => LINSPACE_SHADER_F32,
            },
        );

        let random_int = create_pipeline(
            device,
            "runmat-random-int-layout",
            "runmat-random-int-shader",
            "runmat-random-int-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => RANDOM_INT_SHADER_F64,
                NumericPrecision::F32 => RANDOM_INT_SHADER_F32,
            },
        );

        let random_uniform = create_pipeline(
            device,
            "runmat-random-uniform-layout",
            "runmat-random-uniform-shader",
            "runmat-random-uniform-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => RANDOM_UNIFORM_SHADER_F64,
                NumericPrecision::F32 => RANDOM_UNIFORM_SHADER_F32,
            },
        );

        let random_normal = create_pipeline(
            device,
            "runmat-random-normal-layout",
            "runmat-random-normal-shader",
            "runmat-random-normal-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => RANDOM_NORMAL_SHADER_F64,
                NumericPrecision::F32 => RANDOM_NORMAL_SHADER_F32,
            },
        );

        let stochastic_evolution = create_pipeline(
            device,
            "runmat-stochastic-evolution-layout",
            "runmat-stochastic-evolution-shader",
            "runmat-stochastic-evolution-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => STOCHASTIC_EVOLUTION_SHADER_F64,
                NumericPrecision::F32 => STOCHASTIC_EVOLUTION_SHADER_F32,
            },
        );

        let randperm = create_pipeline(
            device,
            "runmat-randperm-layout",
            "runmat-randperm-shader",
            "runmat-randperm-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => RANDPERM_SHADER_F64,
                NumericPrecision::F32 => RANDPERM_SHADER_F32,
            },
        );

        let fspecial = create_pipeline(
            device,
            "runmat-fspecial-layout",
            "runmat-fspecial-shader",
            "runmat-fspecial-pipeline",
            vec![storage_read_write_entry(0), uniform_entry(1)],
            match precision {
                NumericPrecision::F64 => FSPECIAL_SHADER_F64,
                NumericPrecision::F32 => FSPECIAL_SHADER_F32,
            },
        );

        let imfilter = create_pipeline(
            device,
            "runmat-imfilter-layout",
            "runmat-imfilter-shader",
            "runmat-imfilter-pipeline",
            vec![
                storage_read_entry(0), // Image
                storage_read_entry(1), // KernelOffsets
                storage_read_entry(2), // KernelWeights
                storage_read_write_entry(3),
                uniform_entry(4),
            ],
            match precision {
                NumericPrecision::F64 => IMFILTER_SHADER_F64,
                NumericPrecision::F32 => IMFILTER_SHADER_F32,
            },
        );

        let image_normalize = create_pipeline(
            device,
            "runmat-image-normalize-layout",
            "runmat-image-normalize-shader",
            "runmat-image-normalize-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => IMAGE_NORMALIZE_SHADER_F64,
                NumericPrecision::F32 => IMAGE_NORMALIZE_SHADER_F32,
            },
        );

        let polyval = create_pipeline(
            device,
            "runmat-polyval-layout",
            "runmat-polyval-shader",
            "runmat-polyval-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => POLYVAL_SHADER_F64,
                NumericPrecision::F32 => POLYVAL_SHADER_F32,
            },
        );

        let polyder = create_pipeline(
            device,
            "runmat-polyder-layout",
            "runmat-polyder-shader",
            "runmat-polyder-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => POLYDER_SHADER_F64,
                NumericPrecision::F32 => POLYDER_SHADER_F32,
            },
        );

        let polyint = create_pipeline(
            device,
            "runmat-polyint-layout",
            "runmat-polyint-shader",
            "runmat-polyint-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => POLYINT_SHADER_F64,
                NumericPrecision::F32 => POLYINT_SHADER_F32,
            },
        );

        let diag_from_vector = create_pipeline(
            device,
            "runmat-diag-from-vector-layout",
            "runmat-diag-from-vector-shader",
            "runmat-diag-from-vector-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => DIAG_FROM_VECTOR_SHADER_F64,
                NumericPrecision::F32 => DIAG_FROM_VECTOR_SHADER_F32,
            },
        );

        let diag_extract = create_pipeline(
            device,
            "runmat-diag-extract-layout",
            "runmat-diag-extract-shader",
            "runmat-diag-extract-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => DIAG_EXTRACT_SHADER_F64,
                NumericPrecision::F32 => DIAG_EXTRACT_SHADER_F32,
            },
        );

        let find = create_pipeline(
            device,
            "runmat-find-layout",
            "runmat-find-shader",
            "runmat-find-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                storage_read_write_entry(3),
                storage_read_write_entry(4),
                storage_read_write_entry(5),
                uniform_entry(6),
            ],
            match precision {
                NumericPrecision::F64 => FIND_SHADER_F64,
                NumericPrecision::F32 => FIND_SHADER_F32,
            },
        );

        let bandwidth = create_pipeline(
            device,
            "runmat-bandwidth-layout",
            "runmat-bandwidth-shader",
            "runmat-bandwidth-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => BANDWIDTH_SHADER_F64,
                NumericPrecision::F32 => BANDWIDTH_SHADER_F32,
            },
        );

        let symmetry = create_pipeline(
            device,
            "runmat-symmetry-layout",
            "runmat-symmetry-shader",
            "runmat-symmetry-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => SYMMETRY_SHADER_F64,
                NumericPrecision::F32 => SYMMETRY_SHADER_F32,
            },
        );

        let reduce_nd_mean = create_pipeline(
            device,
            "runmat-reduce-nd-mean-layout",
            "runmat-reduce-nd-mean-shader",
            "runmat-reduce-nd-mean-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                uniform_entry(2),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_ND_MEAN_SHADER_F64,
                NumericPrecision::F32 => REDUCE_ND_MEAN_SHADER_F32,
            },
        );

        let reduce_nd_moments = create_pipeline(
            device,
            "runmat-reduce-nd-moments-layout",
            "runmat-reduce-nd-moments-shader",
            "runmat-reduce-nd-moments-pipeline",
            vec![
                storage_read_entry(0),
                storage_read_write_entry(1),
                storage_read_write_entry(2),
                uniform_entry(3),
            ],
            match precision {
                NumericPrecision::F64 => REDUCE_ND_MOMENTS_SHADER_F64,
                NumericPrecision::F32 => REDUCE_ND_MOMENTS_SHADER_F32,
            },
        );

        Self {
            binary,
            binary_broadcast,
            unary,
            scalar,
            transpose,
            permute,
            flip,
            diff,
            conv1d,
            filter,
            cumsum,
            cumprod,
            cummin,
            cummax,
            circshift,
            tril,
            triu,
            repmat,
            kron,
            matmul,
            matmul_vec4,
            matmul_smallk,
            matmul_tall_skinny,
            matmul_epilogue,
            centered_gram,
            qr_power_iter,
            syrk,
            reduce_global,
            reduce_dim_sum_mean,
            reduce_dim_minmax,
            eye,
            fill,
            linspace,
            random_int,
            random_uniform,
            random_normal,
            stochastic_evolution,
            randperm,
            fspecial,
            imfilter,
            image_normalize,
            polyval,
            polyder,
            polyint,
            diag_from_vector,
            diag_extract,
            find,
            bandwidth,
            symmetry,
            reduce_nd_mean,
            reduce_nd_moments,
        }
    }
}

fn substitute_tokens(src: &str, wg: u32, tile: u32) -> Cow<'_, str> {
    if src.contains("@WG@") || src.contains("@MT@") {
        let mut s = src.to_string();
        s = s.replace("@WG@", &wg.to_string());
        s = s.replace("@MT@", &tile.to_string());
        Cow::Owned(s)
    } else {
        Cow::Borrowed(src)
    }
}

pub fn create_pipeline(
    device: &wgpu::Device,
    layout_label: &str,
    shader_label: &str,
    pipeline_label: &str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    shader_source: &str,
) -> PipelineBundle {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(layout_label),
        entries: &entries,
    });
    let wg = crate::backend::wgpu::config::effective_workgroup_size();
    let mt = crate::backend::wgpu::config::effective_matmul_tile();
    let patched = substitute_tokens(shader_source, wg, mt);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(patched),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&(pipeline_label.to_string() + "-layout")),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            }),
        ),
        module: &module,
        entry_point: "main",
    });

    PipelineBundle { pipeline, layout }
}

pub fn create_shader_module(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ShaderModule {
    let wg = crate::backend::wgpu::config::effective_workgroup_size();
    let mt = crate::backend::wgpu::config::effective_matmul_tile();
    let patched = substitute_tokens(wgsl, wg, mt);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(patched),
    })
}

pub fn create_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    })
}
