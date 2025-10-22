use crate::backend::wgpu::bindings::{storage_read_entry, storage_read_write_entry, uniform_entry};
use crate::backend::wgpu::types::NumericPrecision;
use std::borrow::Cow;

// Shader aliases
const BINARY_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::BINARY_SHADER_F64;
const BINARY_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::BINARY_SHADER_F32;
const UNARY_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::UNARY_SHADER_F64;
const UNARY_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::UNARY_SHADER_F32;
const SCALAR_SHADER_F64: &str = crate::backend::wgpu::shaders::elementwise::SCALAR_SHADER_F64;
const SCALAR_SHADER_F32: &str = crate::backend::wgpu::shaders::elementwise::SCALAR_SHADER_F32;
const TRANSPOSE_SHADER_F64: &str = crate::backend::wgpu::shaders::transpose::TRANSPOSE_SHADER_F64;
const TRANSPOSE_SHADER_F32: &str = crate::backend::wgpu::shaders::transpose::TRANSPOSE_SHADER_F32;
const MATMUL_SHADER_F64: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_F64;
const MATMUL_SHADER_F32: &str = crate::backend::wgpu::shaders::matmul::MATMUL_SHADER_F32;
const REDUCE_GLOBAL_SHADER_F64: &str = crate::backend::wgpu::shaders::reduction::REDUCE_GLOBAL_SHADER_F64;
const REDUCE_GLOBAL_SHADER_F32: &str = crate::backend::wgpu::shaders::reduction::REDUCE_GLOBAL_SHADER_F32;
const REDUCE_DIM_SHADER_F64: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_SHADER_F64;
const REDUCE_DIM_SHADER_F32: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_SHADER_F32;
const REDUCE_DIM_MINMAX_SHADER_F64: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_MINMAX_SHADER_F64;
const REDUCE_DIM_MINMAX_SHADER_F32: &str = crate::backend::wgpu::shaders::reduction::REDUCE_DIM_MINMAX_SHADER_F32;

pub struct PipelineBundle {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

pub struct WgpuPipelines {
    pub binary: PipelineBundle,
    pub unary: PipelineBundle,
    pub scalar: PipelineBundle,
    pub transpose: PipelineBundle,
    pub matmul: PipelineBundle,
    pub reduce_global: PipelineBundle,
    pub reduce_dim_sum_mean: PipelineBundle,
    pub reduce_dim_minmax: PipelineBundle,
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
            "runmat-reduce-dim-ext-layout",
            "runmat-reduce-dim-ext-shader",
            "runmat-reduce-dim-ext-pipeline",
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

        Self {
            binary,
            unary,
            scalar,
            transpose,
            matmul,
            reduce_global,
            reduce_dim_sum_mean,
            reduce_dim_minmax,
        }
    }
}

fn create_pipeline(
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
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&(String::from(pipeline_label) + "-layout")),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        module: &module,
        layout: Some(&pipeline_layout),
        entry_point: "main",
    });
    PipelineBundle { pipeline, layout }
}

pub fn create_pipeline_layout(device: &wgpu::Device, label: &str, bgl: &wgpu::BindGroupLayout) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    })
}

pub fn create_shader_module(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    })
}


