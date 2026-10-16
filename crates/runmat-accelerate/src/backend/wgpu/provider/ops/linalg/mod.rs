use super::*;

mod decomposition;
mod host_boundary;
mod matmul;
pub(in crate::backend::wgpu::provider) use matmul::{
    selected_matmul_pipeline, MatmulPipelineRoute,
};
mod pagefun;
mod stats;
mod structure;
