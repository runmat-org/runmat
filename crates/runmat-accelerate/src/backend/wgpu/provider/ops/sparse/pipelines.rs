use std::sync::{Arc, Mutex};

use super::{
    cg::pipelines::SparseCgPipelines,
    jacobi_kernels::{SparseJacobiApplyPipeline, SparseJacobiPreparePipeline},
    spmv::SparseSpmvPipeline,
};

/// Lazily compiled sparse kernels shared by every logical session on one device.
#[derive(Default)]
pub(crate) struct SparsePipelines {
    pub(super) spmv: Mutex<Option<Arc<SparseSpmvPipeline>>>,
    pub(super) jacobi_prepare: Mutex<Option<Arc<SparseJacobiPreparePipeline>>>,
    pub(super) jacobi_apply: Mutex<Option<Arc<SparseJacobiApplyPipeline>>>,
    pub(super) cg: Mutex<Option<Arc<SparseCgPipelines>>>,
}
