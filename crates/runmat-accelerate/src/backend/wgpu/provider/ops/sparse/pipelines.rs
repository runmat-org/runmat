use std::sync::{Arc, Mutex};

use super::spmv::SparseSpmvPipeline;

/// Lazily compiled sparse kernels shared by every logical session on one device.
#[derive(Default)]
pub(crate) struct SparsePipelines {
    pub(super) spmv: Mutex<Option<Arc<SparseSpmvPipeline>>>,
}
