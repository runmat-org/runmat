use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct PipelineRegistry {
    inner: Mutex<HashMap<u64, Arc<wgpu::ComputePipeline>>>,
}

impl PipelineRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &u64) -> Option<Arc<wgpu::ComputePipeline>> {
        self.inner.lock().ok().and_then(|g| g.get(key).cloned())
    }

    pub fn insert(&self, key: u64, pipeline: Arc<wgpu::ComputePipeline>) {
        if let Ok(mut g) = self.inner.lock() {
            g.insert(key, pipeline);
        }
    }
}
