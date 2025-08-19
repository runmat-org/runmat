use anyhow::Result;
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static REGISTRY: OnceCell<Mutex<HashMap<u64, Vec<f64>>>> = OnceCell::new();

fn registry() -> &'static Mutex<HashMap<u64, Vec<f64>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct InProcessProvider {
    next_id: AtomicU64,
}

impl InProcessProvider {
    pub const fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
        }
    }
}

impl AccelProvider for InProcessProvider {
    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = registry().lock().unwrap();
        guard.insert(id, host.data.to_vec());
        Ok(GpuTensorHandle {
            shape: host.shape.to_vec(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn download(&self, h: &GpuTensorHandle) -> Result<HostTensorOwned> {
        let guard = registry().lock().unwrap();
        if let Some(buf) = guard.get(&h.buffer_id) {
            Ok(HostTensorOwned {
                data: buf.clone(),
                shape: h.shape.clone(),
            })
        } else {
            Err(anyhow::anyhow!("buffer not found: {}", h.buffer_id))
        }
    }

    fn free(&self, h: &GpuTensorHandle) -> Result<()> {
        let mut guard = registry().lock().unwrap();
        guard.remove(&h.buffer_id);
        Ok(())
    }

    fn device_info(&self) -> String {
        "in-process provider (host registry)".to_string()
    }

    fn elem_add(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] + bbuf[i];
        }
        drop(guard);
        // Upload new buffer to registry
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_mul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] * bbuf[i];
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_sub(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = abuf[i] - bbuf[i];
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn elem_div(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        if a.shape != b.shape {
            return Err(anyhow::anyhow!("shape mismatch"));
        }
        let mut out = vec![0.0; abuf.len()];
        for i in 0..abuf.len() {
            out[i] = if bbuf[i] == 0.0 {
                f64::INFINITY * abuf[i].signum()
            } else {
                abuf[i] / bbuf[i]
            };
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: a.shape.clone(),
            device_id: 0,
            buffer_id: id,
        })
    }

    fn matmul(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        // Only support 2D shapes for reference provider
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(anyhow::anyhow!("matmul: only 2D supported"));
        }
        let (ar, ac) = (a.shape[0], a.shape[1]);
        let (br, bc) = (b.shape[0], b.shape[1]);
        if ac != br {
            return Err(anyhow::anyhow!("matmul: inner dims must agree"));
        }
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let bbuf = guard
            .get(&b.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", b.buffer_id))?;
        let mut out = vec![0.0; ar * bc];
        // Column-major multiplication
        for j in 0..bc {
            for i in 0..ar {
                let mut sum = 0.0;
                for k in 0..ac {
                    sum += abuf[i + k * ar] * bbuf[k + j * br];
                }
                out[i + j * ar] = sum;
            }
        }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle {
            shape: vec![ar, bc],
            device_id: 0,
            buffer_id: id,
        })
    }
}

static INSTANCE: OnceCell<InProcessProvider> = OnceCell::new();

/// Register the in-process provider as the global acceleration provider.
/// Safe to call multiple times; only the first call installs the provider.
pub fn register_inprocess_provider() {
    let provider: &'static InProcessProvider = INSTANCE.get_or_init(|| InProcessProvider::new());
    // Safety: we intentionally install a reference with 'static lifetime
    unsafe { runmat_accelerate_api::register_provider(provider) };
}
