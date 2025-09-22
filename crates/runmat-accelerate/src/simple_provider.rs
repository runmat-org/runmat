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

impl Default for InProcessProvider {
    fn default() -> Self {
        Self::new()
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

    fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
        runmat_accelerate_api::ApiDeviceInfo {
            device_id: 0,
            name: "InProcess".to_string(),
            vendor: "RunMat".to_string(),
            memory_bytes: None,
            backend: Some("inprocess".to_string()),
        }
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

    fn unary_sin(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.sin()).collect();
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

    fn unary_cos(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.cos()).collect();
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

    fn unary_abs(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.abs()).collect();
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

    fn unary_exp(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.exp()).collect();
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

    fn unary_log(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.ln()).collect();
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

    fn unary_sqrt(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x.sqrt()).collect();
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

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x + scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x - scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| x * scalar).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = if scalar == 0.0 {
            abuf.iter().map(|&x| f64::INFINITY * x.signum()).collect()
        } else {
            abuf.iter().map(|&x| x / scalar).collect()
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        // compute scalar - a
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf.iter().map(|&x| scalar - x).collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        // compute scalar ./ a
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out: Vec<f64> = abuf
            .iter()
            .map(|&x| if x == 0.0 { f64::INFINITY * scalar.signum() } else { scalar / x })
            .collect();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: a.shape.clone(), device_id: 0, buffer_id: id })
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 { return Err(anyhow::anyhow!("transpose: only 2D supported")); }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let mut out = vec![0.0; abuf.len()];
        for i in 0..rows { for j in 0..cols { out[j * rows + i] = abuf[i + j * rows]; } }
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        Ok(GpuTensorHandle { shape: vec![cols, rows], device_id: 0, buffer_id: id })
    }

    fn reduce_sum(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let s: f64 = abuf.iter().sum();
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, vec![s]);
        Ok(GpuTensorHandle { shape: vec![1,1], device_id: 0, buffer_id: id })
    }

    fn reduce_sum_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 { return Err(anyhow::anyhow!("reduce_sum_dim: only 2D supported")); }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard
            .get(&a.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out = if dim <= 1 {
            // sum over rows -> 1 x cols
            let mut v = vec![0.0f64; cols];
            for c in 0..cols { let mut s = 0.0; for r in 0..rows { s += abuf[r + c * rows]; } v[c] = s; }
            v
        } else {
            // sum over cols -> rows x 1
            let mut v = vec![0.0f64; rows];
            for r in 0..rows { let mut s = 0.0; for c in 0..cols { s += abuf[r + c * rows]; } v[r] = s; }
            v
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard2 = registry().lock().unwrap();
        guard2.insert(id, out);
        let shape = if dim <= 1 { vec![1, cols] } else { vec![rows, 1] };
        Ok(GpuTensorHandle { shape, device_id: 0, buffer_id: id })
    }

    fn reduce_mean(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let mean = if abuf.is_empty() { 0.0 } else { abuf.iter().sum::<f64>() / (abuf.len() as f64) };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![mean]);
        Ok(GpuTensorHandle { shape: vec![1,1], device_id: 0, buffer_id: id })
    }

    fn reduce_mean_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<GpuTensorHandle> {
        if a.shape.len() != 2 { return Err(anyhow::anyhow!("reduce_mean_dim: only 2D supported")); }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let out = if dim <= 1 {
            let mut v = vec![0.0f64; cols];
            for c in 0..cols { let mut s = 0.0; for r in 0..rows { s += abuf[r + c * rows]; } v[c] = s / (rows as f64); }
            v
        } else {
            let mut v = vec![0.0f64; rows];
            for r in 0..rows { let mut s = 0.0; for c in 0..cols { s += abuf[r + c * rows]; } v[r] = s / (cols as f64); }
            v
        };
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, out);
        let shape = if dim <= 1 { vec![1, cols] } else { vec![rows, 1] };
        Ok(GpuTensorHandle { shape, device_id: 0, buffer_id: id })
    }

    fn reduce_min(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let m = abuf.iter().cloned().fold(f64::INFINITY, f64::min);
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![m]);
        Ok(GpuTensorHandle { shape: vec![1,1], device_id: 0, buffer_id: id })
    }

    fn reduce_min_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<runmat_accelerate_api::ReduceDimResult> {
        if a.shape.len() != 2 { return Err(anyhow::anyhow!("reduce_min_dim: only 2D supported")); }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let (vals, inds, vshape) = if dim <= 1 {
            let mut m: Vec<f64> = vec![f64::INFINITY; cols];
            let mut idx: Vec<f64> = vec![1.0; cols];
            for c in 0..cols { for r in 0..rows { let v = abuf[r + c * rows]; if v < m[c] { m[c]=v; idx[c]=(r+1) as f64; } } }
            (m, idx, vec![1, cols])
        } else {
            let mut m: Vec<f64> = vec![f64::INFINITY; rows];
            let mut idx: Vec<f64> = vec![1.0; rows];
            for r in 0..rows { for c in 0..cols { let v = abuf[r + c * rows]; if v < m[r] { m[r]=v; idx[r]=(c+1) as f64; } } }
            (m, idx, vec![rows, 1])
        };
        drop(guard);
        let idv = self.next_id.fetch_add(1, Ordering::Relaxed);
        let idi = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut g = registry().lock().unwrap();
        g.insert(idv, vals);
        g.insert(idi, inds);
        let shape_vals = vshape.clone();
        let shape_inds = vshape;
        Ok(runmat_accelerate_api::ReduceDimResult {
            values: GpuTensorHandle { shape: shape_vals, device_id: 0, buffer_id: idv },
            indices: GpuTensorHandle { shape: shape_inds, device_id: 0, buffer_id: idi },
        })
    }

    fn reduce_max(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let m = abuf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        drop(guard);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, vec![m]);
        Ok(GpuTensorHandle { shape: vec![1,1], device_id: 0, buffer_id: id })
    }

    fn reduce_max_dim(&self, a: &GpuTensorHandle, dim: usize) -> Result<runmat_accelerate_api::ReduceDimResult> {
        if a.shape.len() != 2 { return Err(anyhow::anyhow!("reduce_max_dim: only 2D supported")); }
        let rows = a.shape[0];
        let cols = a.shape[1];
        let guard = registry().lock().unwrap();
        let abuf = guard.get(&a.buffer_id).ok_or_else(|| anyhow::anyhow!("buffer not found: {}", a.buffer_id))?;
        let (vals, inds, vshape) = if dim <= 1 {
            let mut m: Vec<f64> = vec![f64::NEG_INFINITY; cols];
            let mut idx: Vec<f64> = vec![1.0; cols];
            for c in 0..cols { for r in 0..rows { let v = abuf[r + c * rows]; if v > m[c] { m[c]=v; idx[c]=(r+1) as f64; } } }
            (m, idx, vec![1, cols])
        } else {
            let mut m: Vec<f64> = vec![f64::NEG_INFINITY; rows];
            let mut idx: Vec<f64> = vec![1.0; rows];
            for r in 0..rows { for c in 0..cols { let v = abuf[r + c * rows]; if v > m[r] { m[r]=v; idx[r]=(c+1) as f64; } } }
            (m, idx, vec![rows, 1])
        };
        drop(guard);
        let idv = self.next_id.fetch_add(1, Ordering::Relaxed);
        let idi = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut g = registry().lock().unwrap();
        g.insert(idv, vals);
        g.insert(idi, inds);
        let shape_vals = vshape.clone();
        let shape_inds = vshape;
        Ok(runmat_accelerate_api::ReduceDimResult {
            values: GpuTensorHandle { shape: shape_vals, device_id: 0, buffer_id: idv },
            indices: GpuTensorHandle { shape: shape_inds, device_id: 0, buffer_id: idi },
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
    let provider: &'static InProcessProvider = INSTANCE.get_or_init(InProcessProvider::new);
    // Safety: we intentionally install a reference with 'static lifetime
    unsafe { runmat_accelerate_api::register_provider(provider) };
}
