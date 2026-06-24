use super::*;

struct HarnessProvider;

fn harness_store() -> &'static Mutex<HashMap<u64, HostTensorOwned>> {
    static STORE: OnceLock<Mutex<HashMap<u64, HostTensorOwned>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_get(handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
    let guard = harness_store()
        .lock()
        .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
    guard
        .get(&handle.buffer_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing tensor for buffer_id={}", handle.buffer_id))
}

fn store_insert(shape: Vec<usize>, data: Vec<f64>) -> anyhow::Result<GpuTensorHandle> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(anyhow::anyhow!(
            "tensor data len {} does not match shape product {}",
            data.len(),
            expected
        ));
    }
    static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(5000);
    let buffer_id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
    let mut guard = harness_store()
        .lock()
        .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
    guard.insert(
        buffer_id,
        HostTensorOwned {
            data,
            shape: shape.clone(),
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        },
    );
    Ok(GpuTensorHandle {
        shape,
        device_id: 21,
        buffer_id,
    })
}

fn elementwise_binary(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    op: impl Fn(f64, f64) -> f64,
) -> anyhow::Result<GpuTensorHandle> {
    let ah = store_get(a)?;
    let bh = store_get(b)?;
    if ah.shape != bh.shape {
        return Err(anyhow::anyhow!("shape mismatch in elementwise operation"));
    }
    let data = ah
        .data
        .iter()
        .zip(bh.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    store_insert(ah.shape, data)
}

impl AccelProvider for HarnessProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        store_insert(host.shape.to_vec(), host.data.to_vec())
    }

    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move { store_get(h) })
    }

    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()> {
        let mut guard = harness_store()
            .lock()
            .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
        guard.remove(&h.buffer_id);
        Ok(())
    }

    fn device_info(&self) -> String {
        "analysis-harness-provider".to_string()
    }

    fn device_id(&self) -> u32 {
        21
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 21,
            name: "analysis-harness-provider".to_string(),
            vendor: "runmat-tests".to_string(),
            memory_bytes: None,
            backend: Some("harness_gpu".to_string()),
        }
    }

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x + y) })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x - y) })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x * y) })
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        let ah = store_get(a)?;
        let data = ah.data.iter().map(|&x| x * scalar).collect();
        store_insert(ah.shape, data)
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let ah = store_get(a)?;
            let sum = ah.data.iter().sum::<f64>();
            store_insert(vec![1], vec![sum])
        })
    }

    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> anyhow::Result<f64> {
        let ah = store_get(h)?;
        ah.data
            .get(linear_index)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("read_scalar index out of bounds"))
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let src = store_get(source)?;
        let out_len: usize = output_shape.iter().product();
        if indices.len() != out_len {
            return Err(anyhow::anyhow!(
                "indices len {} does not match output len {}",
                indices.len(),
                out_len
            ));
        }
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            let idx = idx as usize;
            let value = src
                .data
                .get(idx)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("gather index out of bounds"))?;
            out.push(value);
        }
        store_insert(output_shape.to_vec(), out)
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        let values_host = store_get(values)?;
        if values_host.data.len() != indices.len() {
            return Err(anyhow::anyhow!(
                "scatter values len {} != indices len {}",
                values_host.data.len(),
                indices.len()
            ));
        }

        let mut guard = harness_store()
            .lock()
            .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
        let target_host = guard
            .get_mut(&target.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("missing target tensor for scatter"))?;
        for (i, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            if idx >= target_host.data.len() {
                return Err(anyhow::anyhow!("scatter index out of bounds"));
            }
            target_host.data[idx] = values_host.data[i];
        }
        Ok(())
    }
}

pub(super) fn with_harness_provider<T>(f: impl FnOnce() -> T) -> T {
    static PROVIDER: HarnessProvider = HarnessProvider;
    let _guard = ThreadProviderGuard::set(Some(&PROVIDER));
    f()
}
