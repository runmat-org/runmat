use super::*;
use futures::FutureExt;

struct HarnessProvider;

fn harness_owner() -> &'static runmat_accelerate::simple_provider::InProcessProvider {
    crate::retained_provider::retained_test_provider!(OWNER: runmat_accelerate::simple_provider::InProcessProvider = runmat_accelerate::simple_provider::InProcessProvider::new())
}

fn store_get(handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
    harness_owner()
        .download(handle)
        .now_or_never()
        .expect("in-process download is synchronous")
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
    harness_owner().upload(&HostTensorView {
        data: &data,
        shape: &shape,
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

    fn device_info(&self) -> String {
        "analysis-harness-provider".to_string()
    }

    fn device_id(&self) -> u32 {
        harness_owner().device_id()
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: self.device_id(),
            name: "analysis-harness-provider".into(),
            vendor: "runmat-tests".to_string(),
            memory_bytes: None,
            backend: Some("harness_gpu".to_string()),
        }
    }

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x + y).map_err(Into::into) })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x - y).map_err(Into::into) })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x * y).map_err(Into::into) })
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        let ah = store_get(a)?;
        let data = ah.data.iter().map(|&x| x * scalar).collect();
        store_insert(ah.shape, data)
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async {
                let ah = store_get(a)?;
                let sum = ah.data.iter().sum::<f64>();
                store_insert(vec![1], vec![sum])
            }
            .await;
            result.map_err(Into::into)
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
        harness_owner().scatter_linear(target, indices, values)
    }
}

#[test]
fn harness_backing_preserves_aliases_and_retires_after_last_physical_lease() {
    let input = store_insert(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let alias = input.clone();
    let physical = input.allocation_lease().downgrade();
    let replacement = store_insert(vec![1, 1], vec![7.0]).unwrap();
    HarnessProvider
        .scatter_linear(&input, &[1], &replacement)
        .unwrap();
    drop(input);
    assert_eq!(store_get(&alias).unwrap().data, vec![1.0, 7.0]);
    assert!(physical.acquire().is_some());
    drop(alias);
    assert!(physical.acquire().is_none());
}

pub(super) fn with_harness_provider<T>(f: impl FnOnce() -> T) -> T {
    static PROVIDER: HarnessProvider = HarnessProvider;
    let _guard = ThreadProviderGuard::set(Some(&PROVIDER));
    f()
}
