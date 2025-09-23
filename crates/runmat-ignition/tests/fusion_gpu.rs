use anyhow::anyhow;
use once_cell::sync::OnceCell;
use runmat_accelerate::fusion_residency;
use runmat_accelerate_api::{
    AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned, HostTensorView,
    ProviderPrecision,
};
use runmat_builtins::Value;
use runmat_gc::gc_test_context;
use runmat_hir::lower;
use runmat_ignition::{compile, interpret, Instr};
use runmat_parser::parse;
use runmat_runtime::gather_if_needed;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

struct TestProvider {
    next_id: AtomicU64,
    buffers: Mutex<HashMap<u64, (Vec<f64>, Vec<usize>)>>,
}

impl TestProvider {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
        }
    }

    fn pull(&self, handle: &GpuTensorHandle) -> anyhow::Result<(Vec<f64>, Vec<usize>)> {
        let guard = self.buffers.lock().unwrap();
        guard
            .get(&handle.buffer_id)
            .cloned()
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id))
    }

    fn push(&self, data: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.buffers
            .lock()
            .unwrap()
            .insert(id, (data, shape.clone()));
        GpuTensorHandle {
            shape,
            device_id: 0,
            buffer_id: id,
        }
    }
}

impl AccelProvider for TestProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.push(host.data.to_vec(), host.shape.to_vec()))
    }

    fn download(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        let (data, shape) = self.pull(handle)?;
        Ok(HostTensorOwned { data, shape })
    }

    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.buffers.lock().unwrap().remove(&handle.buffer_id);
        Ok(())
    }

    fn device_info(&self) -> String {
        "test-provider".to_string()
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 0,
            name: "TestProvider".into(),
            vendor: "RunMat".into(),
            memory_bytes: None,
            backend: Some("test".into()),
        }
    }

    fn fused_elementwise(
        &self,
        _shader: &str,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        len: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        if inputs.len() < 2 {
            return Err(anyhow!("expected two inputs"));
        }
        let (x_buf, _) = self.pull(&inputs[0])?;
        let _ = &inputs[1];
        let x_len = x_buf.len().max(1);
        let input_len = inputs
            .get(0)
            .map(|h| h.shape.iter().product::<usize>())
            .unwrap_or(1)
            .max(1);
        let total = output_shape
            .iter()
            .product::<usize>()
            .max(len)
            .max(input_len)
            .max(1);
        let mut out = Vec::with_capacity(total);
        const B_VALUE: f64 = 2.0;
        for idx in 0..total {
            let x_val = x_buf[idx % x_len];
            out.push(x_val.sin() * x_val + B_VALUE);
        }
        let mut shape = if output_shape.iter().product::<usize>() <= 1 && input_len > 1 {
            inputs[0].shape.clone()
        } else if output_shape.is_empty() {
            vec![total]
        } else {
            output_shape.to_vec()
        };
        if shape.iter().product::<usize>() != total {
            shape = vec![total];
        }
        Ok(self.push(out, shape))
    }
}

static PROVIDER: OnceCell<TestProvider> = OnceCell::new();

fn ensure_provider_registered() {
    let provider: &'static TestProvider = PROVIDER.get_or_init(TestProvider::new);
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
}

#[test]
fn fused_elementwise_residency_and_gather() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [1, 2, 3];
        b = 2;
        y = sin(x) .* x + b;
        "#;

        let ast = parse(source).expect("parse");
        let hir = lower(&ast).expect("lower");
        let bytecode = compile(&hir).expect("compile");

        let vars = interpret(&bytecode).expect("interpret");

        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .last()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");

        let handle = match y_value {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        assert!(
            fusion_residency::is_resident(handle),
            "GPU handle should be marked resident before gather"
        );

        let gathered = gather_if_needed(y_value).expect("gather");
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected gathered tensor, got {other:?}"),
        };
        assert!(
            !fusion_residency::is_resident(handle),
            "Residency should be cleared after gather"
        );

        let expected: Vec<f64> = [1.0f64, 2.0, 3.0]
            .iter()
            .map(|x| x.sin() * x + 2.0)
            .collect();
        assert_eq!(tensor.data.len(), expected.len());
        for (actual, expect) in tensor.data.iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}
