//! Native-F32 arithmetic witness for the inactive staged provider session.
//! This tests the physical hook transcript; it does not mint a selected-MIR
//! equivalence certificate or activate the route.
use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use std::{collections::HashMap, sync::Mutex};

struct NumericProvider {
    owner: TestAllocationOwner,
    probes: Mutex<HashMap<u64, RetirementProbe>>,
    hooks: Mutex<Vec<&'static str>>,
}

impl NumericProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            probes: Mutex::new(HashMap::new()),
            hooks: Mutex::new(Vec::new()),
        }))
    }

    fn make(
        &self,
        shape: Vec<usize>,
        data: Vec<f32>,
        hook: &'static str,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.hooks.lock().unwrap().push(hook);
        let allocated = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(data),
            shape,
            storage: GpuTensorStorage::Real,
        })?;
        self.probes
            .lock()
            .unwrap()
            .insert(allocated.handle.buffer_id(), allocated.probe);
        Ok(allocated.handle)
    }

    fn values(&self, handle: &GpuTensorHandle) -> Vec<f32> {
        let probe = self.probes.lock().unwrap()[&handle.buffer_id()].clone();
        probe
            .with_data(handle, |data| match data {
                HostNumericDataView::F32(data) => data.to_vec(),
                _ => panic!("native F32 owner required"),
            })
            .expect("live numeric payload")
    }

    fn binary(
        &self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        hook: &'static str,
        operation: impl Fn(f32, f32) -> f32,
    ) -> anyhow::Result<GpuTensorHandle> {
        let shape = left.shape().to_vec();
        let left_values = self.values(left);
        let right_values = self.values(right);
        let data = left_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let mut remaining = index;
                let mut offset = 0;
                let mut stride = 1;
                for (axis, extent) in shape.iter().enumerate() {
                    let coordinate = remaining % *extent;
                    remaining /= *extent;
                    let right_extent = right.shape().get(axis).copied().unwrap_or(1);
                    assert!(right_extent == 1 || right_extent == *extent);
                    offset += if right_extent == 1 {
                        0
                    } else {
                        coordinate * stride
                    };
                    stride *= right_extent;
                }
                operation(*value, right_values[offset])
            })
            .collect();
        self.make(shape, data, hook)
    }

    fn reduce(
        &self,
        input: &GpuTensorHandle,
        axes: &[usize],
        hook: &'static str,
    ) -> anyhow::Result<GpuTensorHandle> {
        let input_shape = input.shape();
        let mut shape = input_shape.to_vec();
        for axis in axes {
            shape[*axis] = 1;
        }
        let mut result = vec![0.0f32; shape.iter().product()];
        let data = self.values(input);
        for (index, value) in data.iter().enumerate() {
            let mut remaining = index;
            let mut target = 0;
            let mut stride = 1;
            for (axis, extent) in input_shape.iter().enumerate() {
                let coordinate = remaining % *extent;
                remaining /= *extent;
                target += if shape[axis] == 1 {
                    0
                } else {
                    coordinate * stride
                };
                stride *= shape[axis];
            }
            result[target] += *value;
        }
        let count = axes
            .iter()
            .map(|axis| input_shape[*axis])
            .product::<usize>() as f32;
        for value in &mut result {
            *value /= count;
        }
        self.make(shape, result, hook)
    }
}

impl AccelProvider for NumericProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
    fn device_info(&self) -> String {
        "native-f32-image-fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("legacy f64 upload must not be used")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let HostNumericDataView::F32(data) = host.data else {
            panic!("native F32 control required")
        };
        Ok(self.make(host.shape.to_vec(), data.to_vec(), "upload_f32")?)
    }
    fn fill_like(
        &self,
        input: &GpuTensorHandle,
        control: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = (|| {
            anyhow::ensure!(
                input.device_id() == self.device_id()
                    && input.physical().element_type() == NumericElementType::F32
                    && input.physical().storage() == GpuTensorStorage::Real,
                "native F32 real prototype required"
            );
            let elements = self.values(input).len(); // Checks the retained payload and live lease.
            let single = control as f32;
            anyhow::ensure!(
                f64::from(single).to_bits() == control.to_bits(),
                "fixture requires exactly representable Single fill control"
            );
            self.make(input.shape().to_vec(), vec![single; elements], "fill_like")
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("staged session must not gather")
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axis: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> =
                async move { self.reduce(input, &[axis], "mean_dim") }.await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn reduce_mean_nd<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        axes: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> =
                async move { self.reduce(input, axes, "mean_nd") }.await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.binary(a, b, "sub", |x, y| x - y).map_err(Into::into) })
    }
    fn elem_div<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.binary(a, b, "div", |x, y| x / y).map_err(Into::into) })
    }
    fn elem_pow<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary(a, b, "pow", |x, y| x.powf(y))
                .map_err(Into::into)
        })
    }
    fn scalar_add(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        self.make(
            input.shape().to_vec(),
            self.values(input)
                .into_iter()
                .map(|x| x + scalar as f32)
                .collect(),
            "add",
        )
    }
    fn unary_sqrt<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.make(
                input.shape().to_vec(),
                self.values(input).into_iter().map(f32::sqrt).collect(),
                "sqrt",
            )
        })
    }
}

#[test]
fn nested_source_hooks_publish_numerical_f32_mean_sigma_then_powered_image() {
    let provider = NumericProvider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .make(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0], "input")
        .unwrap();
    let mut source = super::tests::source_recipe();
    for plan in [&mut source.mu, &mut source.variance] {
        plan.calls[0].shape = [1, 1, 2];
        plan.calls[1].shape = [1, 1, 1];
    }
    source.final_power.represented_exponent = 2.0;
    let mut session = ImageNormalizeStageSession::begin(
        provider,
        &input,
        ImageNormalizeDescriptor {
            batch: 1,
            height: 2,
            width: 2,
            epsilon: 0.25,
            gain: None,
            bias: None,
            gamma: Some(2.0),
            clamp_zero: false,
        },
        source,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .expect("native F32 structural subset");
    let (mean, sigma, image) = futures::executor::block_on(async {
        let mean = session.mean().await.unwrap().0;
        let sigma = session.sigma().await.unwrap().0;
        let image = session.output().await.unwrap().0;
        (mean, sigma, image)
    });
    assert_eq!(provider.values(&mean), vec![2.5]);
    let expected_sigma = (1.25f32 + 0.25).sqrt();
    assert_eq!(provider.values(&sigma), vec![expected_sigma]);
    let expected = [1.0f32, 2.0, 3.0, 4.0].map(|value| ((value - 2.5) / expected_sigma).powf(2.0));
    assert_eq!(provider.values(&image), expected);
    assert_eq!(
        provider.hooks.lock().unwrap().as_slice(),
        [
            "input",
            "mean_dim",
            "mean_dim",
            "sub",
            "fill_like",
            "pow",
            "mean_dim",
            "mean_dim",
            "add",
            "sqrt",
            "sub",
            "div",
            "fill_like",
            "pow",
        ]
    );
    assert!(!mean.shares_allocation_with(&sigma));
    assert!(!sigma.shares_allocation_with(&image));
    assert!(input.allocation_lease().validate_live().is_ok());
}
