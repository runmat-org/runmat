//! Actual staged route: malformed reshape cannot authorize a subsequent hook.
use super::*;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum Fault {
    Class,
    Shape,
    Owner,
    Failure,
}
struct Provider {
    inner: InProcessProvider,
    foreign: InProcessProvider,
    fault: Fault,
    at: usize,
    reshapes: AtomicUsize,
    tiles: AtomicUsize,
    produced: Mutex<Vec<allocation::WeakAllocationLease>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "meshgrid stage admission".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("invalid reshape must not gather") })
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let call = self.reshapes.fetch_add(1, Ordering::SeqCst) + 1;
        if call != self.at {
            return self.inner.reshape(input, shape);
        }
        let output = match self.fault {
            Fault::Failure => return Err(std::io::Error::other("axis preparation failed").into()),
            Fault::Class => self
                .inner
                .upload_numeric_raw(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[1., 2.]),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
            Fault::Shape => self.inner.reshape(input, &[2, 1])?,
            Fault::Owner => self.foreign.upload(&HostTensorView {
                data: &[1., 2.],
                shape,
            })?,
        };
        if !output.shares_allocation_with(input) {
            self.produced
                .lock()
                .unwrap()
                .push(output.allocation_lease().downgrade());
        }
        Ok(output)
    }
    fn repmat(
        &self,
        input: &GpuTensorHandle,
        reps: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.tiles.fetch_add(1, Ordering::SeqCst);
        let output = self.inner.repmat(input, reps)?;
        if let Some(output) = &output {
            self.produced
                .lock()
                .unwrap()
                .push(output.allocation_lease().downgrade());
        }
        Ok(output)
    }
}

#[test]
fn reshape_stages_validate_original_axes_before_next_hook_and_keep_failure_phase() {
    for at in [1, 5] {
        for fault in [Fault::Class, Fault::Shape, Fault::Owner, Fault::Failure] {
            let provider = Box::leak(Box::new(Provider {
                inner: InProcessProvider::new(),
                foreign: InProcessProvider::new(),
                fault,
                at,
                reshapes: AtomicUsize::new(0),
                tiles: AtomicUsize::new(0),
                produced: Mutex::new(vec![]),
            }));
            let _active = ThreadProviderGuard::set(Some(provider));
            let axes = (0..3)
                .map(|_| {
                    let handle = provider
                        .inner
                        .upload(&HostTensorView {
                            data: &[1., 2.],
                            shape: &[1, 2],
                        })
                        .unwrap();
                    AxisData {
                        len: 2,
                        is_complex: false,
                        storage: AxisStorage::GpuReal(handle),
                    }
                })
                .collect::<Vec<_>>();
            let error = try_meshgrid_gpu_from_vector_axes(&axes[0], &axes[1], Some(&axes[2]))
                .err()
                .expect("invalid stage");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(provider.reshapes.load(Ordering::SeqCst), at);
            assert_eq!(
                provider.tiles.load(Ordering::SeqCst),
                if at == 1 { 0 } else { 2 }
            );
            if matches!(fault, Fault::Failure) {
                let mut cause: &(dyn std::error::Error + 'static) =
                    error.source.as_deref().unwrap();
                let mut phase = None;
                loop {
                    if let Some(error) = cause.downcast_ref::<ProviderExecutionError>() {
                        phase = Some(error.is_after_effect());
                    }
                    if cause.is::<std::io::Error>() {
                        break;
                    }
                    cause = cause.source().expect("original operational cause");
                }
                assert_eq!(phase, Some(at == 5));
            }
            assert!(provider
                .produced
                .lock()
                .unwrap()
                .iter()
                .all(|lease| lease.acquire().is_none()));
            for axis in &axes {
                axis.gpu_real()
                    .unwrap()
                    .allocation_lease()
                    .validate_live()
                    .unwrap();
            }
        }
    }

    let provider = Box::leak(Box::new(Provider {
        inner: InProcessProvider::new(),
        foreign: InProcessProvider::new(),
        fault: Fault::Failure,
        at: usize::MAX,
        reshapes: AtomicUsize::new(0),
        tiles: AtomicUsize::new(0),
        produced: Mutex::new(vec![]),
    }));
    let _active = ThreadProviderGuard::set(Some(provider));
    let axes = (0..2)
        .map(|_| AxisData {
            len: 2,
            is_complex: false,
            storage: AxisStorage::GpuReal(
                provider
                    .inner
                    .upload(&HostTensorView {
                        data: &[1., 2.],
                        shape: &[1, 2],
                    })
                    .expect("axis"),
            ),
        })
        .collect::<Vec<_>>();
    let grids = try_meshgrid_gpu_from_vector_axes(&axes[0], &axes[1], None)
        .expect("selected provider grid")
        .expect("resident grids");
    let empty = futures::executor::block_on(
        MeshgridEval {
            outputs: grids,
            target_class: PrototypeClass::Real,
            target_residency: DevicePreference::Gpu(provider.device_id()),
        }
        .into_output_list(0),
    )
    .expect("zero-output publication retires unrequested grids");
    assert!(matches!(empty, Value::OutputList(values) if values.as_slice().is_empty()));
    assert!(provider
        .produced
        .lock()
        .unwrap()
        .iter()
        .all(|lease| lease.acquire().is_none()));
    for axis in &axes {
        axis.gpu_real()
            .expect("borrowed axis")
            .allocation_lease()
            .validate_live()
            .expect("borrowed input remains live");
    }
}
