//! Real owned host-backed fixture allocations exercise solver lifetime edges.
use super::*;
use runmat_accelerate_api::allocation::AllocationIdentity;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Mode {
    Success,
    ReductionFailure,
    ReadbackFailure,
    Alias,
}

struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    records: Mutex<Vec<(AllocationIdentity, RetirementProbe)>>,
}

impl Provider {
    fn new(mode: Mode) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            mode,
            records: Mutex::new(Vec::new()),
        }
    }

    fn tensor(&self, data: Vec<f64>) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                shape: vec![data.len()],
                data: HostNumericDataOwned::F64(data),
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.allocation_identity(), allocation.probe));
        allocation.handle
    }

    fn values(&self, handle: &GpuTensorHandle) -> Vec<f64> {
        let probe = self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|(identity, _)| *identity == handle.allocation_identity())
            .unwrap()
            .1
            .clone();
        probe
            .with_data(handle, |data| match data {
                HostNumericDataView::F64(data) => data.to_vec(),
                _ => panic!("fixture uses Double storage"),
            })
            .unwrap()
    }

    fn assert_retirements(&self, expected: &[usize]) {
        let actual: Vec<_> = self
            .records
            .lock()
            .unwrap()
            .iter()
            .map(|(_, probe)| probe.retirements())
            .collect();
        assert_eq!(actual, expected);
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "fea-owning-lifetime-fixture".into()
    }
    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.tensor(host.data.to_vec()))
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("injected readback failure") })
    }
    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.mode, Mode::Alias) {
                assert_eq!(self.values(b), vec![1.0]);
                return Ok(a.clone());
            }
            let data = self
                .values(a)
                .into_iter()
                .zip(self.values(b))
                .map(|(a, b)| a * b)
                .collect();
            Ok(self.tensor(data))
        })
    }
    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let data = self
                .values(a)
                .into_iter()
                .zip(self.values(b))
                .map(|(a, b)| a + b)
                .collect();
            Ok(self.tensor(data))
        })
    }
    fn reduce_sum<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = match self.mode {
                Mode::ReductionFailure => Err(anyhow::anyhow!("injected reduction failure")),
                Mode::Alias => {
                    assert_eq!(input.shape(), &[1]);
                    Ok(input.clone())
                }
                _ => Ok(self.tensor(vec![self.values(input).into_iter().sum()])),
            };
            result.map_err(Into::into)
        })
    }
    fn read_scalar(&self, input: &GpuTensorHandle, index: usize) -> anyhow::Result<f64> {
        anyhow::ensure!(
            !matches!(self.mode, Mode::ReadbackFailure),
            "injected scalar failure"
        );
        Ok(self.values(input)[index])
    }
    fn scatter_linear(
        &self,
        _: &GpuTensorHandle,
        _: &[u32],
        _: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        anyhow::bail!("injected workspace scatter failure before mutation")
    }
}

#[test]
fn dot_retires_owned_intermediates_on_success_and_each_failure() {
    for mode in [Mode::Success, Mode::ReductionFailure, Mode::ReadbackFailure] {
        let provider = Provider::new(mode);
        let a = provider.tensor(vec![1.0, 2.0, 3.0]);
        let b = provider.tensor(vec![4.0, 5.0, 6.0]);
        let mut syncs = 0;
        let result = dot_handle(&provider, &a, &b, &mut syncs);
        assert_eq!(
            result,
            if matches!(mode, Mode::Success) {
                Some(32.0)
            } else {
                None
            }
        );
        let intermediates = if matches!(mode, Mode::ReductionFailure) {
            1
        } else {
            2
        };
        let mut expected = vec![0, 0];
        expected.extend(vec![1; intermediates]);
        provider.assert_retirements(&expected);
        assert_eq!(syncs, u32::from(matches!(mode, Mode::ReadbackFailure)));
        drop((a, b));
        provider.assert_retirements(&vec![1; 2 + intermediates]);
    }
}

#[test]
fn identity_provider_outputs_never_retire_borrowed_dot_operands() {
    let provider = Provider::new(Mode::Alias);
    let a = provider.tensor(vec![6.0]);
    let b = provider.tensor(vec![1.0]);
    assert_eq!(dot_handle(&provider, &a, &b, &mut 0), Some(6.0));
    provider.assert_retirements(&[0, 0]);
    assert_eq!(provider.values(&a), vec![6.0]);
    drop((a, b));
    provider.assert_retirements(&[1, 1]);
}

#[test]
fn workspace_owns_cached_slot_after_failure_until_last_real_alias_drops() {
    let provider = Provider::new(Mode::Success);
    let zero = provider.tensor(vec![0.0]);
    let r = provider.tensor(vec![2.0]);
    let inv = provider.tensor(vec![0.5]);
    let ctx = PreconditionerDeviceContext {
        provider: &provider,
        inv_diag: &inv,
        ilu_l_subdiag: &zero,
        ilu_upper_superdiag: &zero,
        ilu_inv_u_diag: &inv,
        constrained_mask: &zero,
        unconstrained_mask: &zero,
        prev_indices: &[0],
        next_indices: &[0],
        shape: &[1],
        zero_like: &zero,
    };
    let mut workspace = RuntimeTensorWorkspace::new(1).unwrap();
    assert!(
        apply_preconditioner_device(&ctx, SpdPreconditionerKind::Jacobi, &r, &mut workspace)
            .is_none()
    );
    provider.assert_retirements(&[0, 0, 0, 1, 0]);
    let retained = workspace.precond_z.as_ref().unwrap().clone();
    drop(workspace);
    provider.assert_retirements(&[0, 0, 0, 1, 0]);
    drop(retained);
    provider.assert_retirements(&[0, 0, 0, 1, 1]);
    drop((zero, r, inv));
    provider.assert_retirements(&[1, 1, 1, 1, 1]);
}
