use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned, HostTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_value::Value;
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[derive(Clone, Copy, Debug)]
enum Fault {
    PendingReduction,
    PendingDownload,
    FailedDownload,
    Alias,
    WrongStorage,
    PendingUnary,
    MalformedUnary,
}

struct Record {
    id: u64,
    probe: RetirementProbe,
}

struct State {
    // Probes retain the actual fixture owner, not an owning tensor lease.
    // Recording a returned handle here would keep the temporary alive.
    records: Vec<Record>,
    calls: usize,
}

struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    state: Mutex<State>,
}

impl Provider {
    fn allocate(
        &self,
        data: Vec<f64>,
        shape: Vec<usize>,
        storage: GpuTensorStorage,
    ) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(data),
            shape,
            storage,
        })?;
        self.state.lock().unwrap().records.push(Record {
            id: allocation.handle.buffer_id(),
            probe: allocation.probe,
        });
        Ok(allocation.handle)
    }

    fn read(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        let probe = self
            .state
            .lock()
            .unwrap()
            .records
            .iter()
            .find(|record| record.id == handle.buffer_id())
            .ok_or_else(|| anyhow::anyhow!("fixture received unknown allocation"))?
            .probe
            .clone();
        let data = probe.with_data(handle, |data| match data {
            HostNumericDataView::F64(data) => data.to_vec(),
            _ => panic!("this fixture allocates actual F64 storage only"),
        })?;
        Ok(HostTensorOwned {
            data,
            shape: handle.shape().to_vec(),
            storage: runmat_accelerate_api::handle_storage(handle),
        })
    }

    fn unary<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.fault, Fault::PendingUnary) {
                return std::future::pending().await;
            }
            // A lawful allocation with the wrong operation result shape:
            // immutable descriptors cannot be forged to omit storage.
            self.allocate(vec![0.0], vec![1, 1], GpuTensorStorage::Real)
                .inspect(|output| {
                    assert_ne!(output.shape(), input.shape());
                })
        })
    }
}

impl AccelProvider for Provider {
    fn unary_log<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary(input)
    }
    fn unary_log10<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary(input)
    }
    fn unary_log1p<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary(input)
    }
    fn unary_log2<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary(input)
    }
    fn unary_sqrt<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.unary(input)
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owning log/root lifetime fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.allocate(
            view.data.to_vec(),
            view.shape.to_vec(),
            GpuTensorStorage::Real,
        )
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.state.lock().unwrap().calls += 1;
            match self.fault {
                Fault::PendingReduction => return std::future::pending().await,
                Fault::Alias => return input.with_view(input.view().clone()).map_err(Into::into),
                Fault::WrongStorage => {
                    return self
                        .allocate(
                            vec![1.0, 0.0],
                            vec![1, 1],
                            GpuTensorStorage::ComplexInterleaved,
                        )
                        .map_err(Into::into)
                }
                _ => {}
            }
            let minimum = self.read(input)?.data.into_iter().reduce(f64::min).unwrap();
            self.allocate(vec![minimum], vec![1, 1], GpuTensorStorage::Real)
                .map_err(Into::into)
        })
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            match self.fault {
                Fault::PendingDownload => std::future::pending().await,
                Fault::FailedDownload => anyhow::bail!("injected probe download failure"),
                _ => self.read(input),
            }
        })
    }
}

fn fixture(fault: Fault) -> (&'static Provider, GpuTensorHandle) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        fault,
        state: Mutex::new(State {
            records: Vec::new(),
            calls: 0,
        }),
    }));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    (provider, input)
}

// One runtime library supplies both dispatch inventory and implementation.
#[test]
fn logarithm_root_and_gammaln_probe_lifetimes_through_dispatch() {
    for name in [
        "log", "log10", "log1p", "log2", "sqrt", "realsqrt", "gammaln",
    ] {
        for fault in [
            Fault::PendingReduction,
            Fault::PendingDownload,
            Fault::FailedDownload,
            Fault::Alias,
            Fault::WrongStorage,
            Fault::PendingUnary,
            Fault::MalformedUnary,
        ] {
            if name == "gammaln" && matches!(fault, Fault::PendingUnary | Fault::MalformedUnary) {
                continue; // Probe coverage only for this identity.
            }
            let (provider, input) = fixture(fault);
            let _provider = ThreadProviderGuard::set(Some(provider));
            let args = [Value::GpuTensor(input.clone())];
            if name != "gammaln" {
                let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    name, &args, 3,
                ))
                .unwrap_err();
                assert!(error.message().contains("output"), "{name}: {error}");
                assert_eq!(provider.state.lock().unwrap().calls, 0);
            }
            drop(runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            assert_eq!(provider.state.lock().unwrap().calls, 0);
            let mut call = Box::pin(runmat_runtime::dispatcher::call_builtin_async_with_outputs(
                name, &args, 1,
            ));
            let result = call.as_mut().poll(&mut Context::from_waker(&noop_waker()));
            if matches!(
                fault,
                Fault::PendingReduction | Fault::PendingDownload | Fault::PendingUnary
            ) {
                assert!(
                    matches!(result, Poll::Pending),
                    "{name}/{fault:?}: {result:?}"
                );
            } else {
                assert!(
                    matches!(result, Poll::Ready(Err(_))),
                    "{name}/{fault:?}: {result:?}"
                );
            }
            drop(call);
            drop(result);
            assert!(!runmat_accelerate_api::handle_is_logical(&input), "{name}");
            let state = provider.state.lock().unwrap();
            assert_eq!(state.calls, 1, "{name}: no retry after provider failure");
            assert_eq!(
                state.records[0].probe.retirements(),
                0,
                "{name}: borrowed input"
            );
            let expected_temporaries = match fault {
                Fault::PendingReduction | Fault::Alias => 0,
                Fault::MalformedUnary => 2,
                _ => 1,
            };
            assert_eq!(
                state.records.len() - 1,
                expected_temporaries,
                "{name}/{fault:?}"
            );
            for record in &state.records[1..] {
                assert_eq!(
                    record.probe.retirements(),
                    1,
                    "{name}/{fault:?}: temporary {}",
                    record.id
                );
            }
            drop(state);
            assert_eq!(provider.read(&input).unwrap().data, [1.0, 4.0]);
            drop(args);
            drop(input);
            assert_eq!(
                provider.state.lock().unwrap().records[0]
                    .probe
                    .retirements(),
                1,
                "{name}: final input owner"
            );
        }
    }
}
