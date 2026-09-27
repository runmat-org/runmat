use super::*;
use crate::builtins::common::test_support;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{Mutex, OnceLock},
    task::{Context, Poll},
};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    LaterFailure,
    LaterPending,
    BorrowedAlias,
    PriorResultAlias,
}
#[derive(Default)]
struct State {
    fault: Fault,
    second: Option<GpuTensorHandle>,
    outputs: Vec<GpuTensorHandle>,
    freed: Vec<u64>,
    downloads: usize,
}
struct Provider {
    inner: InProcessProvider,
    state: Mutex<State>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, value: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(value)
    }
    fn upload_numeric_raw(&self, value: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        let mut state = self.state.lock().unwrap();
        match state.fault {
            Fault::BorrowedAlias => return Ok(state.second.as_ref().unwrap().clone()),
            Fault::PriorResultAlias if !state.outputs.is_empty() => {
                return Ok(state.outputs[0].clone())
            }
            _ => {}
        }
        let output = self.inner.upload_numeric(value)?;
        state.outputs.push(output.clone());
        Ok(output)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let fault = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                if state
                    .second
                    .as_ref()
                    .is_some_and(|second| gpu_helpers::same_gpu_handle(second, handle))
                {
                    state.fault
                } else {
                    Fault::None
                }
            };
            match fault {
                Fault::LaterFailure => anyhow::bail!("injected later column failure"),
                Fault::LaterPending => std::future::pending().await,
                _ => self.inner.download(handle).await,
            }
        })
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.state.lock().unwrap().freed.push(handle.buffer_id);
        self.inner.free(handle)
    }
}

fn fixture(fault: Fault) -> (&'static Provider, Vec<GpuTensorHandle>, Value) {
    static PROVIDER: OnceLock<Provider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| Provider {
        inner: InProcessProvider::new(),
        state: Mutex::default(),
    });
    let handles: Vec<_> = [[1.0, -1.0], [0.0, 1.0]]
        .iter()
        .map(|data| {
            provider
                .inner
                .upload(&HostTensorView {
                    data,
                    shape: &[2, 1],
                })
                .unwrap()
        })
        .collect();
    *provider.state.lock().unwrap() = State {
        fault,
        second: Some(handles[1].clone()),
        ..State::default()
    };
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        handles.iter().cloned().map(Value::GpuTensor).collect(),
    )
    .unwrap();
    (provider, handles, table)
}

#[test]
fn atan2_tabular_later_failure_drop_and_aliases_release_only_owned_columns() {
    let _serial = test_support::accel_test_lock();
    for fault in [
        Fault::LaterFailure,
        Fault::LaterPending,
        Fault::BorrowedAlias,
        Fault::PriorResultAlias,
    ] {
        let (provider, borrowed, table) = fixture(fault);
        let selected = ThreadProviderGuard::set(Some(provider));
        let inputs = [table, Value::Num(1.0)];
        let mut call = Box::pin(crate::dispatcher::call_builtin_async("atan2", &inputs));
        if matches!(fault, Fault::LaterPending) {
            assert!(matches!(
                call.as_mut().poll(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            assert_eq!(provider.state.lock().unwrap().outputs.len(), 1);
            drop(selected);
            drop(call);
        } else {
            assert!(block_on(call).is_err());
            drop(selected);
        }
        let state = provider.state.lock().unwrap();
        assert_eq!(
            state.outputs.len(),
            usize::from(!matches!(fault, Fault::BorrowedAlias))
        );
        for output in &state.outputs {
            assert_eq!(
                state
                    .freed
                    .iter()
                    .filter(|&&id| id == output.buffer_id)
                    .count(),
                1
            );
        }
        for handle in &borrowed {
            assert!(!state.freed.contains(&handle.buffer_id));
        }
        drop(state);
        for handle in borrowed {
            assert!(block_on(provider.inner.download(&handle)).is_ok());
            provider.inner.free(&handle).unwrap();
        }
    }
}

#[test]
fn atan2_tabular_admits_every_column_before_first_download() {
    let _serial = test_support::accel_test_lock();
    let (provider, borrowed, _) = fixture(Fault::None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::GpuTensor(borrowed[0].clone()),
            Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(0.0, 1.0); 2], vec![2, 1]).unwrap(),
            ),
        ],
    )
    .unwrap();
    assert!(block_on(crate::dispatcher::call_builtin_async(
        "atan2",
        &[table, Value::Num(1.0)]
    ))
    .is_err());
    let state = provider.state.lock().unwrap();
    assert_eq!(state.downloads, 0);
    assert!(state.outputs.is_empty());
    assert!(state.freed.is_empty());
    drop(state);
    for handle in borrowed {
        provider.inner.free(&handle).unwrap();
    }
}

#[test]
fn atan2_tabular_success_publishes_owned_columns_with_numeric_oracle() {
    let _serial = test_support::accel_test_lock();
    let (provider, borrowed, table) = fixture(Fault::None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let output = block_on(crate::dispatcher::call_builtin_async(
        "atan2",
        &[table, Value::Num(1.0)],
    ))
    .unwrap();
    let Value::Object(output) = output else {
        panic!("table result")
    };
    let columns = crate::builtins::table::table_variables(&output).unwrap();
    for (name, expected) in [
        ("A", [1.0_f64.atan(), (-1.0_f64).atan()]),
        ("B", [0.0, 1.0_f64.atan()]),
    ] {
        let Value::GpuTensor(handle) = &columns.fields[name] else {
            panic!("resident column")
        };
        let actual = block_on(provider.inner.download(handle)).unwrap();
        assert_eq!(actual.data, expected);
    }
    assert!(provider.state.lock().unwrap().freed.is_empty());
    for output in provider.state.lock().unwrap().outputs.clone() {
        provider.inner.free(&output).unwrap();
    }
    for handle in borrowed {
        provider.inner.free(&handle).unwrap();
    }
}
