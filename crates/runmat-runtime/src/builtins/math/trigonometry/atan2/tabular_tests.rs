use super::*;
use crate::builtins::common::test_support;
use crate::builtins::math::trigonometry::test_owner;
use futures::{executor::block_on, task::noop_waker};
use runmat_accelerate_api::{allocation::WeakGpuTensorHandle, test_support::RetirementProbe};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::Mutex,
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
    second: Option<WeakGpuTensorHandle>,
    outputs: Vec<(WeakGpuTensorHandle, RetirementProbe)>,
    downloads: usize,
}
struct Provider {
    inner: test_owner::Owner,
    state: Mutex<State>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "atan2 tabular owning fixture".into()
    }
    fn upload(&self, value: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(value)
    }
    fn upload_numeric_raw(
        &self,
        value: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let mut state = self.state.lock().unwrap();
        match state.fault {
            Fault::BorrowedAlias => return Ok(state.second.as_ref().unwrap().acquire()?),
            Fault::PriorResultAlias if !state.outputs.is_empty() => {
                return Ok(state.outputs[0].0.acquire()?)
            }
            _ => {}
        }
        let output = self.inner.upload_numeric(value).map_err(|error| {
            runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
        })?;
        state
            .outputs
            .push((output.downgrade(), self.inner.probe(&output)));
        Ok(output)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let fault = {
                let mut state = self.state.lock().unwrap();
                state.downloads += 1;
                if state.second.as_ref().is_some_and(|second| {
                    second
                        .acquire()
                        .is_ok_and(|second| gpu_helpers::same_gpu_handle(&second, handle))
                }) {
                    state.fault
                } else {
                    Fault::None
                }
            };
            match fault {
                Fault::LaterFailure => anyhow::bail!("injected later column failure"),
                Fault::LaterPending => std::future::pending().await,
                _ => self.inner.download(handle),
            }
        })
    }
}

fn fixture(fault: Fault) -> (&'static Provider, Vec<GpuTensorHandle>, Value) {
    let provider = Box::leak(Box::new(Provider {
        inner: test_owner::Owner::new(),
        state: Mutex::default(),
    }));
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
        second: Some(handles[1].downgrade()),
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
        for (_, probe) in &state.outputs {
            assert_eq!(probe.retirements(), 1);
        }
        for handle in &borrowed {
            assert_eq!(provider.inner.probe(handle).retirements(), 0);
        }
        drop(state);
        for handle in borrowed {
            assert!(provider.inner.download(&handle).is_ok());
            handle.release().unwrap();
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
    assert_eq!(borrowed[0].shape(), [2, 1]);
    let Value::Object(object) = &table else {
        panic!("table fixture")
    };
    assert_eq!(crate::builtins::table::table_height(object).unwrap(), 2);
    let error = block_on(crate::dispatcher::call_builtin_async(
        "atan2",
        &[table, Value::Num(1.0)],
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        ATAN2_ERROR_COMPLEX_UNSUPPORTED.identifier
    );
    let state = provider.state.lock().unwrap();
    assert_eq!(state.downloads, 0);
    assert!(state.outputs.is_empty());
    assert!(borrowed
        .iter()
        .all(|handle| provider.inner.probe(handle).retirements() == 0));
    drop(state);
    for handle in borrowed {
        handle.release().unwrap();
    }
}

#[test]
fn resident_table_columns_use_admitted_shape_without_download() {
    let _serial = test_support::accel_test_lock();
    let owner = test_owner::Owner::new();
    for shape in [
        vec![],
        vec![2],
        vec![2, 1],
        vec![1, 2],
        vec![0, 2],
        vec![2, 1, 2],
    ] {
        let data = vec![0.0; shape.iter().product()];
        let handle = owner
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .unwrap();
        let probe = owner.probe(&handle);
        let host = runmat_value::Tensor::new(data, shape.clone()).unwrap();
        let expected_rows = host.rows();
        let table = crate::builtins::table::table_from_columns(
            vec!["Resident".into(), "Host".into()],
            vec![Value::GpuTensor(handle.clone()), Value::Tensor(host)],
        )
        .unwrap();
        let Value::Object(object) = &table else {
            panic!("table")
        };
        assert_eq!(
            crate::builtins::table::table_height(object).unwrap(),
            expected_rows
        );
        assert_eq!(handle.shape(), shape);
        assert_eq!(probe.retirements(), 0);
        drop(table);
        handle.release().unwrap();
        assert_eq!(probe.retirements(), 1);
    }
}

#[test]
fn atan2_tabular_success_publishes_owned_columns_with_numeric_oracle() {
    let _serial = test_support::accel_test_lock();
    let (provider, borrowed, table) = fixture(Fault::None);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let output_owner = block_on(crate::dispatcher::call_builtin_async(
        "atan2",
        &[table, Value::Num(1.0)],
    ))
    .unwrap();
    let Value::Object(output) = output_owner.value() else {
        panic!("table result")
    };
    let columns = crate::builtins::table::table_variables(output).unwrap();
    for (name, expected) in [
        ("A", [1.0_f64.atan(), (-1.0_f64).atan()]),
        ("B", [0.0, 1.0_f64.atan()]),
    ] {
        let Value::GpuTensor(handle) = &columns.fields[name] else {
            panic!("resident column")
        };
        let actual = provider.inner.download(handle).unwrap();
        assert_eq!(actual.data, expected);
    }
    assert!(provider
        .state
        .lock()
        .unwrap()
        .outputs
        .iter()
        .all(|(_, probe)| probe.retirements() == 0));
    drop(columns);
    drop(output_owner);
    assert!(provider
        .state
        .lock()
        .unwrap()
        .outputs
        .iter()
        .all(|(_, probe)| probe.retirements() == 1));
    for handle in borrowed {
        handle.release().unwrap();
    }
}
