#[cfg(not(target_family = "wasm"))]
#[test]
fn native_async_objects_and_explicit_concurrent_services_remain_send_sync() {
    fn require_send_sync<T: Send + Sync + ?Sized>() {}
    require_send_sync::<dyn super::MexAsyncResult>();
    require_send_sync::<dyn super::MexAsyncHostServices>();
    require_send_sync::<super::MexAsyncResultHandle>();
    require_send_sync::<dyn crate::ConcurrentMexBoundaryHostServices>();
    require_send_sync::<crate::MexCallState>();
}

#[cfg(target_family = "wasm")]
mod local {
    use super::super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    struct ResultOwner(Rc<Cell<usize>>);

    impl Drop for ResultOwner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    struct LocalResult {
        _owner: ResultOwner,
        completion: MexEngineCompletion<Vec<MxArray>>,
    }

    impl MexAsyncResult for LocalResult {
        fn cancel(&self, _allow_interrupt: bool) -> bool {
            false
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn wait(&self, _timeout: Option<Duration>) -> bool {
            true
        }
        fn result(&self) -> MexEngineCompletion<Vec<MxArray>> {
            self.completion.clone()
        }
    }

    struct LocalService {
        drops: Rc<Cell<usize>>,
        payload: RefCell<Option<MxArray>>,
    }

    impl MexAsyncHostServices for LocalService {
        fn submit(
            &self,
            _operation: MexAsyncOperation,
        ) -> Result<MexAsyncResultHandle, MexDiagnostic> {
            Ok(new_async_result(LocalResult {
                _owner: ResultOwner(self.drops.clone()),
                completion: MexEngineCompletion::from_result(Ok(vec![self
                    .payload
                    .borrow_mut()
                    .take()
                    .expect("single submission")])),
            }))
        }
    }

    #[wasm_bindgen_test::wasm_bindgen_test]
    fn worker_local_async_service_and_result_retain_actual_owner_until_final_drop() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let probe = allocation.probe;
        let payload = MxArray::gpu_borrowed(crate::MxClassId::Single, allocation.handle).unwrap();
        let drops = Rc::new(Cell::new(0));
        let service: Box<dyn MexAsyncHostServices> = Box::new(LocalService {
            drops: drops.clone(),
            payload: RefCell::new(Some(payload)),
        });
        let result = service
            .submit(MexAsyncOperation::GetVariable {
                workspace: "base".into(),
                name: "x".into(),
            })
            .unwrap();
        let retained = result.clone();
        drop(service);
        assert!(result.is_ready());
        assert!(result.wait(None));
        assert!(!result.cancel(true));
        let completion = result.result().result.unwrap();
        assert_eq!(completion.len(), 1);
        drop(result);
        assert_eq!(drops.get(), 0);
        assert_eq!(probe.retirements(), 0);
        drop(retained);
        assert_eq!(drops.get(), 1);
        assert_eq!(
            probe.retirements(),
            0,
            "returned completion owns the authentic allocation"
        );
        drop(completion);
        assert_eq!(probe.retirements(), 1);
    }
}
