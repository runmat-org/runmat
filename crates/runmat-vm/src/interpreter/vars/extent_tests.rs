//! Slot-owner conservation only; no simulated GPU arithmetic is asserted.

use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};

#[test]
fn compiled_extent_growth_moves_each_storage_kind_without_detaching_shared_control() {
    for retained in [false, true] {
        for shared in [false, true] {
            let runtime = RuntimeContext::new(std::rc::Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ));
            let allocation = TestAllocationOwner::new()
                .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let identity = runmat_accelerate_api::handle_identity(&allocation.handle);
            let mut frame = InterpreterVars::from_owned(vec![Value::GpuTensor(allocation.handle)]);
            if retained {
                frame = frame.prepare_extent(&runtime, 2).unwrap();
            }
            let alias = shared.then(|| frame.promote_shared(None, None).unwrap());
            let frame = frame.prepare_extent(&runtime, 4).unwrap();
            assert_eq!(frame.len(), 4);
            {
                let value = frame.get(0).unwrap();
                let Value::GpuTensor(handle) = &*value else {
                    panic!("input allocation lease was replaced")
                };
                assert_eq!(runmat_accelerate_api::handle_identity(handle), identity);
            }
            if let Some(alias) = &alias {
                assert!(frame.shares_storage_with(alias));
                assert_eq!(alias.len(), 4);
            }
            assert_eq!(allocation.probe.retirements(), 0);
            drop(frame);
            assert_eq!(allocation.probe.retirements(), usize::from(!shared));
            drop(alias);
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
}
