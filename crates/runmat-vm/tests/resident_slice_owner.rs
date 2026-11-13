//! Resident source ownership is independent of the current allocation default.
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
};
use runmat_value::Value;
use std::{collections::HashMap, sync::OnceLock};

#[test]
fn exact_integer_slices_use_source_owner_after_default_is_removed_or_replaced() {
    static SOURCE: OnceLock<InProcessProvider> = OnceLock::new();
    static OTHER: OnceLock<InProcessProvider> = OnceLock::new();
    let source = SOURCE.get_or_init(InProcessProvider::new);
    let other = OTHER.get_or_init(InProcessProvider::new);
    // Register the physical owner before removing its default selection. This
    // binary has one test, so global default changes cannot contaminate peers.
    unsafe {
        runmat_accelerate_api::register_provider(source);
    }
    let handle = source
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[9007199254740993, 7, u64::MAX, 9]),
            shape: &[2, 2],
        })
        .unwrap();
    let ast = runmat_parser::parse("j = uint64(2); result = A(:, j);").unwrap();
    let variables = HashMap::from([("A".to_string(), 0)]);
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::new(&variables)).unwrap();
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).unwrap();
    let bytecode = runmat_vm::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id).unwrap();
    for replace in [false, true] {
        runmat_accelerate_api::clear_provider();
        if replace {
            unsafe {
                runmat_accelerate_api::register_provider(other);
            }
        }
        let vars = vec![Value::GpuTensor(handle.clone())];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            stacker::grow(32 * 1024 * 1024, || {
                futures::executor::block_on(runmat_vm::interpret_workspace(
                    &bytecode,
                    runmat_vm::InterpreterVars::from_owned(vars),
                    None,
                    runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
                        runmat_runtime::execution::RuntimeExecutionService::new(),
                    )),
                    None,
                    None,
                ))
            })
            .outcome
            .unwrap();
        let slot = bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "result").then_some(*slot))
            .expect("compiled result slot");
        let value = completion.values.get(slot).expect("completed result slot");
        let Value::GpuTensor(result) = &*value else {
            panic!("slice must retain resident integer storage");
        };
        assert_eq!(result.shape(), [2, 1]);
        assert_eq!(result.device_id(), handle.device_id());
        let host = futures::executor::block_on(source.download_integer(result)).unwrap();
        assert_eq!(host.data, HostIntegerDataOwned::U64(vec![u64::MAX, 9]));
        let empty = runmat_runtime::indexing::read_slice::read_gpu_slice_from_plan(
            &handle,
            &runmat_runtime::indexing::plan::IndexPlan::new(
                vec![],
                vec![2, 0],
                vec![2, 0],
                2,
                vec![2, 2],
            ),
        )
        .unwrap();
        let Value::GpuTensor(empty) = empty else {
            panic!("empty slice lost residency");
        };
        assert_eq!(empty.device_id(), handle.device_id());
        let host = futures::executor::block_on(source.download_integer(&empty)).unwrap();
        assert_eq!(host.shape, vec![2, 0]);
        assert_eq!(host.data, HostIntegerDataOwned::U64(vec![]));
    }
    runmat_accelerate_api::clear_provider();
}
