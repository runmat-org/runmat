//! Resident operators reach their original numerical owner before placement.
use super::{
    block_on, register_wgpu_provider, test_helpers, RuntimeContext, Value, WgpuProviderOptions,
};
use runmat_accelerate_api::{AccelProvider, HostNumericDataOwned};

#[test]
#[ignore = "requires an actual WGPU adapter; execute explicitly for resident operator qualification"]
fn actual_resident_transposes_matrix_product_and_power_preserve_original_values() {
    runmat_gc::gc_test_context(|| {
        let provider = register_wgpu_provider(WgpuProviderOptions::default())
            .expect("actual WGPU adapter required");
        let source = [1.0_f32, 3.0, 2.0, 4.0];
        let bytecode = test_helpers::compile_source(
            "A=gpuArray(single([1 2;3 4])); T=A.'; H=A'; P=A^2; M=A*A;",
        )
        .expect("compile canonical resident operators");
        let slot = |name: &str| {
            bytecode
                .var_names
                .iter()
                .find_map(|(slot, actual)| (actual == name).then_some(*slot))
                .expect("compiled operator binding")
        };
        let values = Vec::new();
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(values),
                None,
                runtime,
                None,
                None,
            ))
            .outcome
            .expect("execute each operator through its original owner");
        for (name, expected) in [
            ("A", source),
            ("T", [1.0, 2.0, 3.0, 4.0]),
            ("H", [1.0, 2.0, 3.0, 4.0]),
            ("P", [7.0, 15.0, 10.0, 22.0]),
            ("M", [7.0, 15.0, 10.0, 22.0]),
        ] {
            let handle = match &*completion
                .value()
                .values
                .get(slot(name))
                .expect("completed operator binding")
            {
                Value::GpuTensor(handle) => handle.clone(),
                _ => panic!("{name} lost its original resident output contract"),
            };
            assert_eq!(handle.shape(), &[2, 2]);
            assert_eq!(handle.device_id(), provider.device_id());
            let downloaded = block_on(provider.download_numeric(&handle))
                .expect("read the original resident operation result");
            let HostNumericDataOwned::F32(actual) = downloaded.data else {
                panic!("{name} changed the original single class");
            };
            assert_eq!(actual.as_slice(), &expected, "{name}");
        }
    });
}
