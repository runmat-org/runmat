//! Called within the existing long-lived admitted-provider fixture.
use super::*;

pub(super) fn run() {
    for extensions in [false, true] {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(extensions);
        for (name, input, expected) in [
            ("log", "[1 4]", [0.0, 4.0_f64.ln()]),
            ("log10", "[1 4]", [0.0, 4.0_f64.log10()]),
            ("log1p", "[0 3]", [0.0, 4.0_f64.ln()]),
            ("log2", "[1 4]", [0.0, 2.0]),
            ("sqrt", "[1 4]", [1.0, 2.0]),
            ("realsqrt", "[1 4]", [1.0, 2.0]),
        ] {
            let bytecode = support::compile_source(&format!(
                "a = reshape({input}, [1 1 2]); name = '{name}'; result = feval(name, a);"
            ))
            .unwrap();
            for gpu in [false, true] {
                runmat_accelerate::placement::reset();
                let kind = if gpu {
                    ExecutionCandidateKind::ProviderOperation
                } else {
                    ExecutionCandidateKind::SharedRuntime
                };
                let (context, selection) = context(name, kind);
                let values = execute(&bytecode, context).unwrap();
                let value = output(&bytecode, &values);
                assert_eq!(matches!(value, Value::GpuTensor(_)), gpu, "{name}");
                let Value::Tensor(gathered) =
                    block_on(runmat_runtime::gather_if_needed_async(value)).unwrap()
                else {
                    panic!("dense result");
                };
                assert_eq!(gathered.shape(), vec![1, 1, 2]);
                for (actual, expected) in gathered.as_f64_slice().unwrap().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12, "{name}");
                }
                assert_execution(&selection, gpu, 1);
                let resident = match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                };
                retire_output_after_frame(values, resident);
            }
            for source in [
                format!("a = [1 4]; [x,y,z] = {name}(a);"),
                format!("a = [1 4]; result = {name}(a, 2);"),
            ] {
                let bytecode = support::compile_source(&source).unwrap();
                let (context, selection) = context(name, ExecutionCandidateKind::ProviderOperation);
                assert!(execute(&bytecode, context).is_err(), "{name}");
                assert!(!selection
                    .requests
                    .borrow()
                    .iter()
                    .any(|request| request.signature.operation == name));
            }
        }
    }
}
