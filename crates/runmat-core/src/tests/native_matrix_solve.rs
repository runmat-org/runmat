//! Public matrix syntax and named solves retain the same sparse host boundary.

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunMatSession,
};
use futures::executor::block_on;
use runmat_value::{SparseTensor, Tensor, Value};

#[test]
fn sparse_triangular_operator_and_builtin_preserve_single_on_vm_and_native() {
    runmat_gc::gc_test_context(|| {
        let matrix = Value::SparseTensor(
            SparseTensor::new_f32(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0, 1.0, 3.0]).unwrap(),
        );
        let rhs = Value::Tensor(Tensor::from_f32(vec![2.0, 4.0, 4.0, 8.0], vec![2, 2]).unwrap());
        let expected =
            Value::Tensor(Tensor::from_f32(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]).unwrap());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "native-sparse-triangular@1",
                "nativeSparseTriangular.m",
                "function [x, y] = nativeSparseTriangular(A, B)\nx = A \\ B;\ny = mldivide(A, B);\nend\n",
            ),
            None,
        ))
        .expect("compile sparse solve forms");
        for control in [
            InvocationControl::default(),
            InvocationControl::default().force_generic_native(),
        ] {
            let result = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function("nativeSparseTriangular".into()),
                    arguments: crate::ProcedureArguments::Source(vec![matrix.clone(), rhs.clone()]),
                    requested_outputs: 2,
                },
                &control,
            ))
            .expect("execute sparse solve forms");
            assert_eq!(result.as_slice(), &[expected.clone(), expected.clone()]);
        }
    });
}
