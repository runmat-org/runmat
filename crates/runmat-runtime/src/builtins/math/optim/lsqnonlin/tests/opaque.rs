//! Existing vector control also exercises the opaque callback ABI and owner.
use super::*;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
pub(super) fn controls() {
    for (underdetermined, subproblem, limit, failure) in [
        (false, "cg", 0., false),
        (false, "cg", 200., false),
        (false, "factorization", 200., false),
        (true, "factorization", 200., false),
        (false, "cg", 200., true),
    ] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let f = crate::user_functions::test_support::exact_local_binding(&runtime, 1);
        let j = crate::user_functions::test_support::exact_local_binding(&runtime, 2);
        let _entry = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
            f.clone(),
            j.clone(),
        ]);
        let fun = crate::user_functions::test_support::exact_local_value("opaque_residual", f);
        let multiply = crate::user_functions::test_support::exact_local_value("opaque_product", j);
        let calls = Arc::new(AtomicUsize::new(0));
        let products = Arc::new(Mutex::new([0usize; 3]));
        let backing = Arc::new(Mutex::new(HashMap::new()));
        let (observed, roles, owners) = (calls.clone(), products.clone(), backing.clone());
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, requested| {
                let values = if function == 1 {
                    assert_eq!(requested, 2);
                    let Value::Tensor(x) = &args[0] else {
                        panic!("original x shape");
                    };
                    assert_eq!(x.shape(), &[2, 1]);
                    let x = x.materialize_f64();
                    let (residual, matrix, rows) = if underdetermined {
                        (vec![x[0] + x[1] - 3.], vec![1., 1.], 1)
                    } else {
                        (
                            vec![x[0] * x[0] - 4., x[1] - 3., x[0] * x[0] + x[1] - 7.],
                            vec![2. * x[0], 0., 2. * x[0], 0., 1., 1.],
                            3,
                        )
                    };
                    let matrix = Tensor::new(matrix, vec![rows, 2]).unwrap();
                    let generation = observed.fetch_add(1, Ordering::SeqCst) + 1;
                    owners
                        .lock()
                        .unwrap()
                        .insert(generation, matrix.as_f64_slice().unwrap().as_ptr() as usize);
                    vec![
                        tensor(residual, vec![rows, 1]),
                        Value::Struct(StructValue::from_fields_unbounded([
                            ("generation".into(), Value::Num(generation as f64)),
                            ("matrix".into(), Value::Tensor(matrix)),
                        ])),
                    ]
                } else {
                    if failure {
                        return Box::pin(async {
                            Err(crate::build_runtime_error("original multiply failure")
                                .with_identifier("RunMat:test:OriginalProduct")
                                .with_source(std::io::Error::other("original product cause"))
                                .build())
                        });
                    }
                    assert_eq!(function, 2);
                    assert_eq!(requested, 1);
                    let Value::Struct(info) = &args[0] else {
                        panic!("arbitrary structured Jinfo");
                    };
                    let Value::Num(generation) = &info.fields["generation"] else {
                        panic!("generation");
                    };
                    let Value::Tensor(matrix) = &info.fields["matrix"] else {
                        panic!("original backing");
                    };
                    assert_eq!(
                        matrix.as_f64_slice().unwrap().as_ptr() as usize,
                        owners.lock().unwrap()[&(*generation as usize)],
                        "tracking did not copy Jinfo numeric backing"
                    );
                    let Value::Tensor(y) = &args[1] else {
                        panic!("typed product matrix");
                    };
                    let Value::Num(flag) = &args[2] else {
                        panic!("numeric product role");
                    };
                    let flag = *flag;
                    let m = matrix.shape()[0];
                    let n = matrix.shape()[1];
                    let matrix = matrix.as_f64_slice().unwrap();
                    let yv = y.as_f64_slice().unwrap();
                    let (rows, lanes, role) = if flag > 0. {
                        (m, 1, 0)
                    } else if flag < 0. {
                        (n, 1, 1)
                    } else {
                        (n, 2, 2)
                    };
                    assert_eq!(y.shape(), &[if flag < 0. { m } else { n }, lanes]);
                    roles.lock().unwrap()[role] += 1;
                    let mut out = vec![0.; rows * lanes];
                    for lane in 0..lanes {
                        if flag > 0. {
                            for row in 0..m {
                                for column in 0..n {
                                    out[lane * m + row] +=
                                        matrix[column * m + row] * yv[lane * n + column];
                                }
                            }
                        } else if flag < 0. {
                            for column in 0..n {
                                for row in 0..m {
                                    out[lane * n + column] +=
                                        matrix[column * m + row] * yv[lane * m + row];
                                }
                            }
                        } else {
                            for column in 0..n {
                                for row in 0..m {
                                    let mut inner = 0.;
                                    for axis in 0..n {
                                        inner += matrix[axis * m + row] * yv[lane * n + axis];
                                    }
                                    out[lane * n + column] += matrix[column * m + row] * inner;
                                }
                            }
                        }
                    }
                    if subproblem == "cg" && flag != 0. {
                        let mut pointers = vec![0];
                        let mut indices = Vec::new();
                        let mut coefficients = Vec::new();
                        for column in 0..lanes {
                            for row in 0..rows {
                                let value = out[column * rows + row];
                                if value != 0. {
                                    indices.push(row);
                                    coefficients.push(value);
                                }
                            }
                            pointers.push(coefficients.len());
                        }
                        vec![Value::SparseTensor(
                            runmat_value::SparseTensor::new(
                                rows,
                                lanes,
                                pointers,
                                indices,
                                coefficients,
                            )
                            .unwrap(),
                        )]
                    } else {
                        vec![tensor(out, vec![rows, lanes])]
                    }
                };
                Box::pin(async move {
                    crate::execution::RootedValueSequence::new(
                        runmat_value::ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|error| error.into_runtime_error())
                })
            },
        )))
        .unwrap();
        let options = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            "optimoptions",
            &[
                Value::from("lsqnonlin"),
                Value::from("SpecifyObjectiveGradient"),
                Value::Bool(true),
                Value::from("JacobMult"),
                multiply,
                Value::from("SubproblemAlgorithm"),
                Value::from(subproblem),
                Value::from("PrecondBandWidth"),
                Value::Num(0.),
                Value::from("MaxPCGIter"),
                Value::Num(20.),
                Value::from("MaxFunctionEvaluations"),
                Value::Num(limit),
            ],
            1,
        ))
        .expect("actual typed JacobMult constructor");
        let _outputs = crate::output_count::push_output_count(Some(7)).unwrap();
        let returned = block_on(runtime.scope(lsqnonlin_builtin(
            fun,
            vec![
                tensor(vec![1., 0.], vec![2, 1]),
                tensor(vec![], vec![0, 0]),
                tensor(vec![], vec![0, 0]),
                options.value().clone(),
            ],
        )));
        if failure {
            let error = returned.unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:test:OriginalProduct"));
            assert_eq!(
                error.source.as_ref().unwrap().to_string(),
                "original product cause"
            );
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            continue;
        }
        let returned = returned.expect("actual opaque solver");
        let Value::OutputList(out) = returned else {
            panic!("seven outputs");
        };
        let Value::Struct(info) = &out[4] else {
            panic!("information");
        };
        assert_eq!(
            info.fields["funcCount"],
            Value::Num(calls.load(Ordering::SeqCst) as f64)
        );
        if limit == 0. {
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(*products.lock().unwrap(), [0, 0, 0]);
            assert_eq!(info.fields["algorithm"], Value::from("not-started"));
            assert!(matches!(out[1],Value::Num(value) if value.is_nan()));
            continue;
        }
        let Value::Tensor(x) = &out[0] else {
            panic!("solution");
        };
        let x = x.materialize_f64();
        let Value::Tensor(j) = &out[6] else {
            panic!("actual output matrix, not Jinfo");
        };
        if underdetermined {
            assert_eq!(info.fields["algorithm"], Value::from("levenberg-marquardt"));
            assert!((x[0] + x[1] - 3.).abs() < 1e-5);
            assert_eq!(j.shape(), &[1, 2]);
            assert_eq!(j.materialize_f64(), vec![1., 1.]);
        } else {
            assert_eq!(
                info.fields["algorithm"],
                Value::from("trust-region-reflective")
            );
            assert!((x[0] - 2.).abs() < 1e-5 && (x[1] - 3.).abs() < 1e-5);
            assert_eq!(j.shape(), &[3, 2]);
            assert_eq!(
                j.materialize_f64(),
                vec![2. * x[0], 0., 2. * x[0], 0., 1., 1.]
            );
            let roles = *products.lock().unwrap();
            assert!(
                roles.iter().all(|count| *count > 0),
                "all documented product roles are real calls"
            );
            if subproblem == "cg" {
                assert!(matches!(info.fields["cgiterations"],Value::Num(v) if v>0.));
            }
        }
    }
}
