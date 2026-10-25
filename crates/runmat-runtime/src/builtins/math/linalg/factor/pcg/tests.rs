use super::kernel::{self, PcgSolveInput, PcgWork};
use crate::builtins::math::linalg::ops::sparse_operator::SparseLinearOperator;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use runmat_value::{ComplexElement, SparseTensor, Tensor, Value};
use std::cell::RefCell;
use std::rc::Rc;

mod preconditioner_failure;

struct ManagedParallel(ManagedHostEffectLedger);

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pcg_csc_real_and_hermitian_complex_converge_without_dense_coefficients() {
    let real = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0, 1.0, 3.0],
    )
    .unwrap();
    let SparseLinearOperator::RealF64(operator) =
        SparseLinearOperator::borrow_unbounded(&real).unwrap()
    else {
        panic!("real double operator");
    };
    let (mut x, mut best, mut r, mut p, mut ap, mut residuals) =
        ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 3]);
    let outcome = futures::executor::block_on(kernel::solve(
        PcgSolveInput {
            rhs: &[1.0, 2.0],
            initial: None,
            tolerance: 1e-12,
            maximum_iterations: 2,
            work: PcgWork {
                x: &mut x,
                best: &mut best,
                residual: &mut r,
                direction: &mut p,
                intermediate: None,
                residual_norms: &mut residuals,
                coefficients: None,
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
        None,
    ))
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!(outcome.iterations <= 2 && outcome.relres < 1e-12);
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    assert_eq!(residuals[0], 5.0_f64.sqrt());

    let (mut limited_x, mut best, mut r, mut p, mut ap, mut history) =
        ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
    let limited = futures::executor::block_on(kernel::solve(
        PcgSolveInput {
            rhs: &[1.0, 2.0],
            initial: None,
            tolerance: 1e-12,
            maximum_iterations: 1,
            work: PcgWork {
                x: &mut limited_x,
                best: &mut best,
                residual: &mut r,
                direction: &mut p,
                intermediate: None,
                residual_norms: &mut history,
                coefficients: None,
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
        None,
    ))
    .unwrap();
    assert_eq!(limited.flag, 1);
    assert_eq!(limited.iterations, 1);
    assert_eq!(limited.residual_count, 2);
    assert!(history[1] < history[0]);

    let identity = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
    let SparseLinearOperator::RealF64(identity) =
        SparseLinearOperator::borrow_unbounded(&identity).unwrap()
    else {
        panic!("double CSC identity");
    };
    let (mut x, mut best, mut r, mut p, mut ap, mut history) =
        ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 3]);
    let huge = futures::executor::block_on(kernel::solve(
        PcgSolveInput {
            rhs: &[1e200, -1e200],
            initial: None,
            tolerance: 1e-12,
            maximum_iterations: 2,
            work: PcgWork {
                x: &mut x,
                best: &mut best,
                residual: &mut r,
                direction: &mut p,
                intermediate: None,
                residual_norms: &mut history,
                coefficients: None,
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| identity.apply_into_unbounded(input, output),
        || Ok(()),
        None,
    ))
    .unwrap();
    assert_eq!(huge.flag, 0);
    assert_eq!(x, [1e200, -1e200]);
    assert_eq!(huge.relres, 0.0);
    assert!(history[0].is_finite() && history[0] > 1e200);

    let complex = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![(2.0, 0.0), (0.0, -1.0), (0.0, 1.0), (2.0, 0.0)],
    )
    .unwrap();
    let SparseLinearOperator::ComplexF32(operator) =
        SparseLinearOperator::borrow_unbounded(&complex).unwrap()
    else {
        panic!("complex single operator");
    };
    let mut x = [ComplexElement(0.0, 0.0); 2];
    let mut best = [ComplexElement(0.0, 0.0); 2];
    let mut r = [ComplexElement(0.0, 0.0); 2];
    let mut p = [ComplexElement(0.0, 0.0); 2];
    let mut ap = [ComplexElement(0.0, 0.0); 2];
    let mut residuals = [0.0; 3];
    let outcome = futures::executor::block_on(kernel::solve(
        PcgSolveInput {
            rhs: &[ComplexElement(1.0, 0.0), ComplexElement(0.0, 0.0)],
            initial: None,
            tolerance: 1e-6,
            maximum_iterations: 2,
            work: PcgWork {
                x: &mut x,
                best: &mut best,
                residual: &mut r,
                direction: &mut p,
                intermediate: None,
                residual_norms: &mut residuals,
                coefficients: None,
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
        None,
    ))
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!((x[0].0 - 2.0 / 3.0).abs() < 1e-5);
    assert!((x[1].1 - 1.0 / 3.0).abs() < 1e-5);

    // The public request/execution path accepts native dense coefficients
    // without first producing a sparse or converted coefficient owner.
    let dense = Value::Tensor(Tensor::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let request = super::request::parse(
        dense,
        rhs,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap();
    let Value::Tensor(solution) =
        futures::executor::block_on(super::execution::execute(request, None, Some(1))).unwrap()
    else {
        panic!("dense pcg returned a non-real result");
    };
    let values = solution.as_f64_slice().unwrap();
    assert!((values[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((values[1] - 7.0 / 11.0).abs() < 1e-12);

    let plan = super::execution::quoted_f64_plan_for_test(2, 2, Some(5)).unwrap();
    let ledger =
        ManagedHostEffectLedger::new(plan.scratch_peak_bytes() + plan.retained_output_bytes());
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let dense = Value::Tensor(Tensor::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let request = super::request::parse(
        dense,
        rhs,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap();
    let Value::OutputList(outputs) =
        futures::executor::block_on(super::execution::execute(request, Some(&context), Some(5)))
            .unwrap()
    else {
        panic!("five managed outputs retain an exact list owner");
    };
    assert_eq!(outputs.len(), 5);
    assert!(outputs.has_retained_claim());
    let Value::Tensor(solution) = &outputs[0] else {
        panic!("first output is the native solution");
    };
    assert!((solution.as_f64_slice().unwrap()[0] - 1.0 / 11.0).abs() < 1e-12);
    assert_eq!(outputs[1], Value::Num(0.0));
    let Value::Tensor(history) = &outputs[4] else {
        panic!("fifth output is residual history");
    };
    assert_eq!(history.as_f64_slice().unwrap().len(), 3);
    let alias = outputs.clone();
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(outputs);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // alpha exceeds f32::MAX here, but alpha * p and the answer are finite.
    // Casting alpha before multiplication would manufacture infinity.
    let coefficient = f32::from_bits(1 << 19); // exactly 2^-130
    let rhs = 1.0_f32 / 1024.0; // exactly 2^-10
    let tiny = SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![coefficient]).unwrap();
    let SparseLinearOperator::RealF32(tiny) =
        SparseLinearOperator::borrow_unbounded(&tiny).unwrap()
    else {
        panic!("single CSC operator");
    };
    let (mut x, mut best, mut r, mut p, mut ap, mut history) = (
        [0.0_f32; 1],
        [0.0_f32; 1],
        [0.0_f32; 1],
        [0.0_f32; 1],
        [0.0_f32; 1],
        [0.0; 2],
    );
    let outcome = futures::executor::block_on(kernel::solve(
        PcgSolveInput {
            rhs: &[rhs],
            initial: None,
            tolerance: 1e-3,
            maximum_iterations: 1,
            work: PcgWork {
                x: &mut x,
                best: &mut best,
                residual: &mut r,
                direction: &mut p,
                intermediate: None,
                residual_norms: &mut history,
                coefficients: None,
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| tiny.apply_into_unbounded(input, output),
        || Ok(()),
        None,
    ))
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!(x[0].is_finite());
    assert_eq!(x[0], 2.0_f32.powi(120));
    assert_eq!(coefficient * x[0], rhs);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pcg_reuses_selected_callable_a_and_both_preconditioners_without_replaying_callbacks() {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let calls = Rc::new(RefCell::new([0usize; 3]));
    let observed = Rc::clone(&calls);
    runtime.set_bound_function_invoker(Some(Rc::new(
        move |_runtime, binding, arguments, requested, _environment| {
            assert_eq!(requested, 1);
            let index = binding.function - 9100;
            observed.borrow_mut()[index] += 1;
            let [Value::Tensor(input)] = arguments.supplied() else {
                panic!("one native vector argument");
            };
            let input = input.as_f64_slice().unwrap();
            assert_eq!(input.len(), 2);
            let output = if index == 0 {
                vec![4.0 * input[0] + input[1], input[0] + 3.0 * input[1]]
            } else {
                vec![0.5 * input[0], 0.5 * input[1]]
            };
            let output = Value::Tensor(Tensor::new(output, vec![2, 1]).unwrap());
            crate::execution::admit_executable_sequence_with(
                &_runtime.clone(),
                move || async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(output)
                },
            )
        },
    )));
    let target = |index: usize| {
        crate::user_functions::test_support::exact_local_handle(
            &runtime,
            "pcg_callback",
            9100 + index,
        )
    };
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let request = super::request::parse(
        target(0),
        rhs,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1e-12),
            Value::Num(2.0),
            target(1),
            target(2),
        ]),
    )
    .unwrap();
    let Value::Tensor(solution) =
        futures::executor::block_on(super::execution::execute(request, Some(&runtime), Some(1)))
            .unwrap()
    else {
        panic!("callable pcg returned a non-real result");
    };
    let values = solution.as_f64_slice().unwrap();
    assert!((values[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((values[1] - 7.0 / 11.0).abs() < 1e-12);
    // A*x0, two A*p products, then one independent convergence certificate.
    assert_eq!(*calls.borrow(), [4, 2, 2]);
}

#[test]
fn native_four_report_the_residual_of_the_returned_rounded_iterate() {
    use super::kernel::PcgScalar;
    fn identity<T: PcgScalar>(rhs: T, reject: bool) {
        let (mut x, mut best, mut r, mut p, mut ap) = (
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
        );
        let mut history = [0.0; 2];
        let result = futures::executor::block_on(kernel::solve(
            PcgSolveInput {
                rhs: &[rhs],
                initial: None,
                tolerance: 1e-12,
                maximum_iterations: 1,
                work: PcgWork {
                    x: &mut x,
                    best: &mut best,
                    residual: &mut r,
                    direction: &mut p,
                    intermediate: None,
                    residual_norms: &mut history,
                    coefficients: None,
                },
                operation_scratch: &mut ap,
            },
            None,
            |input, output| {
                output.copy_from_slice(input);
                Ok(())
            },
            || Ok(()),
            None,
        ));
        if reject {
            assert_eq!(
                result.unwrap_err().gpu_gather_retry,
                crate::GpuGatherRetry::Never
            );
        } else {
            let outcome = result.unwrap();
            assert_eq!(outcome.flag, 0);
            assert_eq!(outcome.relres, 0.0);
            assert!(x[0] == rhs);
        }
    }
    fn check<T: PcgScalar>(rhs: T) {
        fn magnitude<T: PcgScalar>(value: T) -> f64 {
            let (real, imaginary) = value.components();
            real.hypot(imaginary)
        }
        let (mut x, mut best, mut r, mut p, mut ap) = (
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
            [T::zero(); 1],
        );
        let mut history = [0.0; 3];
        let mut products = 0;
        let outcome = futures::executor::block_on(kernel::solve(
            PcgSolveInput {
                rhs: &[rhs],
                initial: None,
                tolerance: 1e-12,
                maximum_iterations: 2,
                work: PcgWork {
                    x: &mut x,
                    best: &mut best,
                    residual: &mut r,
                    direction: &mut p,
                    intermediate: None,
                    residual_norms: &mut history,
                    coefficients: None,
                },
                operation_scratch: &mut ap,
            },
            None,
            |input, output| {
                products += 1;
                output[0] = T::zero();
                output[0].add_scaled(7.0, input[0]);
                Ok(())
            },
            || Ok(()),
            None,
        ))
        .unwrap();
        // Independent native A*x, not the recurrence's r or reported relres.
        let mut ax = T::zero();
        ax.add_scaled(7.0, x[0]);
        let mut actual = rhs;
        actual.subtract_scaled(1.0, ax);
        let relative = magnitude(actual) / magnitude(rhs);
        assert!((outcome.relres - relative).abs() <= 1e-15);
        assert_eq!(outcome.flag == 0, relative <= 1e-12);
        assert!((history[outcome.iterations] - magnitude(actual)).abs() <= 1e-12);
        assert!(products >= 3, "candidate requires its own A*x product");
    }
    check(31.0f32);
    check(31.0f64);
    check(ComplexElement(31.0f32, 13.0f32));
    check(ComplexElement(31.0f64, 13.0f64));
    for nonfinite in [f64::NAN, f64::INFINITY] {
        identity(nonfinite as f32, true);
        identity(nonfinite, true);
        identity(ComplexElement(nonfinite as f32, 1.0f32), true);
        identity(ComplexElement(nonfinite, 1.0f64), true);
    }
    identity(3e38f32, false);
    identity(1.3e308f64, false);
    identity(ComplexElement(3e38f32, 3e38f32), false);
    identity(ComplexElement(1.3e308f64, 1.3e308f64), false);
}

#[test]
fn candidate_residual_callback_error_is_terminal_without_replay() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    for cancel in [false, true] {
        let cancellation = CancellationSource::new();
        let callback_cancellation = cancellation.clone();
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
        let calls = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&calls);
        runtime.set_bound_function_invoker(Some(Rc::new(
            move |_runtime, _binding, arguments, _requested, _environment| {
                let Value::Tensor(input) = &arguments.supplied()[0] else {
                    panic!("native input")
                };
                let input = input.as_f64_slice().unwrap()[0];
                observed.borrow_mut().push(input);
                let call = observed.borrow().len();
                if call == 3 && cancel {
                    callback_cancellation.cancel(CancellationReason::User);
                }
                crate::execution::admit_executable_sequence_with(
                    &_runtime.clone(),
                    move || async move {
                        if call == 3 && !cancel {
                            return Err(crate::build_runtime_error("candidate residual failed")
                                .with_identifier("RunMat:Test:CandidateResidual")
                                .with_source(std::io::Error::other("original candidate cause"))
                                .build());
                        }
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Tensor(Tensor::new(vec![7.0 * input], vec![1, 1]).unwrap()),
                        )
                    },
                )
            },
        )));
        let target = crate::user_functions::test_support::exact_local_handle(
            &runtime,
            "candidate_residual",
            9301,
        );
        let request = super::request::parse(
            target,
            Value::Tensor(Tensor::new(vec![31.0], vec![1, 1]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1e-12),
                Value::Num(2.0),
            ]),
        )
        .unwrap();
        let error = futures::executor::block_on(super::execution::execute(
            request,
            Some(&runtime),
            Some(5),
        ))
        .unwrap_err();
        if cancel {
            assert!(error
                .identifier()
                .is_some_and(|id| id.ends_with(":Cancelled")));
            let mut source = std::error::Error::source(&error);
            let mut typed_cancelled = false;
            while let Some(cause) = source {
                typed_cancelled |= matches!(
                    cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
                    Some(crate::execution::ExecutionServiceError::Cancelled)
                );
                source = cause.source();
            }
            assert!(typed_cancelled, "retain the actual cancellation cause");
        } else {
            assert_eq!(error.identifier(), Some("RunMat:Test:CandidateResidual"));
            assert!(error
                .source
                .as_ref()
                .is_some_and(|cause| cause.to_string() == "original candidate cause"));
        }
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let calls = calls.borrow();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], 0.0);
        assert_eq!(calls[1], 31.0);
        assert!((calls[2] - 31.0 / 7.0).abs() < 1e-14);
    }
}
