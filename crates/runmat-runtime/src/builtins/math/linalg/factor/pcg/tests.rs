use super::kernel::{self, PcgSolveInput, PcgWork};
use crate::builtins::math::linalg::ops::sparse_operator::SparseLinearOperator;
use runmat_value::{ComplexElement, SparseTensor, Tensor, Value};
use std::cell::RefCell;
use std::rc::Rc;

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
    let outcome = kernel::solve(
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
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!(outcome.iterations <= 2 && outcome.relres < 1e-12);
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    assert_eq!(residuals[0], 5.0_f64.sqrt());

    let (mut limited_x, mut best, mut r, mut p, mut ap, mut history) =
        ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
    let limited = kernel::solve(
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
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
    )
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
    let huge = kernel::solve(
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
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| identity.apply_into_unbounded(input, output),
        || Ok(()),
    )
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
    let outcome = kernel::solve(
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
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| operator.apply_into_unbounded(input, output),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!((x[0].0 - 2.0 / 3.0).abs() < 1e-5);
    assert!((x[1].1 - 1.0 / 3.0).abs() < 1e-5);

    // The public request/execution path accepts native dense coefficients
    // without first producing a sparse or converted coefficient owner.
    let dense = Value::Tensor(Tensor::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
    let rhs = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let request = super::request::parse(dense, rhs, Vec::new()).unwrap();
    let Value::Tensor(solution) =
        futures::executor::block_on(super::execution::execute(request, None, Some(1))).unwrap()
    else {
        panic!("dense pcg returned a non-real result");
    };
    let values = solution.as_f64_slice().unwrap();
    assert!((values[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((values[1] - 7.0 / 11.0).abs() < 1e-12);

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
    let outcome = kernel::solve(
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
            },
            operation_scratch: &mut ap,
        },
        None,
        |input, output| tiny.apply_into_unbounded(input, output),
        || Ok(()),
    )
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
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(output)
            })
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
        vec![Value::Num(1e-12), Value::Num(2.0), target(1), target(2)],
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
    assert_eq!(*calls.borrow(), [3, 2, 2]);
}
