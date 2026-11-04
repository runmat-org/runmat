use super::*;

struct Matrix;

impl GmresOperations<f64> for Matrix {
    type Error = ();

    async fn measured<R>(
        &mut self,
        work: runmat_sparse_solvers::krylov::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, runmat_sparse_solvers::krylov::usage::BatchCompletion),
    ) -> Result<R, ()> {
        assert!(runmat_simulation_usage::UsageContext::capture().is_none());
        work.operations().unwrap();
        Ok(run(self).0)
    }

    async fn coefficient(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), ()> {
        output[0] = 4.0 * input[0] + input[1];
        output[1] = 2.0 * input[0] + 3.0 * input[1];
        Ok(())
    }

    async fn precondition(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        _intermediate: &mut [f64],
    ) -> Result<PreconditionerStatus, ()> {
        output.copy_from_slice(input);
        Ok(PreconditionerStatus::Applied)
    }

    fn checkpoint(&mut self) -> Result<(), ()> {
        Ok(())
    }
    async fn cooperate(&mut self, _: u64) -> Result<(), ()> {
        <Self as GmresOperations<f64>>::checkpoint(self)
    }
}

#[test]
fn nonsymmetric_two_by_two_uses_true_residual() {
    let mut x = [0.0; 2];
    let mut best = [0.0; 2];
    let mut candidate = [0.0; 2];
    let mut previous = [0.0; 2];
    let mut residual = [0.0; 2];
    let mut product = [0.0; 2];
    let mut preconditioned = [0.0; 2];
    let mut intermediate = [0.0; 2];
    let mut basis = [0.0; 6];
    let mut hessenberg = [Complex64::new(0.0, 0.0); 6];
    let mut cosines = [0.0; 2];
    let mut sines = [Complex64::new(0.0, 0.0); 2];
    let mut rotated_rhs = [Complex64::new(0.0, 0.0); 3];
    let mut coefficients = [Complex64::new(0.0, 0.0); 2];
    let mut history = [0.0; 3];
    let mut work = GmresWork {
        x: &mut x,
        best: &mut best,
        candidate: &mut candidate,
        previous: &mut previous,
        residual: &mut residual,
        product: &mut product,
        preconditioned: &mut preconditioned,
        intermediate: &mut intermediate,
        basis: &mut basis,
        hessenberg: &mut hessenberg,
        cosines: &mut cosines,
        sines: &mut sines,
        rotated_rhs: &mut rotated_rhs,
        coefficients: &mut coefficients,
        history: &mut history,
    };
    let outcome = futures::executor::block_on(solve(
        &[1.0, 0.0],
        None,
        GmresParameters {
            restart: 2,
            max_outer: 1,
            tolerance: 1e-12,
        },
        &mut work,
        &mut Matrix,
    ))
    .unwrap();
    assert_eq!(outcome.flag, 0);
    assert!((x[0] - 0.3).abs() < 1e-12);
    assert!((x[1] + 0.2).abs() < 1e-12);
    assert!(outcome.relres < 1e-12);
    assert_eq!(outcome.residual_count, 3);

    let mut restarted_history = [0.0; 21];
    let mut restarted = GmresWork {
        x: &mut x,
        best: &mut best,
        candidate: &mut candidate,
        previous: &mut previous,
        residual: &mut residual,
        product: &mut product,
        preconditioned: &mut preconditioned,
        intermediate: &mut intermediate,
        basis: &mut basis[..4],
        hessenberg: &mut hessenberg,
        cosines: &mut cosines,
        sines: &mut sines,
        rotated_rhs: &mut rotated_rhs,
        coefficients: &mut coefficients,
        history: &mut restarted_history,
    };
    let restarted_outcome = futures::executor::block_on(solve(
        &[1.0, 0.0],
        None,
        GmresParameters {
            restart: 1,
            max_outer: 20,
            tolerance: 1e-10,
        },
        &mut restarted,
        &mut Matrix,
    ))
    .unwrap();
    assert_eq!(restarted_outcome.flag, 0);
    assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-10);
    assert!((2.0 * x[0] + 3.0 * x[1]).abs() < 1e-10);
    assert!(restarted_outcome.residual_count > 2);
}

#[test]
fn complex_rotation_annihilates_the_lower_component() {
    let a = Complex64::new(3.0, 4.0);
    let b = Complex64::new(-2.0, 5.0);
    let (c, s) = rotation(a, b).unwrap();
    let lower = -s.conj() * a + c * b;
    assert!(lower.norm() <= 1e-14 * a.norm().hypot(b.norm()));
    assert!((c * a + s * b).norm() > 0.0);
    assert!(rotation(Complex64::new(1.3e308, 0.0), Complex64::new(1.3e308, 0.0)).is_none());
}

struct Identity(f64);
impl<T: GmresScalar> GmresOperations<T> for Identity {
    type Error = ();
    async fn measured<R>(
        &mut self,
        work: runmat_sparse_solvers::krylov::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, runmat_sparse_solvers::krylov::usage::BatchCompletion),
    ) -> Result<R, ()> {
        assert!(runmat_simulation_usage::UsageContext::capture().is_none());
        work.operations().unwrap();
        Ok(run(self).0)
    }
    async fn coefficient(&mut self, input: &[T], output: &mut [T]) -> Result<(), ()> {
        for (&input, output) in input.iter().zip(output) {
            *output = T::from_complex(input.to_complex() * self.0).ok_or(())?;
        }
        Ok(())
    }
    async fn precondition(
        &mut self,
        input: &[T],
        output: &mut [T],
        _: &mut [T],
    ) -> Result<PreconditionerStatus, ()> {
        output.copy_from_slice(input);
        Ok(PreconditionerStatus::Applied)
    }
    fn checkpoint(&mut self) -> Result<(), ()> {
        Ok(())
    }
    async fn cooperate(&mut self, _: u64) -> Result<(), ()> {
        <Self as GmresOperations<T>>::checkpoint(self)
    }
}

#[test]
fn native_four_scaled_norms_preserve_finite_ratios_and_true_residuals() {
    use runmat_value::ComplexElement;
    fn check<T: GmresScalar>(value: Complex64, coefficient: f64) {
        let rhs = [T::from_complex(value).unwrap(); 4];
        let initial = [T::from_complex(value / coefficient).unwrap(); 4];
        let tolerance = if coefficient < 1e-300 { 1e-12 } else { 1e-6 };
        for exact_start in [false, true] {
            let mut x = [T::zero(); 4];
            let mut best = x;
            let mut candidate = x;
            let mut previous = x;
            let mut residual = x;
            let mut product = x;
            let mut preconditioned = x;
            let mut intermediate = x;
            let mut basis = [T::zero(); 8];
            let mut hessenberg = [Complex64::new(0.0, 0.0); 2];
            let mut cosines = [0.0];
            let mut sines = [Complex64::new(0.0, 0.0)];
            let mut rotated_rhs = [Complex64::new(0.0, 0.0); 2];
            let mut coefficients = [Complex64::new(0.0, 0.0)];
            let mut history = [0.0; 3];
            let mut work = GmresWork {
                x: &mut x,
                best: &mut best,
                candidate: &mut candidate,
                previous: &mut previous,
                residual: &mut residual,
                product: &mut product,
                preconditioned: &mut preconditioned,
                intermediate: &mut intermediate,
                basis: &mut basis,
                hessenberg: &mut hessenberg,
                cosines: &mut cosines,
                sines: &mut sines,
                rotated_rhs: &mut rotated_rhs,
                coefficients: &mut coefficients,
                history: &mut history,
            };
            let outcome = futures::executor::block_on(solve(
                &rhs,
                exact_start.then_some(initial.as_slice()),
                GmresParameters {
                    restart: 1,
                    max_outer: 2,
                    tolerance,
                },
                &mut work,
                &mut Identity(coefficient),
            ))
            .unwrap();
            assert_eq!(outcome.flag, 0);
            // Independent native diagonal-pencil residual, scaled before
            // squaring. Tiny subnormal A=b admits x=1 without forming 1/A.
            let scale = value.re.abs().max(value.im.abs());
            let mut residual = 0.0f64;
            let mut reference = 0.0f64;
            for (&b, &x) in rhs.iter().zip(x.iter()) {
                let b = b.to_complex();
                let ax = T::from_complex(x.to_complex() * coefficient)
                    .unwrap()
                    .to_complex();
                let r = (b - ax) / scale;
                let normalized = b / scale;
                residual += r.re * r.re + r.im * r.im;
                reference += normalized.re * normalized.re + normalized.im * normalized.im;
            }
            let relative = (residual / reference).sqrt();
            assert!(relative <= tolerance);
            assert!((outcome.relres - relative).abs() <= 1e-12);
            if exact_start {
                assert_eq!(
                    (outcome.outer, outcome.inner, outcome.residual_count),
                    (0, 0, 1)
                );
                assert_eq!(history[0], 0.0);
            } else {
                assert!(outcome.residual_count >= 2);
                if value.re == 1e308 {
                    assert!(
                        history[0].is_infinite(),
                        "unrepresentable absolute norm is not a solver breakdown"
                    );
                }
            }
        }
    }
    check::<f32>(Complex64::new(2e38, 0.0), 1.0);
    check::<f64>(Complex64::new(1e308, 0.0), 1.0);
    check::<ComplexElement<f32>>(Complex64::new(2e38, 2e38), 1.0);
    check::<ComplexElement<f64>>(Complex64::new(1e308, 1e308), 1.0);
    check::<f64>(Complex64::new(1e-309, 0.0), 1e-309);
    check::<ComplexElement<f64>>(Complex64::new(1e-309, 1e-309), 1e-309);
}
