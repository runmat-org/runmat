//! Native arithmetic policy for the value-free shared GMRES core.
use num_complex::Complex64;
use runmat_value::ComplexElement;

pub(super) struct Native<T>(std::marker::PhantomData<T>);
impl<T: GmresScalar> runmat_sparse_solvers::krylov::gmres::Arithmetic<T> for Native<T> {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain = T::DOMAIN;
    fn zero() -> T {
        T::zero()
    }
    fn to_complex(value: T) -> Complex64 {
        value.to_complex()
    }
    fn from_complex(value: Complex64) -> Option<T> {
        T::from_complex(value)
    }
    fn divide_complex(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64) {
        runmat_value::divide_complex_f64(numerator, denominator)
    }
}
pub(super) trait GmresScalar: Copy + PartialEq {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain;
    fn zero() -> Self;
    fn to_complex(self) -> Complex64;
    fn from_complex(value: Complex64) -> Option<Self>;
}

impl GmresScalar for f32 {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
        runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Real;
    fn zero() -> Self {
        0.0
    }

    fn to_complex(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }

    fn from_complex(value: Complex64) -> Option<Self> {
        (value.im == 0.0 && value.re.is_finite() && value.re.abs() <= f32::MAX as f64)
            .then_some(value.re as f32)
    }
}

impl GmresScalar for f64 {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
        runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Real;
    fn zero() -> Self {
        0.0
    }

    fn to_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }

    fn from_complex(value: Complex64) -> Option<Self> {
        (value.im == 0.0 && value.re.is_finite()).then_some(value.re)
    }
}

impl GmresScalar for ComplexElement<f32> {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
        runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Complex;
    fn zero() -> Self {
        ComplexElement(0.0, 0.0)
    }

    fn to_complex(self) -> Complex64 {
        Complex64::new(f64::from(self.0), f64::from(self.1))
    }

    fn from_complex(value: Complex64) -> Option<Self> {
        (value.re.is_finite()
            && value.im.is_finite()
            && value.re.abs() <= f32::MAX as f64
            && value.im.abs() <= f32::MAX as f64)
            .then_some(ComplexElement(value.re as f32, value.im as f32))
    }
}

impl GmresScalar for ComplexElement<f64> {
    const DOMAIN: runmat_sparse_solvers::krylov::usage::ArithmeticDomain =
        runmat_sparse_solvers::krylov::usage::ArithmeticDomain::Complex;
    fn zero() -> Self {
        ComplexElement(0.0, 0.0)
    }

    fn to_complex(self) -> Complex64 {
        Complex64::new(self.0, self.1)
    }

    fn from_complex(value: Complex64) -> Option<Self> {
        (value.re.is_finite() && value.im.is_finite()).then_some(ComplexElement(value.re, value.im))
    }
}
