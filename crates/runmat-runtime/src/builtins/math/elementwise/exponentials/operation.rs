use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, EXPM1_CHARACTER_INPUT_EXTENSION,
    EXPM1_ERROR_INTERNAL, EXPM1_ERROR_INVALID_INPUT, EXPM1_INTEGER_INPUT_EXTENSION,
    EXPM1_LOGICAL_INPUT_EXTENSION, EXP_CHARACTER_INPUT_EXTENSION, EXP_ERROR_INTERNAL,
    EXP_ERROR_INVALID_INPUT, EXP_INTEGER_INPUT_EXTENSION, EXP_LOGICAL_INPUT_EXTENSION,
};
use runmat_value::SymbolicFunction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExponentialOperation {
    Exp,
    Expm1,
}

impl ExponentialOperation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Expm1 => "expm1",
        }
    }

    pub(super) const fn invalid_input(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Exp => &EXP_ERROR_INVALID_INPUT,
            Self::Expm1 => &EXPM1_ERROR_INVALID_INPUT,
        }
    }

    pub(super) const fn internal_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Exp => &EXP_ERROR_INTERNAL,
            Self::Expm1 => &EXPM1_ERROR_INTERNAL,
        }
    }

    pub(super) const fn integer_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Exp => &EXP_INTEGER_INPUT_EXTENSION,
            Self::Expm1 => &EXPM1_INTEGER_INPUT_EXTENSION,
        }
    }

    pub(super) const fn logical_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Exp => &EXP_LOGICAL_INPUT_EXTENSION,
            Self::Expm1 => &EXPM1_LOGICAL_INPUT_EXTENSION,
        }
    }

    pub(super) const fn character_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Exp => &EXP_CHARACTER_INPUT_EXTENSION,
            Self::Expm1 => &EXPM1_CHARACTER_INPUT_EXTENSION,
        }
    }

    pub(super) const fn symbolic_function(self) -> Option<SymbolicFunction> {
        match self {
            Self::Exp => Some(SymbolicFunction::Exp),
            Self::Expm1 => None,
        }
    }

    pub(super) const fn preserves_sparse_zeros(self) -> bool {
        matches!(self, Self::Expm1)
    }

    pub(super) fn apply_f64(self, value: f64) -> f64 {
        match self {
            Self::Exp => value.exp(),
            Self::Expm1 => value.exp_m1(),
        }
    }

    pub(super) fn apply_f32(self, value: f32) -> f32 {
        match self {
            Self::Exp => value.exp(),
            Self::Expm1 => value.exp_m1(),
        }
    }

    pub(super) fn apply_complex_f64(self, real: f64, imag: f64) -> (f64, f64) {
        match self {
            Self::Exp => exp_complex(real, imag),
            Self::Expm1 => expm1_complex(real, imag),
        }
    }

    pub(super) fn apply_complex_f32(self, real: f32, imag: f32) -> (f32, f32) {
        match self {
            Self::Exp => exp_complex(real, imag),
            Self::Expm1 => expm1_complex(real, imag),
        }
    }
}

fn exp_complex<T>(real: T, imag: T) -> (T, T)
where
    T: ExponentialFloat,
{
    if imag == T::ZERO {
        return (real.exp(), imag);
    }
    let exp_real = real.exp();
    (exp_real * imag.cos(), exp_real * imag.sin())
}

fn expm1_complex<T>(real: T, imag: T) -> (T, T)
where
    T: ExponentialFloat,
{
    if imag == T::ZERO {
        return (real.exp_m1(), imag);
    }
    if !real.is_finite() || !imag.is_finite() {
        let exp_real = real.exp();
        return (exp_real * imag.cos() - T::ONE, exp_real * imag.sin());
    }
    let half = T::HALF * imag;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let cos_imag_minus_one = -T::TWO * sin_half * sin_half;
    let sin_imag = T::TWO * sin_half * cos_half;
    let expm1_real = real.exp_m1();
    let exp_real = expm1_real + T::ONE;
    (
        expm1_real + exp_real * cos_imag_minus_one,
        exp_real * sin_imag,
    )
}

trait ExponentialFloat:
    Copy
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;

    fn exp(self) -> Self;
    fn exp_m1(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn is_finite(self) -> bool;
}

macro_rules! impl_exponential_float {
    ($type:ty) => {
        impl ExponentialFloat for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const HALF: Self = 0.5;

            fn exp(self) -> Self {
                self.exp()
            }
            fn exp_m1(self) -> Self {
                self.exp_m1()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}

impl_exponential_float!(f32);
impl_exponential_float!(f64);
