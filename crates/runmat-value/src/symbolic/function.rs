pub(super) const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicFunction {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Heaviside,
    Dirac,
    DiracDerivative(u32),
}

impl SymbolicFunction {
    pub fn name(self) -> &'static str {
        match self {
            SymbolicFunction::Sin => "sin",
            SymbolicFunction::Cos => "cos",
            SymbolicFunction::Tan => "tan",
            SymbolicFunction::Exp => "exp",
            SymbolicFunction::Log => "log",
            SymbolicFunction::Sqrt => "sqrt",
            SymbolicFunction::Heaviside => "heaviside",
            SymbolicFunction::Dirac => "dirac",
            SymbolicFunction::DiracDerivative(_) => "dirac",
        }
    }

    pub(super) fn apply_numeric_constant(self, value: f64) -> Option<f64> {
        let result = match self {
            SymbolicFunction::Sin => value.sin(),
            SymbolicFunction::Cos => value.cos(),
            SymbolicFunction::Tan => value.tan(),
            SymbolicFunction::Exp => value.exp(),
            SymbolicFunction::Log => value.ln(),
            SymbolicFunction::Sqrt => value.sqrt(),
            SymbolicFunction::Heaviside if value > 0.0 => 1.0,
            SymbolicFunction::Heaviside if value < 0.0 => 0.0,
            SymbolicFunction::Heaviside if value == 0.0 => 0.5,
            SymbolicFunction::Heaviside => value,
            SymbolicFunction::Dirac if value == 0.0 => f64::INFINITY,
            SymbolicFunction::Dirac => 0.0,
            SymbolicFunction::DiracDerivative(_) if value == 0.0 => f64::INFINITY,
            SymbolicFunction::DiracDerivative(_) => 0.0,
        };
        (!result.is_nan()).then_some(result)
    }

    pub(super) fn apply_constant(self, value: f64) -> Option<f64> {
        self.apply_numeric_constant(value)
            .filter(|result| result.is_finite())
    }
}
