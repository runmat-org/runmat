#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Complex {
    pub real: f64,
    pub imaginary: f64,
}

impl Complex {
    pub(super) const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    pub(super) const fn conjugate(self) -> Self {
        Self::new(self.real, -self.imaginary)
    }

    pub(super) fn norm(self) -> f64 {
        self.real.hypot(self.imaginary)
    }

    pub(super) fn scale(self, factor: f64) -> Self {
        Self::new(self.real * factor, self.imaginary * factor)
    }

    pub(super) fn reciprocal(self) -> Option<Self> {
        let denominator = self.real * self.real + self.imaginary * self.imaginary;
        if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
            return None;
        }
        Some(Self::new(
            self.real / denominator,
            -self.imaginary / denominator,
        ))
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, right: Self) -> Self::Output {
        Self::new(self.real + right.real, self.imaginary + right.imaginary)
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, right: Self) {
        self.real += right.real;
        self.imaginary += right.imaginary;
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, right: Self) -> Self::Output {
        Self::new(
            self.real * right.real - self.imaginary * right.imaginary,
            self.real * right.imaginary + self.imaginary * right.real,
        )
    }
}
