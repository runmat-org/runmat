//! Independent real and imaginary response scales for real-basis least squares.

use super::{solver::stable_l2, Complex64, PolyfitKernelError, PolyfitKernelResult};

#[derive(Clone, Copy)]
pub(super) struct ResponseScale {
    pub real: f64,
    pub imaginary: f64,
}

impl ResponseScale {
    pub fn from_values(values: &[Complex64]) -> Self {
        let magnitude = |component: fn(&Complex64) -> f64| {
            let largest = values
                .iter()
                .map(component)
                .map(f64::abs)
                .fold(0.0, f64::max);
            if largest == 0.0 {
                1.0
            } else {
                largest
            }
        };
        Self {
            real: magnitude(|value| value.re),
            imaginary: magnitude(|value| value.im),
        }
    }

    pub fn normalize(self, value: &Complex64) -> Complex64 {
        Complex64::new(value.re / self.real, value.im / self.imaginary)
    }

    pub fn residual_norm(self, residuals: &[Complex64]) -> PolyfitKernelResult<f64> {
        let norm = stable_l2(
            residuals
                .iter()
                .flat_map(|value| [value.re * self.real, value.im * self.imaginary]),
        );
        if norm.is_finite() {
            Ok(norm)
        } else {
            Err(PolyfitKernelError::input(
                "polyfit: requested residual norm exceeds finite output range",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::ProviderPolyfitOutputs;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn constant_complex_fit_preserves_independent_component_ranges() {
        for observation in [
            Complex64::new(f64::MAX, 1e-300),
            Complex64::new(1e-300, f64::MAX),
        ] {
            let solved = super::super::solver::solve_polyfit(
                &[0.0],
                &[observation],
                0,
                None,
                ProviderPolyfitOutputs::Coefficients,
            )
            .unwrap();
            assert_eq!(solved.coeffs, vec![observation]);
            assert!(solved.is_complex);
            assert!(solved.normr.is_none());
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn residual_norm_restores_each_component_before_combining() {
        let scale = ResponseScale {
            real: 3.0,
            imaginary: 4.0,
        };
        assert_eq!(
            scale.residual_norm(&[Complex64::new(1.0, 1.0)]).unwrap(),
            5.0
        );
        let scale = ResponseScale {
            real: f64::MAX,
            imaginary: f64::MAX,
        };
        assert!(scale.residual_norm(&[Complex64::new(1.0, 1.0)]).is_err());
    }
}
