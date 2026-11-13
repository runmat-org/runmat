//! Generalized-Newtonian (inelastic, shear-thinning or yield-stress) viscosity laws: the
//! apparent viscosity μ(γ̇) of the shear rate γ̇ = √(2 D:D), D the rate of strain.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeneralizedNewtonianLaw {
    /// μ = K γ̇^(n−1), with γ̇ held at or above `minimum_shear_rate_per_s` (the low-shear
    /// plateau that keeps a shear-thinning fluid's viscosity finite at rest).
    PowerLaw {
        consistency_pa_s_n: f64,
        power_index: f64,
        minimum_shear_rate_per_s: f64,
    },
    /// Carreau–Yasuda: μ = μ∞ + (μ₀ − μ∞)[1 + (λγ̇)^a]^((n−1)/a); a = 2 is Carreau.
    CarreauYasuda {
        zero_shear_viscosity_pa_s: f64,
        infinite_shear_viscosity_pa_s: f64,
        relaxation_time_s: f64,
        power_index: f64,
        transition_exponent: f64,
    },
    /// Cross: μ = μ∞ + (μ₀ − μ∞)/(1 + (λγ̇)^m).
    Cross {
        zero_shear_viscosity_pa_s: f64,
        infinite_shear_viscosity_pa_s: f64,
        time_constant_s: f64,
        rate_index: f64,
    },
    /// Herschel–Bulkley with Papanastasiou regularization (Bingham for n = 1):
    /// μ = τ_y (1 − e^(−mγ̇))/γ̇ + K γ̇^(n−1).
    HerschelBulkley {
        yield_stress_pa: f64,
        consistency_pa_s_n: f64,
        power_index: f64,
        regularization_s: f64,
        minimum_shear_rate_per_s: f64,
    },
    /// Casson with Papanastasiou regularization: √μ = √(τ_y(1 − e^(−mγ̇))/γ̇) + √μ_p.
    Casson {
        yield_stress_pa: f64,
        plastic_viscosity_pa_s: f64,
        regularization_s: f64,
        minimum_shear_rate_per_s: f64,
    },
}

impl GeneralizedNewtonianLaw {
    pub fn is_valid(&self) -> bool {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
        match *self {
            Self::PowerLaw {
                consistency_pa_s_n,
                power_index,
                minimum_shear_rate_per_s,
            } => {
                positive(consistency_pa_s_n)
                    && positive(power_index)
                    && positive(minimum_shear_rate_per_s)
            }
            Self::CarreauYasuda {
                zero_shear_viscosity_pa_s,
                infinite_shear_viscosity_pa_s,
                relaxation_time_s,
                power_index,
                transition_exponent,
            } => {
                positive(zero_shear_viscosity_pa_s)
                    && nonnegative(infinite_shear_viscosity_pa_s)
                    && infinite_shear_viscosity_pa_s <= zero_shear_viscosity_pa_s
                    && nonnegative(relaxation_time_s)
                    && positive(power_index)
                    && positive(transition_exponent)
            }
            Self::Cross {
                zero_shear_viscosity_pa_s,
                infinite_shear_viscosity_pa_s,
                time_constant_s,
                rate_index,
            } => {
                positive(zero_shear_viscosity_pa_s)
                    && nonnegative(infinite_shear_viscosity_pa_s)
                    && infinite_shear_viscosity_pa_s <= zero_shear_viscosity_pa_s
                    && nonnegative(time_constant_s)
                    && positive(rate_index)
            }
            Self::HerschelBulkley {
                yield_stress_pa,
                consistency_pa_s_n,
                power_index,
                regularization_s,
                minimum_shear_rate_per_s,
            } => {
                nonnegative(yield_stress_pa)
                    && positive(consistency_pa_s_n)
                    && positive(power_index)
                    && positive(regularization_s)
                    && positive(minimum_shear_rate_per_s)
            }
            Self::Casson {
                yield_stress_pa,
                plastic_viscosity_pa_s,
                regularization_s,
                minimum_shear_rate_per_s,
            } => {
                nonnegative(yield_stress_pa)
                    && positive(plastic_viscosity_pa_s)
                    && positive(regularization_s)
                    && positive(minimum_shear_rate_per_s)
            }
        }
    }

    /// Apparent viscosity in Pa·s at shear rate `shear_rate_per_s` (≥ 0).
    pub fn viscosity(&self, shear_rate_per_s: f64) -> f64 {
        let rate = shear_rate_per_s.max(0.0);
        match *self {
            Self::PowerLaw {
                consistency_pa_s_n,
                power_index,
                minimum_shear_rate_per_s,
            } => consistency_pa_s_n * rate.max(minimum_shear_rate_per_s).powf(power_index - 1.0),
            Self::CarreauYasuda {
                zero_shear_viscosity_pa_s: mu0,
                infinite_shear_viscosity_pa_s: mu_inf,
                relaxation_time_s: lambda,
                power_index: n,
                transition_exponent: a,
            } => mu_inf + (mu0 - mu_inf) * (1.0 + (lambda * rate).powf(a)).powf((n - 1.0) / a),
            Self::Cross {
                zero_shear_viscosity_pa_s: mu0,
                infinite_shear_viscosity_pa_s: mu_inf,
                time_constant_s: lambda,
                rate_index: m,
            } => mu_inf + (mu0 - mu_inf) / (1.0 + (lambda * rate).powf(m)),
            Self::HerschelBulkley {
                yield_stress_pa,
                consistency_pa_s_n,
                power_index,
                regularization_s,
                minimum_shear_rate_per_s,
            } => {
                let rate = rate.max(minimum_shear_rate_per_s);
                yield_stress_pa * yield_ramp(regularization_s, rate)
                    + consistency_pa_s_n * rate.powf(power_index - 1.0)
            }
            Self::Casson {
                yield_stress_pa,
                plastic_viscosity_pa_s,
                regularization_s,
                minimum_shear_rate_per_s,
            } => {
                let rate = rate.max(minimum_shear_rate_per_s);
                ((yield_stress_pa * yield_ramp(regularization_s, rate)).sqrt()
                    + plastic_viscosity_pa_s.sqrt())
                .powi(2)
            }
        }
    }

    /// The low-shear (largest) viscosity of the law: its zero-shear plateau, or its value at
    /// the minimum shear rate. Used as the fluid's reference viscosity.
    pub fn low_shear_viscosity(&self) -> f64 {
        match *self {
            Self::CarreauYasuda {
                zero_shear_viscosity_pa_s,
                ..
            }
            | Self::Cross {
                zero_shear_viscosity_pa_s,
                ..
            } => zero_shear_viscosity_pa_s,
            Self::PowerLaw {
                minimum_shear_rate_per_s,
                power_index,
                ..
            } if power_index < 1.0 => self.viscosity(minimum_shear_rate_per_s),
            Self::PowerLaw { .. } => self.viscosity(1.0),
            Self::HerschelBulkley {
                minimum_shear_rate_per_s,
                ..
            }
            | Self::Casson {
                minimum_shear_rate_per_s,
                ..
            } => self.viscosity(minimum_shear_rate_per_s),
        }
    }
}

/// Papanastasiou (1 − e^(−mγ̇))/γ̇, which tends to m as γ̇ → 0.
fn yield_ramp(regularization_s: f64, rate: f64) -> f64 {
    let x = regularization_s * rate;
    if x < 1e-8 {
        regularization_s * (1.0 - 0.5 * x)
    } else {
        -(-x).exp_m1() / rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laws_reduce_to_their_limits() {
        let carreau = GeneralizedNewtonianLaw::CarreauYasuda {
            zero_shear_viscosity_pa_s: 0.056,
            infinite_shear_viscosity_pa_s: 0.00345,
            relaxation_time_s: 3.313,
            power_index: 0.3568,
            transition_exponent: 2.0,
        };
        assert!((carreau.viscosity(0.0) - 0.056).abs() < 1e-15);
        assert!((carreau.viscosity(1e9) - 0.00345).abs() < 1e-4);
        let power = GeneralizedNewtonianLaw::PowerLaw {
            consistency_pa_s_n: 12.0,
            power_index: 0.45,
            minimum_shear_rate_per_s: 1e-3,
        };
        assert!((power.viscosity(15.7) - 12.0 * 15.7f64.powf(-0.55)).abs() < 1e-12);
        assert_eq!(power.viscosity(0.0), power.viscosity(1e-3));
        // Bingham: τ = μγ̇ → τ_y + Kγ̇ well above the regularization scale.
        let bingham = GeneralizedNewtonianLaw::HerschelBulkley {
            yield_stress_pa: 10.0,
            consistency_pa_s_n: 0.1,
            power_index: 1.0,
            regularization_s: 1000.0,
            minimum_shear_rate_per_s: 1e-6,
        };
        let rate = 5.0;
        assert!((bingham.viscosity(rate) * rate - (10.0 + 0.1 * rate)).abs() < 1e-9);
        assert!(carreau.is_valid() && power.is_valid() && bingham.is_valid());
    }
}
