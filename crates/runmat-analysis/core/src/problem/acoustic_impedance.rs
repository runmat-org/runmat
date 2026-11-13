//! Frequency-dependent specific acoustic impedance laws, in the e^{+iωt} phasor convention
//! of the acoustic solvers: p = Z v_n with v_n the outward normal particle velocity.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "law", deny_unknown_fields)]
pub enum AcousticImpedanceLaw {
    /// Tabulated resistance R and reactance X (Pa·s/m) at ascending frequencies, linear in
    /// frequency between samples and constant beyond the first and last.
    Table {
        frequencies_hz: Vec<f64>,
        resistance_pa_s_per_m: Vec<f64>,
        reactance_pa_s_per_m: Vec<f64>,
    },
    /// Rigidly backed porous layer of thickness d: Delany–Bazley characteristic impedance
    /// Z_c and wavenumber k_c from the flow resistivity σ, in the adjacent fluid (ρ₀, c₀);
    /// surface impedance Z = −i Z_c cot(k_c d).
    DelanyBazley {
        flow_resistivity_pa_s_per_m2: f64,
        thickness_m: f64,
    },
}

impl AcousticImpedanceLaw {
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Table {
                frequencies_hz,
                resistance_pa_s_per_m,
                reactance_pa_s_per_m,
            } => {
                !frequencies_hz.is_empty()
                    && frequencies_hz.len() == resistance_pa_s_per_m.len()
                    && frequencies_hz.len() == reactance_pa_s_per_m.len()
                    && frequencies_hz.iter().all(|f| f.is_finite() && *f > 0.0)
                    && frequencies_hz.windows(2).all(|pair| pair[0] < pair[1])
                    // Passive: nonnegative resistance; a lossless wall has R = 0 and X ≠ 0.
                    && resistance_pa_s_per_m.iter().all(|r| r.is_finite() && *r >= 0.0)
                    && reactance_pa_s_per_m.iter().all(|x| x.is_finite())
                    && resistance_pa_s_per_m
                        .iter()
                        .zip(reactance_pa_s_per_m)
                        .all(|(r, x)| r.hypot(*x) > 0.0)
            }
            Self::DelanyBazley {
                flow_resistivity_pa_s_per_m2,
                thickness_m,
            } => {
                flow_resistivity_pa_s_per_m2.is_finite()
                    && *flow_resistivity_pa_s_per_m2 > 0.0
                    && thickness_m.is_finite()
                    && *thickness_m > 0.0
            }
        }
    }

    /// Z = (R, X) at `frequency_hz` facing a fluid of density `density` and sound speed
    /// `sound_speed`; `None` for an invalid law or a nonfinite result.
    pub fn evaluate(
        &self,
        frequency_hz: f64,
        density: f64,
        sound_speed: f64,
    ) -> Option<(f64, f64)> {
        if !self.is_valid() || !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return None;
        }
        let z = match self {
            Self::Table {
                frequencies_hz,
                resistance_pa_s_per_m,
                reactance_pa_s_per_m,
            } => {
                let last = frequencies_hz.len() - 1;
                let at = |i: usize| (resistance_pa_s_per_m[i], reactance_pa_s_per_m[i]);
                if frequency_hz <= frequencies_hz[0] {
                    at(0)
                } else if frequency_hz >= frequencies_hz[last] {
                    at(last)
                } else {
                    let upper = frequencies_hz.partition_point(|f| *f <= frequency_hz);
                    let (f0, f1) = (frequencies_hz[upper - 1], frequencies_hz[upper]);
                    let t = (frequency_hz - f0) / (f1 - f0);
                    let (a, b) = (at(upper - 1), at(upper));
                    (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
                }
            }
            Self::DelanyBazley {
                flow_resistivity_pa_s_per_m2,
                thickness_m,
            } => {
                if !(density.is_finite() && density > 0.0 && sound_speed.is_finite())
                    || sound_speed <= 0.0
                {
                    return None;
                }
                let x = density * frequency_hz / flow_resistivity_pa_s_per_m2;
                let zc = Complex(
                    density * sound_speed * (1.0 + 0.0571 * x.powf(-0.754)),
                    -density * sound_speed * 0.087 * x.powf(-0.732),
                );
                let k0 = std::f64::consts::TAU * frequency_hz / sound_speed;
                let kc = Complex(
                    k0 * (1.0 + 0.0978 * x.powf(-0.700)),
                    -k0 * 0.189 * x.powf(-0.595),
                );
                let argument = Complex(kc.0 * thickness_m, kc.1 * thickness_m);
                let cot = argument.cos().div(argument.sin());
                // −i Z_c cot(k_c d)
                let product = zc.mul(cot);
                (product.1, -product.0)
            }
        };
        (z.0.is_finite() && z.1.is_finite() && z.0.hypot(z.1) > 0.0).then_some(z)
    }
}

#[derive(Clone, Copy)]
struct Complex(f64, f64);
impl Complex {
    fn mul(self, other: Self) -> Self {
        Self(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }
    fn div(self, other: Self) -> Self {
        let d = other.0 * other.0 + other.1 * other.1;
        Self(
            (self.0 * other.0 + self.1 * other.1) / d,
            (self.1 * other.0 - self.0 * other.1) / d,
        )
    }
    fn cos(self) -> Self {
        Self(self.0.cos() * self.1.cosh(), -self.0.sin() * self.1.sinh())
    }
    fn sin(self) -> Self {
        Self(self.0.sin() * self.1.cosh(), self.0.cos() * self.1.sinh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_interpolates_linearly_and_holds_its_ends() {
        let law = AcousticImpedanceLaw::Table {
            frequencies_hz: vec![100.0, 200.0],
            resistance_pa_s_per_m: vec![400.0, 600.0],
            reactance_pa_s_per_m: vec![-100.0, 100.0],
        };
        assert_eq!(law.evaluate(150.0, 1.2, 343.0), Some((500.0, 0.0)));
        assert_eq!(law.evaluate(50.0, 1.2, 343.0), Some((400.0, -100.0)));
        assert_eq!(law.evaluate(500.0, 1.2, 343.0), Some((600.0, 100.0)));
    }

    #[test]
    fn delany_bazley_layer_is_passive_and_absorbs_more_at_high_frequency() {
        // 50 mm, σ = 20 kPa·s/m²: absorption α = 1 − |(Z − ρc)/(Z + ρc)|² rises with frequency.
        let law = AcousticImpedanceLaw::DelanyBazley {
            flow_resistivity_pa_s_per_m2: 20_000.0,
            thickness_m: 0.05,
        };
        let (rho, c) = (1.204, 343.0);
        let alpha = |f: f64| {
            let (r, x) = law.evaluate(f, rho, c).unwrap();
            let num = (r - rho * c).hypot(x);
            let den = (r + rho * c).hypot(x);
            1.0 - (num / den).powi(2)
        };
        let (r, x) = law.evaluate(125.0, rho, c).unwrap();
        assert!(r > 0.0, "passive layer");
        // A thin rigidly backed layer is stiffness-like (negative reactance) at low frequency.
        assert!(x < 0.0);
        assert!(alpha(125.0) < alpha(500.0) && alpha(500.0) < alpha(1000.0));
        assert!(alpha(1000.0) > 0.6 && alpha(125.0) < 0.4);
    }
}
