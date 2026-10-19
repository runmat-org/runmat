//! First gain-drop crossing for a zero-delay SISO transfer function.
//!
//! This is not registered separately: the existing matrix `bandwidth`
//! binding must select the transfer-function overload without disturbing its
//! matrix/selector semantics.
//! The current RunMat SISO-tf form requires finite, nonzero DC gain. Zero,
//! infinite, and removable-but-exactly-singular DC models are typed unsupported
//! cases; no MATLAB parity is claimed for those models.

mod crossing;
mod polynomial;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use runmat_value::Value;

use super::{model, require_host_owner, FrequencyApi};
use crate::builtins::control::tf_model::{control_error, TfSampleDomain};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const NAME: &str = "bandwidth";

fn unsupported(message: impl Into<String>) -> crate::RuntimeError {
    control_error(NAME, "RunMat:bandwidth:UnsupportedModel", message)
}

pub(crate) async fn bandwidth(
    runtime: Option<&RuntimeContext>,
    system: Value,
    drop_db: Option<&Value>,
) -> BuiltinResult<Value> {
    require_host_owner(FrequencyApi::Bandwidth, runtime)?;
    let drop_db = match drop_db {
        None => -3.0,
        Some(Value::Num(value)) if value.is_finite() && *value < 0.0 => *value,
        _ => {
            return Err(control_error(
                NAME,
                "RunMat:bandwidth:InvalidArgument",
                "bandwidth(sys,dbdrop) requires a finite negative dB drop",
            ))
        }
    };
    let ratio = 10.0_f64.powf(drop_db / 20.0);
    if ratio == 0.0 {
        return Err(control_error(
            NAME,
            "RunMat:bandwidth:InvalidArgument",
            "bandwidth: requested gain drop is below representable magnitude",
        ));
    }
    let model = model(FrequencyApi::Bandwidth, system, runtime).await?;
    let period = match model.sample_domain() {
        TfSampleDomain::Continuous => None,
        TfSampleDomain::Discrete(period) => Some(period.value()),
        TfSampleDomain::DiscreteUnspecified => {
            return Err(unsupported(
                "bandwidth: physical frequency requires a positive sample time",
            ))
        }
    };
    let physical = model.physical_polynomials();
    let dc_point = if period.is_some() {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    };
    let dc = physical
        .evaluate(dc_point)
        .ok_or_else(|| unsupported("bandwidth: the model requires a finite nonzero DC gain"))?;
    let dc_gain = dc.norm();
    if !dc_gain.is_finite() || dc_gain == 0.0 {
        return Err(unsupported(
            "bandwidth: the model requires a finite nonzero DC gain",
        ));
    }
    let equation = crossing::equation(
        physical.numerator(),
        physical.denominator(),
        period.is_some(),
        ratio,
        runtime,
    )?;
    let Some(bracket) = equation.first_positive_root(runtime)? else {
        return Ok(Value::Num(f64::INFINITY));
    };
    let threshold_log = dc_gain.ln() + ratio.ln();
    let lower = bracket.lower.to_f64().unwrap_or(0.0);
    let mut candidate = bracket.upper.to_f64().ok_or_else(|| {
        unsupported("bandwidth: crossing is outside representable frequency range")
    })?;
    if !candidate.is_finite() || candidate < lower {
        return Err(unsupported(
            "bandwidth: crossing is outside representable frequency range",
        ));
    }
    if polynomial::binary(candidate)? <= bracket.upper {
        candidate = candidate.next_up();
    }
    // `to_f64` rounds to nearest. Move just above the exact first root and
    // verify this represented point still lies before any second root.
    for _ in 0..64 {
        let represented = polynomial::binary(candidate)?;
        if represented > bracket.upper && equation.sign_at(&represented, runtime)? < 0 {
            let response = physical
                .evaluate(physical_point(candidate, period))
                .ok_or_else(|| unsupported("bandwidth: crossing meets a singular response"))?;
            let magnitude = response.norm();
            if magnitude.is_finite() && (magnitude == 0.0 || magnitude.ln() < threshold_log) {
                return Ok(Value::Num(physical_frequency(candidate, period)?));
            }
        }
        candidate = candidate.next_up();
        if !candidate.is_finite() {
            break;
        }
    }
    Err(unsupported(
        "bandwidth: physical crossing cannot be certified at binary64 precision",
    ))
}

fn physical_point(parameter: f64, period: Option<f64>) -> Complex64 {
    if period.is_some() {
        let angle = 2.0 * parameter.atan();
        Complex64::new(angle.cos(), angle.sin())
    } else {
        Complex64::new(0.0, parameter)
    }
}

fn physical_frequency(parameter: f64, period: Option<f64>) -> BuiltinResult<f64> {
    let frequency = match period {
        None => parameter,
        Some(sample_time) => 2.0 * parameter.atan() / sample_time,
    };
    if !frequency.is_finite() {
        return Err(unsupported(
            "bandwidth: crossing frequency is not representable",
        ));
    }
    Ok(frequency)
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use num_traits::ToPrimitive;

    use super::{crossing, polynomial};

    fn real(value: f64) -> Complex64 {
        Complex64::new(value, 0.0)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn algebraic_crossing_finds_narrow_notch_and_discrete_lowpass() {
        // A fixed 200-point logarithmic grid can miss this notch entirely.
        let notch = crossing::equation(
            &[real(1.0), real(2.0e-4), real(1.0)],
            &[real(1.0), real(2.0e-2), real(1.0)],
            false,
            10.0_f64.powf(-3.0 / 20.0),
            None,
        )
        .unwrap();
        let first = notch.first_positive_root(None).unwrap().unwrap();
        let frequency = first.upper.to_f64().unwrap();
        assert!(frequency > 0.9 && frequency < 1.0);
        assert_eq!(
            notch
                .sign_at(&polynomial::binary(1.0).unwrap(), None)
                .unwrap(),
            -1
        );
        let point = Complex64::new(0.0, frequency);
        let numerator = point * point + real(2.0e-4) * point + real(1.0);
        let denominator = point * point + real(2.0e-2) * point + real(1.0);
        let ratio = (numerator / denominator).norm();
        assert!((ratio - 10.0_f64.powf(-3.0 / 20.0)).abs() < 1.0e-8);

        // H(z)=(1-a)/(z-a), a=1/2: unit DC gain and a finite Nyquist drop.
        let discrete = crossing::equation(
            &[real(0.5)],
            &[real(1.0), real(-0.5)],
            true,
            10.0_f64.powf(-3.0 / 20.0),
            None,
        )
        .unwrap();
        let first = discrete.first_positive_root(None).unwrap().unwrap();
        let parameter = first.upper.to_f64().unwrap();
        assert!(parameter > 0.3 && parameter < 0.5);
        assert_eq!(
            discrete
                .sign_at(&polynomial::binary(parameter * 2.0).unwrap(), None)
                .unwrap(),
            -1
        );
        let angle = 2.0 * parameter.atan();
        let point = Complex64::new(angle.cos(), angle.sin());
        let ratio = (real(0.5) / (point - real(0.5))).norm();
        assert!((ratio - 10.0_f64.powf(-3.0 / 20.0)).abs() < 1.0e-10);

        let constant = crossing::equation(
            &[real(1.0)],
            &[real(1.0)],
            false,
            10.0_f64.powf(-3.0 / 20.0),
            None,
        )
        .unwrap();
        assert!(constant.first_positive_root(None).unwrap().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn represented_large_crossing_is_not_false_infinity() {
        let one = polynomial::binary(1.0).unwrap();
        let tiny = polynomial::binary(1.0e-308).unwrap();
        let equation =
            polynomial::CrossingPolynomial::new(vec![one.clone(), &one - &tiny, -tiny]).unwrap();
        let first = equation.first_positive_root(None).unwrap().unwrap();
        let frequency = first.upper.to_f64().unwrap();
        assert!(frequency.is_finite() && frequency > 9.0e307);

        // A removable squared factor at x=1 is a contact, not a -3dB drop;
        // the first actual descending crossing is the simple root at x=2.
        let contact_then_drop = polynomial::CrossingPolynomial::new(
            [2.0, -5.0, 4.0, -1.0]
                .into_iter()
                .map(|value| polynomial::binary(value).unwrap())
                .collect(),
        )
        .unwrap();
        let first = contact_then_drop
            .first_positive_root(None)
            .unwrap()
            .unwrap();
        assert!(first.lower <= polynomial::binary(2.0).unwrap());
        assert!(first.upper >= polynomial::binary(2.0).unwrap());
        assert_eq!(
            contact_then_drop
                .sign_at(&polynomial::binary(3.0).unwrap(), None)
                .unwrap(),
            -1
        );

        let tangent_only = polynomial::CrossingPolynomial::new(
            [1.0, -2.0, 1.0]
                .into_iter()
                .map(|value| polynomial::binary(value).unwrap())
                .collect(),
        )
        .unwrap();
        assert!(tangent_only.first_positive_root(None).unwrap().is_none());
    }
}
