//! Butterworth numerical design and output adapters.
use super::*;
use num_complex::Complex64;
use runmat_value::ComplexTensor;
use std::f64::consts::PI;

const REAL_TOL: f64 = 1.0e-8;
#[derive(Debug, Clone)]
struct Zpk {
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: Complex64,
}

#[derive(Debug, Clone)]
pub(super) struct ButterDesign {
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: Complex64,
    numerator: Vec<Complex64>,
    denominator: Vec<Complex64>,
}

impl ButterDesign {
    pub(super) fn numerator_value(&self) -> BuiltinResult<Value> {
        row_coefficients_to_value(&self.numerator)
    }

    pub(super) fn denominator_value(&self) -> BuiltinResult<Value> {
        row_coefficients_to_value(&self.denominator)
    }

    pub(super) fn zpk_values(&self) -> BuiltinResult<(Value, Value, Value)> {
        Ok((
            column_complex_to_value(&self.zeros)?,
            column_complex_to_value(&self.poles)?,
            scalar_complex_to_value(self.gain)?,
        ))
    }

    pub(super) async fn state_space_values(
        &self,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<(Value, Value, Value, Value)> {
        let b = real_coefficients(&self.numerator)?;
        let a = real_coefficients(&self.denominator)?;
        let order = a.len().saturating_sub(1);
        cooperation.completed(0).await?;
        let (a_mat, b_mat, c_mat, d_val) = transfer_function_to_state_space(&b, &a)?;
        cooperation
            .completed((order as u64).saturating_mul(order as u64))
            .await?;
        Ok((
            Value::Tensor(a_mat),
            Value::Tensor(b_mat),
            Value::Tensor(c_mat),
            Value::Num(d_val),
        ))
    }
}

pub(super) async fn design_butterworth(
    args: &ButterArgs,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<ButterDesign> {
    let analog = analog_zpk(args)?;
    cooperation.completed(args.order as u64).await?;
    let zpk = if args.analog {
        analog
    } else {
        bilinear_zpk(&analog)?
    };
    cooperation.completed(zpk.poles.len() as u64).await?;
    let numerator = polynomial_from_roots(&zpk.zeros, cooperation)
        .await?
        .into_iter()
        .map(|coeff| coeff * zpk.gain)
        .collect::<Vec<_>>();
    let denominator = polynomial_from_roots(&zpk.poles, cooperation).await?;
    let (numerator, denominator) = normalize_transfer_function(numerator, denominator)?;

    Ok(ButterDesign {
        zeros: zpk.zeros,
        poles: zpk.poles,
        gain: zpk.gain,
        numerator,
        denominator,
    })
}

fn analog_zpk(args: &ButterArgs) -> BuiltinResult<Zpk> {
    let cutoff = if args.analog {
        args.cutoff.clone()
    } else {
        args.cutoff
            .iter()
            .map(|value| (PI * value / 2.0).tan())
            .collect::<Vec<_>>()
    };
    let prototype_poles = butterworth_prototype(args.order);

    let zpk = match args.kind {
        FilterKind::Lowpass => lowpass_zpk(&prototype_poles, cutoff[0], args.order),
        FilterKind::Highpass => highpass_zpk(&prototype_poles, cutoff[0], args.order),
        FilterKind::Bandpass => bandpass_zpk(&prototype_poles, cutoff[0], cutoff[1], args.order),
        FilterKind::Bandstop => bandstop_zpk(&prototype_poles, cutoff[0], cutoff[1], args.order),
    };
    validate_zpk(&zpk)?;
    Ok(zpk)
}

fn butterworth_prototype(order: usize) -> Vec<Complex64> {
    (0..order)
        .map(|idx| {
            let angle = PI * (2.0 * idx as f64 + order as f64 + 1.0) / (2.0 * order as f64);
            Complex64::from_polar(1.0, angle)
        })
        .collect()
}

fn lowpass_zpk(prototype_poles: &[Complex64], cutoff: f64, order: usize) -> Zpk {
    Zpk {
        zeros: Vec::new(),
        poles: prototype_poles.iter().map(|pole| *pole * cutoff).collect(),
        gain: Complex64::new(cutoff.powi(order as i32), 0.0),
    }
}

fn highpass_zpk(prototype_poles: &[Complex64], cutoff: f64, order: usize) -> Zpk {
    Zpk {
        zeros: vec![Complex64::new(0.0, 0.0); order],
        poles: prototype_poles
            .iter()
            .map(|pole| Complex64::new(cutoff, 0.0) / *pole)
            .collect(),
        gain: Complex64::new(1.0, 0.0),
    }
}

fn bandpass_zpk(prototype_poles: &[Complex64], low: f64, high: f64, order: usize) -> Zpk {
    let bandwidth = high - low;
    let center_sq = low * high;
    let mut poles = Vec::with_capacity(order * 2);
    for pole in prototype_poles {
        let scaled = *pole * bandwidth;
        let discriminant = scaled * scaled - Complex64::new(4.0 * center_sq, 0.0);
        let root = discriminant.sqrt();
        poles.push((scaled + root) * 0.5);
        poles.push((scaled - root) * 0.5);
    }
    Zpk {
        zeros: vec![Complex64::new(0.0, 0.0); order],
        poles,
        gain: Complex64::new(bandwidth.powi(order as i32), 0.0),
    }
}

fn bandstop_zpk(prototype_poles: &[Complex64], low: f64, high: f64, order: usize) -> Zpk {
    let bandwidth = high - low;
    let center = (low * high).sqrt();
    let center_sq = low * high;
    let mut poles = Vec::with_capacity(order * 2);
    for pole in prototype_poles {
        let scaled = Complex64::new(bandwidth, 0.0) / *pole;
        let discriminant = scaled * scaled - Complex64::new(4.0 * center_sq, 0.0);
        let root = discriminant.sqrt();
        poles.push((scaled + root) * 0.5);
        poles.push((scaled - root) * 0.5);
    }
    let mut zeros = Vec::with_capacity(order * 2);
    for _ in 0..order {
        zeros.push(Complex64::new(0.0, center));
        zeros.push(Complex64::new(0.0, -center));
    }
    Zpk {
        zeros,
        poles,
        gain: Complex64::new(1.0, 0.0),
    }
}

fn validate_zpk(zpk: &Zpk) -> BuiltinResult<()> {
    if !complex_is_finite(zpk.gain)
        || zpk.zeros.iter().any(|value| !complex_is_finite(*value))
        || zpk.poles.iter().any(|value| !complex_is_finite(*value))
    {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "filter design produced non-finite roots or gain",
        ));
    }
    Ok(())
}

fn bilinear_zpk(analog: &Zpk) -> BuiltinResult<Zpk> {
    let one = Complex64::new(1.0, 0.0);
    let mut zeros = Vec::with_capacity(analog.poles.len());
    for zero in &analog.zeros {
        let denominator = one - *zero;
        if denominator.norm() <= EPS {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INTERNAL,
                "analog zero cannot be mapped through bilinear transform",
            ));
        }
        zeros.push((one + *zero) / denominator);
    }

    let mut poles = Vec::with_capacity(analog.poles.len());
    for pole in &analog.poles {
        let denominator = one - *pole;
        if denominator.norm() <= EPS {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INTERNAL,
                "analog pole cannot be mapped through bilinear transform",
            ));
        }
        poles.push((one + *pole) / denominator);
    }

    let degree = poles.len().saturating_sub(zeros.len());
    zeros.extend(std::iter::repeat_n(Complex64::new(-1.0, 0.0), degree));

    let zero_factor = analog
        .zeros
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, zero| acc * (one - *zero));
    let pole_factor = analog
        .poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, pole| acc * (one - *pole));
    if pole_factor.norm() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "bilinear gain normalization failed",
        ));
    }
    let gain = analog.gain * zero_factor / pole_factor;

    let zpk = Zpk { zeros, poles, gain };
    validate_zpk(&zpk)?;
    Ok(zpk)
}

async fn polynomial_from_roots(
    roots: &[Complex64],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex64>> {
    let mut coeffs = vec![Complex64::new(1.0, 0.0)];
    for root in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); coeffs.len() + 1];
        for (idx, coeff) in coeffs.iter().enumerate() {
            next[idx] += *coeff;
            next[idx + 1] -= *coeff * *root;
        }
        coeffs = next;
        cooperation.completed(coeffs.len() as u64).await?;
    }
    Ok(coeffs)
}

pub(super) fn normalize_transfer_function(
    mut numerator: Vec<Complex64>,
    mut denominator: Vec<Complex64>,
) -> BuiltinResult<(Vec<Complex64>, Vec<Complex64>)> {
    ensure_finite_coefficients(&numerator, "numerator")?;
    ensure_finite_coefficients(&denominator, "denominator")?;
    let Some(leading) = denominator.first().copied() else {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "empty denominator",
        ));
    };
    if leading.norm() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "zero leading denominator coefficient",
        ));
    }
    for coeff in &mut numerator {
        *coeff /= leading;
    }
    for coeff in &mut denominator {
        *coeff /= leading;
    }
    if numerator.len() < denominator.len() {
        let mut padded = vec![Complex64::new(0.0, 0.0); denominator.len() - numerator.len()];
        padded.extend(numerator);
        numerator = padded;
    }
    scrub_coefficients(&mut numerator);
    scrub_coefficients(&mut denominator);
    ensure_finite_coefficients(&numerator, "numerator")?;
    ensure_finite_coefficients(&denominator, "denominator")?;
    Ok((numerator, denominator))
}

fn ensure_finite_coefficients(coeffs: &[Complex64], label: &str) -> BuiltinResult<()> {
    if coeffs.iter().any(|value| !complex_is_finite(*value)) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            format!("filter design produced non-finite {label} coefficients"),
        ));
    }
    Ok(())
}

fn scrub_coefficients(coeffs: &mut [Complex64]) {
    for coeff in coeffs {
        if coeff.re.abs() <= REAL_TOL {
            coeff.re = 0.0;
        }
        if coeff.im.abs() <= REAL_TOL {
            coeff.im = 0.0;
        }
    }
}

fn row_coefficients_to_value(coeffs: &[Complex64]) -> BuiltinResult<Value> {
    complex_slice_to_value(coeffs, vec![1, coeffs.len()])
}

fn column_complex_to_value(values: &[Complex64]) -> BuiltinResult<Value> {
    complex_slice_to_value(values, vec![values.len(), 1])
}

fn scalar_complex_to_value(value: Complex64) -> BuiltinResult<Value> {
    if value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()) {
        Ok(Value::Num(value.re))
    } else {
        let tensor = ComplexTensor::new(vec![(value.re, value.im)], vec![1, 1])
            .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn complex_slice_to_value(values: &[Complex64], shape: Vec<usize>) -> BuiltinResult<Value> {
    if values
        .iter()
        .all(|value| value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()))
    {
        let tensor = Tensor::new(values.iter().map(|value| value.re).collect(), shape)
            .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::Tensor(tensor))
    } else {
        let tensor = ComplexTensor::new(
            values.iter().map(|value| (value.re, value.im)).collect(),
            shape,
        )
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn real_coefficients(values: &[Complex64]) -> BuiltinResult<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            if value.im.abs() <= REAL_TOL * (1.0 + value.re.abs()) {
                Ok(value.re)
            } else {
                Err(butter_error_with_detail(
                    &BUTTER_ERROR_INTERNAL,
                    "state-space output requires real coefficients",
                ))
            }
        })
        .collect()
}

fn transfer_function_to_state_space(
    numerator: &[f64],
    denominator: &[f64],
) -> BuiltinResult<(Tensor, Tensor, Tensor, f64)> {
    if denominator.len() < 2 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion requires a positive-order denominator",
        ));
    }

    let mut a = denominator.to_vec();
    let mut b = numerator.to_vec();
    let leading = a[0];
    if leading.abs() <= EPS {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion received a zero leading denominator",
        ));
    }
    for coeff in &mut a {
        *coeff /= leading;
    }
    for coeff in &mut b {
        *coeff /= leading;
    }
    if b.len() < a.len() {
        let mut padded = vec![0.0; a.len() - b.len()];
        padded.extend(b);
        b = padded;
    } else if b.len() > a.len() {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INTERNAL,
            "state-space conversion requires a proper transfer function",
        ));
    }

    let order = a.len() - 1;
    let mut a_data = vec![0.0; order * order];
    for col in 0..order {
        a_data[col * order] = -a[col + 1];
    }
    for row in 1..order {
        a_data[row + (row - 1) * order] = 1.0;
    }
    let a_tensor = Tensor::new(a_data, vec![order, order])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    let mut b_data = vec![0.0; order];
    b_data[0] = 1.0;
    let b_tensor = Tensor::new(b_data, vec![order, 1])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    let d_value = b[0];
    let c_data = (0..order)
        .map(|idx| b[idx + 1] - a[idx + 1] * d_value)
        .collect::<Vec<_>>();
    let c_tensor = Tensor::new(c_data, vec![1, order])
        .map_err(|detail| butter_error_with_detail(&BUTTER_ERROR_INTERNAL, detail))?;

    Ok((a_tensor, b_tensor, c_tensor, d_value))
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
