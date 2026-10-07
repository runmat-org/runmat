use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GauspulsParams {
    pub fc: f64,
    pub bw: f64,
    pub bwr: f64,
}

pub(crate) fn default_params() -> GauspulsParams {
    GauspulsParams {
        fc: DEFAULT_FC,
        bw: DEFAULT_BW,
        bwr: DEFAULT_BWR,
    }
}

pub(crate) fn validate_params(params: GauspulsParams) -> Result<GauspulsParams, String> {
    if !params.fc.is_finite() || params.fc <= 0.0 {
        return Err(format!(
            "carrier frequency must be positive and finite, got {}",
            params.fc
        ));
    }
    if !params.bw.is_finite() || params.bw <= 0.0 {
        return Err(format!(
            "bandwidth must be positive and finite, got {}",
            params.bw
        ));
    }
    if !params.bwr.is_finite() || params.bwr >= 0.0 {
        return Err(format!(
            "bandwidth reference must be negative and finite, got {}",
            params.bwr
        ));
    }
    Ok(params)
}

pub(crate) fn validate_tpe(tpe: f64) -> Result<f64, String> {
    if !tpe.is_finite() || tpe >= 0.0 {
        Err(format!(
            "cutoff envelope level must be negative and finite, got {tpe}"
        ))
    } else {
        Ok(tpe)
    }
}

pub(crate) fn gauspuls_scalar(t: f64, params: GauspulsParams) -> f64 {
    let (in_phase, _, _) = gauspuls_components_scalar(t, params);
    in_phase
}

pub(crate) fn gauspuls_components_scalar(t: f64, params: GauspulsParams) -> (f64, f64, f64) {
    if t.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if !t.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let a = gaussian_shape_factor(params);
    let envelope = (-a * t * t).exp();
    let phase = 2.0 * PI * params.fc * t;
    (envelope * phase.cos(), envelope * phase.sin(), envelope)
}

pub(crate) fn gauspuls_cutoff(params: GauspulsParams, tpe: f64) -> f64 {
    let a = gaussian_shape_factor(params);
    (-db_to_log_amplitude(tpe) / a).sqrt()
}

pub(crate) fn gauspuls_tensor(tensor: Tensor, params: GauspulsParams) -> Result<Tensor, String> {
    let shape = tensor.shape().to_vec();
    ensure_exact_integer_tensor_boundary(&tensor, "sample time")?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| err.to_string())?;
    let storage = match storage {
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| gauspuls_scalar(f64::from(value), params) as f32)
                .collect(),
        ),
        storage => NumericStorage::F64(
            storage
                .materialize_f64()
                .into_iter()
                .map(|value| gauspuls_scalar(value, params))
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(storage, shape).map_err(|err| err.to_string())
}

pub(crate) fn gauspuls_components_tensor(
    tensor: Tensor,
    params: GauspulsParams,
) -> Result<(Tensor, Tensor, Tensor), String> {
    let shape = tensor.shape().to_vec();
    ensure_exact_integer_tensor_boundary(&tensor, "sample time")?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| err.to_string())?;
    let output_is_single = matches!(&storage, NumericStorage::F32(_));
    let values = storage.materialize_f64();
    let mut in_phase = Vec::with_capacity(values.len());
    let mut quadrature = Vec::with_capacity(values.len());
    let mut envelope = Vec::with_capacity(values.len());
    for value in values {
        let (yi, yq, ye) = gauspuls_components_scalar(value, params);
        in_phase.push(yi);
        quadrature.push(yq);
        envelope.push(ye);
    }
    let build = |values: Vec<f64>, shape: Vec<usize>| {
        let storage = if output_is_single {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        } else {
            NumericStorage::F64(values)
        };
        Tensor::from_numeric_storage(storage, shape).map_err(|err| err.to_string())
    };
    Ok((
        build(in_phase, shape.clone())?,
        build(quadrature, shape.clone())?,
        build(envelope, shape)?,
    ))
}

pub(super) fn ensure_exact_integer_tensor_boundary(
    tensor: &Tensor,
    role: &str,
) -> Result<(), String> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .all(crate::builtins::common::validation::integer_is_exact_f64)
    {
        Ok(())
    } else {
        Err(format!(
            "integer {role} must be exactly representable as double"
        ))
    }
}

fn gaussian_shape_factor(params: GauspulsParams) -> f64 {
    let numerator = -(PI * params.fc * params.bw).powi(2);
    numerator / (4.0 * db_to_log_amplitude(params.bwr))
}

fn db_to_log_amplitude(db: f64) -> f64 {
    db * (std::f64::consts::LN_10 / 20.0)
}
