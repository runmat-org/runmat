//! Host SISO frequency response and output materialization.

use super::*;
use crate::builtins::control::frequency_response;

pub(super) struct NyquistResponse {
    pub(super) re: Vec<f64>,
    pub(super) im: Vec<f64>,
    pub(super) w: Vec<f64>,
    #[cfg(feature = "plot-core")]
    pub(super) mirror_negative_frequency: bool,
}

impl NyquistResponse {
    pub(super) fn re_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.re.clone())
    }
    pub(super) fn im_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.im.clone())
    }
    fn w_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.w.clone())
    }
    pub(super) fn outputs(&self, count: usize) -> BuiltinResult<Vec<Value>> {
        let mut values = Vec::with_capacity(count);
        if count >= 1 {
            values.push(self.re_value()?);
        }
        if count >= 2 {
            values.push(self.im_value()?);
        }
        if count >= 3 {
            values.push(self.w_value()?);
        }
        Ok(values)
    }
}

pub(super) fn evaluate(model: &TfModel, w: Vec<f64>) -> BuiltinResult<NyquistResponse> {
    let mut re = Vec::with_capacity(w.len());
    let mut im = Vec::with_capacity(w.len());
    let physical = model.physical_polynomials();
    for &frequency in &w {
        let point = frequency_response::point_at_frequency(
            frequency_response::FrequencyApi::Nyquist,
            model,
            frequency,
        )?;
        let value = physical.evaluate(point).ok_or_else(|| {
            control_error(
                BUILTIN_NAME,
                "RunMat:nyquist:InvalidArgument",
                "nyquist: frequency response is singular at a requested frequency",
            )
        })?;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(control_error(
                BUILTIN_NAME,
                "RunMat:nyquist:InvalidModel",
                "nyquist: finite frequency response cannot be represented at a requested frequency",
            ));
        }
        re.push(value.re);
        im.push(value.im);
    }
    Ok(NyquistResponse {
        re,
        im,
        w,
        #[cfg(feature = "plot-core")]
        mirror_negative_frequency: model
            .numerator
            .iter()
            .chain(&model.denominator)
            .all(|value| value.im == 0.0),
    })
}
