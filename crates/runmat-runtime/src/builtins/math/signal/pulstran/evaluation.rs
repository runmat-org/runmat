use super::*;

pub(super) async fn evaluate_pulse_train(
    t: &Tensor,
    delays: &[PulseInstance],
    source: &PulseSource,
) -> BuiltinResult<Tensor> {
    let times = tensor::tensor_values_f64_cow(t);
    let mut out = vec![0.0; times.len()];
    for pulse in delays {
        match source {
            PulseSource::Rect { width } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * rectpuls_scalar(time - pulse.delay, *width);
                }
            }
            PulseSource::Tri { width, skew } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * tripuls_scalar(time - pulse.delay, *width, *skew);
                }
            }
            PulseSource::Gaus { params } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * gauspuls_scalar(time - pulse.delay, *params);
                }
            }
            PulseSource::Prototype { samples, fs } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * prototype_value(samples, *fs, time - pulse.delay);
                }
            }
            PulseSource::Callback { handle, args } => {
                let shifted = shifted_time_value(t, pulse.delay)?;
                let mut call_args = Vec::with_capacity(args.len() + 1);
                call_args.push(shifted);
                call_args.extend(args.iter().cloned());
                let value = crate::call_feval_async_with_outputs(handle.clone(), &call_args, 1)
                    .await
                    .map_err(|err| {
                        pulstran_error_with_source(
                            &PULSTRAN_ERROR_INVALID_PULSE,
                            "pulse function call failed",
                            err,
                        )
                    })?;
                let samples = callback_samples(value, times.len()).await?;
                for (idx, sample) in samples.into_iter().enumerate() {
                    out[idx] += pulse.amplitude * sample;
                }
            }
        }
    }
    Tensor::new(out, t.shape().to_vec())
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

fn shifted_time_value(t: &Tensor, delay: f64) -> BuiltinResult<Value> {
    let data = tensor::tensor_values_f64_cow(t)
        .iter()
        .map(|value| value - delay)
        .collect::<Vec<_>>();
    Tensor::new(data, t.shape().to_vec())
        .map(tensor_into_value)
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

pub(super) async fn callback_samples(value: Value, expected_len: usize) -> BuiltinResult<Vec<f64>> {
    reject_nonreal_operand(&value, &PULSTRAN_ERROR_INVALID_PULSE)?;
    // Callback results are waveform samples, not callback extra arguments:
    // preserve the historical typed-integer admission but require exact
    // binary64 projection before the numeric evaluator materializes them.
    let value = if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        let prepared =
            crate::builtins::common::validation::prepare_direct_integer_f64_boundary(&value, None)
                .await
                .map_err(|failure| {
                    pulstran_error_with_source(
                        &PULSTRAN_ERROR_INVALID_PULSE,
                        "callback sample integer validation failed",
                        failure.into_runtime_error(),
                    )
                })?;
        if !prepared.is_exact() {
            return Err(pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PULSE,
                "callback samples must be exactly representable as double",
            ));
        }
        match prepared.into_resident_payload() {
            Some(payload) => payload
                .into_value()
                .map_err(|detail| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, detail))?,
            None => value,
        }
    } else {
        value
    };
    let tensor = real_tensor_arg(value, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    let actual_len = tensor.len();
    if actual_len == expected_len {
        Ok(tensor::tensor_into_values_f64(tensor))
    } else if actual_len == 1 {
        Ok(vec![tensor::tensor_value_f64(&tensor, 0); expected_len])
    } else {
        Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            format!(
                "pulse function returned {} samples for {expected_len} input samples",
                actual_len
            ),
        ))
    }
}

fn prototype_value(samples: &[f64], fs: f64, t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if !t.is_finite() || samples.is_empty() {
        return 0.0;
    }
    let x = t * fs;
    if x < 0.0 || x > (samples.len() - 1) as f64 {
        return 0.0;
    }
    if samples.len() == 1 {
        return if x == 0.0 { samples[0] } else { 0.0 };
    }
    let lower = x.floor() as usize;
    if lower + 1 >= samples.len() {
        return samples[lower];
    }
    let frac = x - lower as f64;
    samples[lower] * (1.0 - frac) + samples[lower + 1] * frac
}
