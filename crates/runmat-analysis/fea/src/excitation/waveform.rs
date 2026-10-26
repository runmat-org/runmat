use runmat_analysis_core::{
    SignalSampleApproach, SignalValue, StandardWaveform, StandardWaveformKind,
};

use super::{
    model::ComplexScalar,
    sample::{ComplexTensor, SampledValue},
};
use super::{model::ComplexVector3, realize::RealizationError};

pub(super) fn sample<Q>(
    waveform: &StandardWaveform<Q>,
    time_seconds: f64,
    approach: SignalSampleApproach,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    if !time_seconds.is_finite() {
        return Err(RealizationError::new("waveform sample time must be finite"));
    }
    let side = side_at_event(waveform, time_seconds, approach);
    let gain = gain(waveform, time_seconds, side);
    combine(
        &waveform.amplitude,
        waveform.offset.as_ref(),
        gain,
        component,
    )
}

pub(super) fn breakpoints<Q>(
    waveform: &StandardWaveform<Q>,
    start: f64,
    end: f64,
) -> impl Iterator<Item = f64> {
    let mut points = [waveform.start_seconds.canonical_value(), 0.0, 0.0];
    let mut count = 1;
    if let Some(duration) = waveform
        .duration_seconds
        .map(|quantity| quantity.canonical_value())
    {
        points[count] = waveform.start_seconds.canonical_value() + duration;
        count += 1;
        if waveform.kind == StandardWaveformKind::RectangularPulse {
            if let Some(duty) = waveform.duty_cycle {
                points[count] = waveform.start_seconds.canonical_value() + duty * duration;
                count += 1;
            }
        }
    }
    points[..count].sort_by(f64::total_cmp);
    let mut previous = None;
    points.into_iter().take(count).filter(move |point| {
        if !(*point > start && *point <= end) {
            return false;
        }
        let unique = previous != Some(point.to_bits());
        previous = Some(point.to_bits());
        unique
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventSide {
    Left,
    At,
    Right,
}

fn side_at_event<Q>(
    waveform: &StandardWaveform<Q>,
    time: f64,
    approach: SignalSampleApproach,
) -> EventSide {
    let is_event = time == waveform.start_seconds.canonical_value()
        || waveform
            .duration_seconds
            .map(|quantity| quantity.canonical_value())
            .is_some_and(|duration| time == waveform.start_seconds.canonical_value() + duration)
        || (waveform.kind == StandardWaveformKind::RectangularPulse
            && waveform
                .duration_seconds
                .map(|quantity| quantity.canonical_value())
                .zip(waveform.duty_cycle)
                .is_some_and(|(duration, duty)| {
                    time == waveform.start_seconds.canonical_value() + duty * duration
                }));
    if !is_event {
        return EventSide::At;
    }
    match approach {
        SignalSampleApproach::LeftLimit => EventSide::Left,
        SignalSampleApproach::At => EventSide::At,
        SignalSampleApproach::RightLimit => EventSide::Right,
    }
}

fn gain<Q>(waveform: &StandardWaveform<Q>, time: f64, side: EventSide) -> f64 {
    let elapsed = time - waveform.start_seconds.canonical_value();
    if elapsed < 0.0 || (elapsed == 0.0 && side == EventSide::Left) {
        return 0.0;
    }
    let duration = waveform
        .duration_seconds
        .map(|quantity| quantity.canonical_value());
    if let Some(duration) = duration {
        if elapsed > duration || (elapsed == duration && side != EventSide::Left) {
            return match waveform.kind {
                StandardWaveformKind::Ramp => 1.0,
                _ => 0.0,
            };
        }
    }
    match waveform.kind {
        StandardWaveformKind::Step => 1.0,
        StandardWaveformKind::Ramp => elapsed / duration.unwrap_or(1.0),
        StandardWaveformKind::Sine => {
            let frequency = waveform
                .frequency_hz
                .unwrap_or(runmat_types::physical::Frequency::new(0.0));
            let angle = frequency.angular_frequency() * runmat_types::physical::Time::new(elapsed);
            (angle + waveform.phase_radians).canonical_value().sin()
        }
        StandardWaveformKind::HalfSinePulse => {
            (std::f64::consts::PI * elapsed / duration.unwrap_or(1.0)).sin()
        }
        StandardWaveformKind::TriangularPulse => {
            let normalized = elapsed / duration.unwrap_or(1.0);
            1.0 - (2.0 * normalized - 1.0).abs()
        }
        StandardWaveformKind::RectangularPulse => {
            let high_end = duration.unwrap_or(0.0) * waveform.duty_cycle.unwrap_or(0.0);
            if elapsed < high_end || (elapsed == high_end && side == EventSide::Left) {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn combine<Q>(
    amplitude: &SignalValue<Q>,
    offset: Option<&SignalValue<Q>>,
    gain: f64,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    match (amplitude, offset) {
        (SignalValue::Scalar(amplitude), None) => Ok(SampledValue::Scalar(ComplexScalar::real(
            gain * component(amplitude),
        ))),
        (SignalValue::Scalar(amplitude), Some(SignalValue::Scalar(offset))) => {
            Ok(SampledValue::Scalar(ComplexScalar::real(
                component(offset) + gain * component(amplitude),
            )))
        }
        (SignalValue::Vector3(amplitude), None) => {
            Ok(SampledValue::Vector3(ComplexVector3::real([
                gain * component(&amplitude[0]),
                gain * component(&amplitude[1]),
                gain * component(&amplitude[2]),
            ])))
        }
        (SignalValue::Vector3(amplitude), Some(SignalValue::Vector3(offset))) => {
            Ok(SampledValue::Vector3(ComplexVector3::real([
                component(&offset[0]) + gain * component(&amplitude[0]),
                component(&offset[1]) + gain * component(&amplitude[1]),
                component(&offset[2]) + gain * component(&amplitude[2]),
            ])))
        }
        (SignalValue::Tensor { shape, values }, offset) => {
            let offset = match offset {
                None => None,
                Some(SignalValue::Tensor {
                    shape: other,
                    values,
                }) if other == shape => Some(values),
                _ => {
                    return Err(RealizationError::new(
                        "waveform amplitude and offset shapes do not match",
                    ))
                }
            };
            if offset.is_some_and(|offset| offset.len() != values.len()) {
                return Err(RealizationError::new(
                    "waveform amplitude and offset counts differ",
                ));
            }
            let real = values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    gain * component(value) + offset.map_or(0.0, |values| component(&values[index]))
                })
                .collect::<Vec<_>>();
            let imaginary = vec![0.0; real.len()];
            Ok(SampledValue::Tensor(ComplexTensor::new(
                shape.clone(),
                real,
                imaginary,
            )?))
        }
        _ => Err(RealizationError::new(
            "waveform amplitude and offset shapes do not match",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::Force;

    fn waveform(kind: StandardWaveformKind) -> StandardWaveform<Force> {
        StandardWaveform {
            kind,
            amplitude: SignalValue::Scalar(Force::new(10.0)),
            offset: Some(SignalValue::Scalar(Force::new(2.0))),
            start_seconds: runmat_types::physical::Time::new(1.0),
            duration_seconds: (Some(2.0)).map(runmat_types::physical::Time::new),
            frequency_hz: (None).map(runmat_types::physical::Frequency::new),
            phase_radians: runmat_types::physical::Angle::new(0.0),
            duty_cycle: None,
        }
    }

    fn scalar_at(waveform: &StandardWaveform<Force>, time: f64) -> f64 {
        let SampledValue::Scalar(value) =
            sample(waveform, time, SignalSampleApproach::At, |value| {
                value.canonical_value()
            })
            .unwrap()
        else {
            panic!("expected scalar waveform sample");
        };
        value.real
    }

    #[test]
    fn finite_waveforms_have_exact_documented_support_and_shapes() {
        let ramp = waveform(StandardWaveformKind::Ramp);
        assert_eq!(scalar_at(&ramp, 0.5), 2.0);
        assert_eq!(scalar_at(&ramp, 2.0), 7.0);
        assert_eq!(scalar_at(&ramp, 3.0), 12.0);

        let half_sine = waveform(StandardWaveformKind::HalfSinePulse);
        assert!((scalar_at(&half_sine, 2.0) - 12.0).abs() < 1.0e-12);
        assert_eq!(scalar_at(&half_sine, 3.0), 2.0);

        let triangle = waveform(StandardWaveformKind::TriangularPulse);
        assert_eq!(scalar_at(&triangle, 2.0), 12.0);
        assert_eq!(scalar_at(&triangle, 3.0), 2.0);

        let mut rectangular = waveform(StandardWaveformKind::RectangularPulse);
        rectangular.duty_cycle = Some(0.25);
        assert_eq!(scalar_at(&rectangular, 1.25), 12.0);
        assert_eq!(scalar_at(&rectangular, 1.5), 2.0);
        let SampledValue::Scalar(left) = sample(
            &rectangular,
            1.5,
            SignalSampleApproach::LeftLimit,
            |value| value.canonical_value(),
        )
        .unwrap() else {
            panic!("expected scalar waveform sample");
        };
        assert_eq!(left.real, 12.0);
    }

    #[test]
    fn sine_and_step_waveforms_are_evaluated_analytically() {
        let mut sine = waveform(StandardWaveformKind::Sine);
        sine.duration_seconds = (None).map(runmat_types::physical::Time::new);
        sine.frequency_hz = (Some(0.25)).map(runmat_types::physical::Frequency::new);
        assert!((scalar_at(&sine, 2.0) - 12.0).abs() < 1.0e-12);

        let mut step = waveform(StandardWaveformKind::Step);
        step.duration_seconds = (None).map(runmat_types::physical::Time::new);
        assert_eq!(scalar_at(&step, 0.5), 2.0);
        assert_eq!(scalar_at(&step, 1.0), 12.0);
    }

    #[test]
    fn waveform_breakpoints_include_every_piecewise_boundary_once_in_range() {
        let ramp = waveform(StandardWaveformKind::Ramp);
        assert_eq!(breakpoints(&ramp, 0.0, 4.0).collect::<Vec<_>>(), [1.0, 3.0]);

        let mut rectangular = waveform(StandardWaveformKind::RectangularPulse);
        rectangular.duty_cycle = Some(0.25);
        assert_eq!(
            breakpoints(&rectangular, 1.0, 3.0).collect::<Vec<_>>(),
            [1.5, 3.0]
        );

        let mut sine = waveform(StandardWaveformKind::Sine);
        sine.duration_seconds = (None).map(runmat_types::physical::Time::new);
        sine.frequency_hz = (Some(1.0)).map(runmat_types::physical::Frequency::new);
        assert_eq!(breakpoints(&sine, 0.0, 4.0).collect::<Vec<_>>(), [1.0]);
    }
}
