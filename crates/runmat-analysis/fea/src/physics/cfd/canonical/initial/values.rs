//! Typed initial values reuse shared field admission and quantity-aware samplers.
use super::*;
use crate::{execution::InitialFieldBinding, topology::AnalysisProjection};
use runmat_analysis_core::{
    CoordinateFrame, InitialConditionValue, InitialValue, PhysicalSignal, PrescribedValue,
    Pressure, Velocity,
};
pub(super) enum Vector<'a> {
    Uniform([f64; 3]),
    Field(&'a [f64]),
}
impl Vector<'_> {
    pub fn at(&self, n: usize) -> Result<[f64; 3], Error> {
        match self {
            Self::Uniform(v) => Ok(*v),
            Self::Field(v) => {
                let row = v
                    .get(n * 3..n * 3 + 3)
                    .ok_or(Error::Invalid("initial vector row"))?;
                Ok([row[0], row[1], row[2]])
            }
        }
    }
}
pub(super) enum Scalar<'a> {
    Uniform(f64),
    Field(&'a [f64]),
}
impl Scalar<'_> {
    pub fn at(&self, n: usize) -> Result<f64, Error> {
        match self {
            Self::Uniform(v) => Ok(*v),
            Self::Field(v) => v
                .get(n)
                .copied()
                .ok_or(Error::Invalid("initial scalar row")),
        }
    }
}
fn field<'a>(
    condition: &InitialCondition,
    source: &AnalysisProjection,
    fields: &'a [InitialFieldBinding],
) -> Result<&'a [f64], Error> {
    let reference = condition.field_artifact_ref().ok_or(Error::Invalid(
        "missing original initial artifact declaration",
    ))?;
    let binding = fields
        .iter()
        .find(|f| f.reference() == reference)
        .ok_or(Error::Invalid("missing original initial artifact"))?;
    binding
        .admit(
            condition,
            runmat_execution::Digest::from_bytes(*source.mesh_digest.bytes()),
            source.nodes.len(),
        )
        .map(|f| f.values())
        .map_err(|e| Error::InputRealization(e.to_string()))
}
pub(super) fn velocity<'a>(
    model: &AnalysisModel,
    c: &InitialConditionValue<Velocity>,
    condition: &InitialCondition,
    source: &AnalysisProjection,
    fields: &'a [InitialFieldBinding],
) -> Result<Vector<'a>, Error> {
    if c.frame != CoordinateFrame::Global {
        return Err(Error::Unsupported("initial velocity frame"));
    }
    let v = match &c.value {
        InitialValue::FieldArtifact(_) => {
            return Ok(Vector::Field(field(condition, source, fields)?))
        }
        InitialValue::UniformVector3(v) => v.map(|v| v.canonical_value()),
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach,
        } => {
            let prescribed = PrescribedValue::<[Velocity; 3]>::Signal(signal_id.clone());
            let Some(PhysicalSignal::Velocity(s)) = prescribed.signal(&model.signals) else {
                return Err(Error::Invalid("initial velocity signal quantity/identity"));
            };
            let point = crate::excitation::initial_signal_sample_point(
                &s.representation,
                *abscissa,
                *approach,
            )
            .map_err(|e| Error::InputRealization(e.to_string()))?;
            let v =
                crate::excitation::sample_prescribed_velocity(&prescribed, &model.signals, point)
                    .map_err(|e| Error::InputRealization(e.to_string()))?;
            if v.imaginary != [0.; 3] {
                return Err(Error::Unsupported("complex initial velocity"));
            }
            v.real
        }
        _ => {
            return Err(Error::Unsupported(
                "initial velocity requires original vector",
            ))
        }
    };
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::Invalid("nonfinite initial velocity"));
    }
    Ok(Vector::Uniform(v))
}
pub(super) fn pressure<'a>(
    model: &AnalysisModel,
    c: &InitialConditionValue<Pressure>,
    condition: &InitialCondition,
    source: &AnalysisProjection,
    fields: &'a [InitialFieldBinding],
) -> Result<Scalar<'a>, Error> {
    if c.frame != CoordinateFrame::Global {
        return Err(Error::Unsupported("initial pressure frame"));
    }
    let v = match &c.value {
        InitialValue::FieldArtifact(_) => {
            return Ok(Scalar::Field(field(condition, source, fields)?))
        }
        InitialValue::Uniform(v) => v.canonical_value(),
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach,
        } => {
            let mut matching = model.signals.iter().filter(|s| s.signal_id() == signal_id);
            let Some(PhysicalSignal::Pressure(s)) =
                matching.next().filter(|_| matching.next().is_none())
            else {
                return Err(Error::Invalid("initial pressure signal quantity/identity"));
            };
            let point = crate::excitation::initial_signal_sample_point(
                &s.representation,
                *abscissa,
                *approach,
            )
            .map_err(|e| Error::InputRealization(e.to_string()))?;
            let v = crate::excitation::sample_prescribed_pressure(
                &PrescribedValue::Signal(signal_id.clone()),
                &model.signals,
                point,
            )
            .map_err(|e| Error::InputRealization(e.to_string()))?;
            if v.imaginary != 0. {
                return Err(Error::Unsupported("complex initial pressure"));
            }
            v.real
        }
        _ => {
            return Err(Error::Unsupported(
                "initial pressure requires original scalar",
            ))
        }
    };
    if !v.is_finite() {
        return Err(Error::Invalid("nonfinite initial pressure"));
    }
    Ok(Scalar::Uniform(v))
}
