use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, InitialConditionValue, InitialValue, PhysicalSignal, Signal,
    SignalRepresentation, TabulatedSignalDomain,
};

use crate::excitation::{expect_vector3, sample_signal, ExcitationSamplePoint};

use super::{StructuralInitialStateError, StructuralInitialValueSourceEvidence};

pub(super) fn realize_value<Q>(
    model: &AnalysisModel,
    condition: &InitialConditionValue<Q>,
    typed_signal: impl Fn(&PhysicalSignal) -> Option<&Signal<Q>>,
    component: impl Fn(&Q) -> f64 + Copy,
) -> Result<([f64; 3], StructuralInitialValueSourceEvidence), StructuralInitialStateError> {
    let id = &condition.initial_condition_id;
    let (values, source) = match &condition.value {
        InitialValue::UniformVector3(values) => (
            values.each_ref().map(component),
            StructuralInitialValueSourceEvidence::UniformVector3,
        ),
        InitialValue::Uniform(_) => {
            return Err(StructuralInitialStateError::InvalidValueShape {
                initial_condition_id: id.clone(),
            });
        }
        InitialValue::FieldArtifact(artifact) => {
            return Err(StructuralInitialStateError::UnmaterializedFieldArtifact {
                initial_condition_id: id.clone(),
                artifact_id: artifact.artifact_id.clone(),
            });
        }
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach,
        } => {
            if !abscissa.is_finite() || abscissa.to_bits() == (-0.0_f64).to_bits() {
                return Err(StructuralInitialStateError::InvalidRealizedValue {
                    initial_condition_id: id.clone(),
                });
            }
            let physical = model
                .signals
                .iter()
                .find(|signal| signal.signal_id() == signal_id)
                .ok_or_else(|| StructuralInitialStateError::MissingSignal {
                    initial_condition_id: id.clone(),
                    signal_id: signal_id.0.clone(),
                })?;
            let signal = typed_signal(physical).ok_or_else(|| {
                StructuralInitialStateError::IncompatibleSignal {
                    initial_condition_id: id.clone(),
                }
            })?;
            if let Some(frame) = &signal.frame {
                if !matches!(frame, CoordinateFrame::Global) {
                    return Err(StructuralInitialStateError::UnsupportedSignalFrame {
                        initial_condition_id: id.clone(),
                        frame: frame.context_label(),
                    });
                }
            }
            let sample = match &signal.representation {
                SignalRepresentation::Tabulated(table) => match &table.domain {
                    TabulatedSignalDomain::TimeSeconds => ExcitationSamplePoint::TimeSeconds {
                        value: *abscissa,
                        approach: *approach,
                    },
                    TabulatedSignalDomain::StudyParameter { parameter_id, .. } => {
                        ExcitationSamplePoint::StudyParameter {
                            parameter_id,
                            canonical_value: *abscissa,
                            approach: *approach,
                        }
                    }
                    TabulatedSignalDomain::FrequencyHertz => {
                        return Err(StructuralInitialStateError::UnsupportedSignalDomain {
                            initial_condition_id: id.clone(),
                        });
                    }
                },
                _ => ExcitationSamplePoint::TimeSeconds {
                    value: *abscissa,
                    approach: *approach,
                },
            };
            let sampled = sample_signal(signal, sample, component)
                .and_then(expect_vector3)
                .map_err(|error| StructuralInitialStateError::SignalSample {
                    initial_condition_id: id.clone(),
                    detail: error.to_string(),
                })?;
            if sampled.imaginary != [0.0; 3] {
                return Err(StructuralInitialStateError::InvalidRealizedValue {
                    initial_condition_id: id.clone(),
                });
            }
            (
                sampled.real,
                StructuralInitialValueSourceEvidence::SignalSample {
                    signal_id: signal_id.clone(),
                    abscissa_bits: abscissa.to_bits(),
                    approach: *approach,
                },
            )
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(StructuralInitialStateError::InvalidRealizedValue {
            initial_condition_id: id.clone(),
        });
    }
    Ok((values, source))
}
