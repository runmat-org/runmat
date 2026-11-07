//! A heat-generation region authored as a total heat rate P (W) is compiled once, against the
//! solver mesh, into a uniform density P / V over the meshed region volume V. Every thermal
//! kernel then reads an ordinary density, and the generated heat integrates to exactly P.
use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationDistribution, ExcitationTarget,
    HeatGenerationDensity, PhysicalSignal, Power, Signal, SignalKnot, SignalKnotValue,
    SignalOrdinate, SignalRepresentation, SignalValue,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::AnalysisStudyCompilationError;
use crate::topology::AnalysisProjection;

pub(super) fn distribute_heat_rate_sources(
    model: &mut AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<(), AnalysisStudyCompilationError> {
    let totals = model
        .excitations
        .iter()
        .filter_map(|action| match action {
            ExcitationAction::HeatGeneration(excitation)
                if excitation.distribution == ExcitationDistribution::PreserveResultant =>
            {
                Some(excitation.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if totals.is_empty() {
        return Ok(());
    }
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AnalysisStudyCompilationError::InvalidMesh(error.to_string()))?;
    for excitation in totals {
        let failure = |reason: String| AnalysisStudyCompilationError::HeatRate {
            excitation_id: excitation.excitation_id.0.clone(),
            reason,
        };
        let volume_m3 = projection
            .elements
            .iter()
            .filter(|element| match &excitation.target {
                ExcitationTarget::WholeModel => true,
                ExcitationTarget::Region(target) => {
                    target.selects_volume_element(&element.region_id, &element.provenance)
                }
                _ => false,
            })
            .map(|element| element.signed_volume_m3)
            .sum::<f64>();
        if !(volume_m3.is_finite() && volume_m3 > 0.0) {
            return Err(failure(
                "a total heat rate needs a solid region with positive meshed volume".into(),
            ));
        }
        let index = model
            .signals
            .iter()
            .position(|signal| signal.signal_id() == &excitation.signal_id)
            .ok_or_else(|| failure("its heat-rate signal is missing".into()))?;
        let PhysicalSignal::HeatRate(signal) = &model.signals[index] else {
            return Err(failure(
                "a total heat generation reads a heat-rate (W) signal".into(),
            ));
        };
        let density = density_signal(signal, volume_m3).map_err(|reason| failure(reason.into()))?;
        // The density keeps the signal identity, so every excitation that shares the signal
        // shares the conversion; a shared signal on regions of different volume is rejected.
        if model.excitations.iter().any(|other| {
            other.excitation().signal_id == excitation.signal_id
                && other.excitation().excitation_id != excitation.excitation_id
        }) {
            return Err(failure(
                "a total heat-rate signal must drive exactly one heat-generation region".into(),
            ));
        }
        if let SignalRepresentation::Constant(SignalValue::Scalar(total)) = &signal.representation {
            tracing::info!(
                excitation = %excitation.excitation_id.0,
                total_w = total.canonical_value(),
                region_volume_m3 = volume_m3,
                density_w_per_m3 = total.canonical_value() / volume_m3,
                "heat generation: total heat rate distributed uniformly as P / V over the meshed region volume"
            );
        } else {
            tracing::info!(
                excitation = %excitation.excitation_id.0,
                region_volume_m3 = volume_m3,
                "heat generation: tabulated total heat rate distributed uniformly as P / V over the meshed region volume"
            );
        }
        model.signals[index] = PhysicalSignal::HeatGenerationDensity(density);
        for action in &mut model.excitations {
            if let ExcitationAction::HeatGeneration(other) = action {
                if other.excitation_id == excitation.excitation_id {
                    other.distribution = ExcitationDistribution::IntensityPerVolume;
                }
            }
        }
    }
    Ok(())
}

/// `signal / volume`: a constant or tabulated scalar heat rate as a heat-generation density.
fn density_signal(
    signal: &Signal<Power>,
    volume_m3: f64,
) -> Result<Signal<HeatGenerationDensity>, &'static str> {
    let scale = |value: &Power| HeatGenerationDensity::new(value.canonical_value() / volume_m3);
    let value = |value: &SignalValue<Power>| match value {
        SignalValue::Scalar(power) => Ok(SignalValue::Scalar(scale(power))),
        _ => Err("a total heat rate is a scalar"),
    };
    let ordinate = |ordinate: &SignalOrdinate<Power>| {
        Ok::<_, &'static str>(match ordinate {
            SignalOrdinate::Real(real) => SignalOrdinate::Real(value(real)?),
            SignalOrdinate::Complex { real, imaginary } => SignalOrdinate::Complex {
                real: value(real)?,
                imaginary: value(imaginary)?,
            },
        })
    };
    let representation = match &signal.representation {
        SignalRepresentation::Constant(constant) => {
            SignalRepresentation::Constant(value(constant)?)
        }
        SignalRepresentation::Tabulated(table) => {
            let mut knots = Vec::with_capacity(table.knots.len());
            for knot in &table.knots {
                knots.push(SignalKnot {
                    abscissa: knot.abscissa,
                    value: match &knot.value {
                        SignalKnotValue::Continuous(value) => {
                            SignalKnotValue::Continuous(ordinate(value)?)
                        }
                        SignalKnotValue::Jump {
                            left_limit,
                            right_limit,
                            continuity,
                        } => SignalKnotValue::Jump {
                            left_limit: ordinate(left_limit)?,
                            right_limit: ordinate(right_limit)?,
                            continuity: *continuity,
                        },
                    },
                });
            }
            SignalRepresentation::Tabulated(runmat_analysis_core::TabulatedSignal {
                domain: table.domain.clone(),
                knots,
                interpolation: table.interpolation.clone(),
                extrapolation: table.extrapolation.clone(),
            })
        }
        _ => return Err("a total heat rate must be a constant or tabulated signal"),
    };
    Ok(Signal {
        signal_id: signal.signal_id.clone(),
        shape: signal.shape.clone(),
        frame: signal.frame.clone(),
        representation,
        precision: signal.precision,
        evidence: signal.evidence.clone(),
        provenance: signal.provenance.clone(),
    })
}

#[cfg(test)]
mod tests;
