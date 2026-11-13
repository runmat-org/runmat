//! Sampled global-frame shared body-force authority, not a solver-generated RHS.
use super::{geometry_ops::*, CanonicalFluidError as Error};
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    CoordinateFrame, EntityDimension, ExcitationAction, ExcitationDistribution,
    ExcitationKinematics, ExcitationTarget, ExpectedTopology, PhysicalSignal, SignalProvenance,
};

pub(super) fn compile(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    actions: &[ExcitationAction],
    signals: &[PhysicalSignal],
    sample: crate::excitation::ExcitationSamplePoint<'_>,
) -> Result<Vec<V>, Error> {
    let mut force = vec![[0.0; 3]; t.cells.len()];
    for action in actions {
        let a = match action {
            ExcitationAction::BodyForce(a) => a,
            // Validated and bound by the physical boundary owner.
            ExcitationAction::MassFlow(_)
            | ExcitationAction::InletVelocity(_)
            | ExcitationAction::WallMotion(_) => continue,
            _ => return Err(Error::Unsupported("unsupported fluid excitation")),
        };
        if a.frame != CoordinateFrame::Global
            || a.kinematics != ExcitationKinematics::FixedFrame
            || a.distribution != ExcitationDistribution::IntensityPerVolume
            || !a.active_step_ids.is_empty()
            || a.combination_group.is_some()
        {
            return Err(Error::Unsupported("bodyforce frame/distribution/schedule"));
        }
        let mut matches = signals.iter().filter(|s| s.signal_id() == &a.signal_id);
        let Some(PhysicalSignal::BodyForceDensity(s)) = matches.next() else {
            return Err(Error::Invalid(
                "exact bodyforce signal quantity and identity",
            ));
        };
        if matches.next().is_some() {
            return Err(Error::Invalid("duplicate bodyforce signal identity"));
        }
        if s.provenance != SignalProvenance::Authored
            || s.frame.as_ref() != Some(&CoordinateFrame::Global)
        {
            return Err(Error::Unsupported(
                "unauthenticated/non-global bodyforce source",
            ));
        }
        if a.direction.is_some() && s.shape != runmat_analysis_core::SignalShape::Scalar {
            return Err(Error::Unsupported(
                "vector fluid bodyforce already declares direction",
            ));
        }
        if let Some(direction) = a.direction {
            if (norm(direction.vector) - 1.0).abs() > 1e-12 {
                return Err(Error::Invalid("bodyforce unit direction"));
            }
        }
        let sampled = crate::excitation::sample_signal(s, sample, |v| v.canonical_value())
            .map_err(|e| Error::InputRealization(e.to_string()))?;
        let v = crate::excitation::expect_vector3_or_direction(sampled, a.direction)
            .map_err(|e| Error::InputRealization(e.to_string()))?;
        if v.imaginary != [0.0; 3] || v.real.iter().any(|v| !v.is_finite()) {
            return Err(Error::Unsupported("nonreal/nonfinite fluid body force"));
        }
        let v = v.real;
        let region = match &a.target {
            ExcitationTarget::WholeModel => None,
            ExcitationTarget::Region(target) => {
                if target.entity_dimension != EntityDimension::Volume
                    || !matches!(
                        target.expected_topology,
                        ExpectedTopology::MeshElement | ExpectedTopology::Cell
                    )
                    || target.resolved_entities().is_empty()
                    || target.validate_resolution().is_err()
                {
                    return Err(Error::Invalid("bodyforce exact volume target"));
                }
                Some(target)
            }
            _ => return Err(Error::Unsupported("bodyforce nonvolume target")),
        };
        let mut covered = region.map(|t| vec![false; t.resolved_entities().len()]);
        let mut matched = 0usize;
        for (c, cell) in t.cells.iter().enumerate() {
            let selected = match region {
                None => true,
                Some(target) => {
                    match target
                        .resolved_entities()
                        .binary_search(&source.elements[cell.source_element_index].region_id)
                    {
                        Ok(i) => {
                            if let Some(covered) = &mut covered {
                                covered[i] = true;
                            }
                            true
                        }
                        Err(_) => false,
                    }
                }
            };
            if selected {
                force[c] = add(force[c], v);
                matched += 1;
            }
        }
        if matched == 0 {
            return Err(Error::Invalid("empty bodyforce volume"));
        }
        if covered.is_some_and(|v| v.into_iter().any(|seen| !seen)) {
            return Err(Error::Invalid(
                "incomplete original bodyforce volume coverage",
            ));
        }
    }
    Ok(force)
}
