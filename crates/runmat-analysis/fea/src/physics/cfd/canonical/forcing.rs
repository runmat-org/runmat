//! Steady global-frame shared body-force authority, not a solver-generated RHS.
use super::{geometry_ops::*, CanonicalFluidError as Error};
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    CoordinateFrame, EntityDimension, ExcitationAction, ExcitationDistribution,
    ExcitationKinematics, ExcitationTarget, ExpectedTopology, PhysicalSignal, SignalProvenance,
    SignalRepresentation, SignalShape, SignalValue,
};

pub(super) fn compile(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    actions: &[ExcitationAction],
    signals: &[PhysicalSignal],
) -> Result<Vec<V>, Error> {
    let mut force = vec![[0.0; 3]; t.cells.len()];
    for action in actions {
        let ExcitationAction::BodyForce(a) = action else {
            return Err(Error::Unsupported("non-bodyforce fluid excitation"));
        };
        if a.frame != CoordinateFrame::Global
            || a.kinematics != ExcitationKinematics::FixedFrame
            || a.distribution != ExcitationDistribution::IntensityPerVolume
            || !a.active_step_ids.is_empty()
            || a.combination_group.is_some()
        {
            return Err(Error::Unsupported("bodyforce frame/distribution/schedule"));
        }
        let matches = signals
            .iter()
            .filter(|s| s.signal_id() == &a.signal_id)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(Error::Invalid("exact bodyforce signal identity"));
        }
        let PhysicalSignal::BodyForceDensity(s) = matches[0] else {
            return Err(Error::Invalid("bodyforce signal quantity"));
        };
        if s.provenance != SignalProvenance::Authored
            || s.frame.as_ref() != Some(&CoordinateFrame::Global)
        {
            return Err(Error::Unsupported(
                "unauthenticated/non-global bodyforce source",
            ));
        }
        let v = match (&s.shape, &s.representation) {
            (SignalShape::Vector3, SignalRepresentation::Constant(SignalValue::Vector3(v)))
                if a.direction.is_none() =>
            {
                v.map(|x| x.newtons_per_cubic_meter)
            }
            (SignalShape::Scalar, SignalRepresentation::Constant(SignalValue::Scalar(v))) => {
                let d = a
                    .direction
                    .ok_or(Error::Invalid("scalar bodyforce requires signed direction"))?
                    .vector;
                if (norm(d) - 1.0).abs() > 1e-12 {
                    return Err(Error::Invalid("bodyforce unit direction"));
                }
                scale(d, v.newtons_per_cubic_meter)
            }
            _ => return Err(Error::Unsupported("bodyforce signal representation")),
        };
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("nonfinite bodyforce"));
        }
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
