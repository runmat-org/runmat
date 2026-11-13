//! Signed integral mass input on the original exact face selection.
//! Positive rate is into the fluid; reporting remains outward from each owner.
use super::{bind_face, select_faces, BoundaryInput, Error, FaceBoundary};
use crate::physics::cfd::canonical::geometry_ops::{dot, scale};
use runmat_analysis_core::{
    CoordinateFrame, ExcitationAction, ExcitationDistribution, ExcitationKinematics,
    ExcitationTarget, PhysicalSignal, SignalProvenance,
};

pub(super) fn compile<'a>(
    input: &BoundaryInput<'a, '_>,
    out: &mut [FaceBoundary<'a>],
) -> Result<(), Error> {
    for action in input.excitations {
        let ExcitationAction::MassFlow(a) = action else {
            continue;
        };
        if a.frame != CoordinateFrame::Global
            || a.direction.is_some()
            || a.kinematics != ExcitationKinematics::FixedFrame
            || a.distribution != ExcitationDistribution::PreserveResultant
            || !a.active_step_ids.is_empty()
            || a.combination_group.is_some()
            || !a.projection.preserve_resultant
            || a.projection.preserve_moment
        {
            return Err(Error::Unsupported(
                "mass flow requires fixed scalar total normal-flow authority",
            ));
        }
        let tolerance = a.projection.relative_tolerance;
        if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(Error::Invalid("mass flow projection tolerance"));
        }
        let ExcitationTarget::Region(target) = &a.target else {
            return Err(Error::Unsupported(
                "mass flow requires an exact boundary region",
            ));
        };
        let mut matching = input
            .signals
            .iter()
            .filter(|s| s.signal_id() == &a.signal_id);
        let Some(PhysicalSignal::MassFlowRate(signal)) = matching.next() else {
            return Err(Error::Invalid(
                "exact mass-flow signal quantity and identity",
            ));
        };
        if matching.next().is_some() {
            return Err(Error::Invalid("duplicate mass-flow signal identity"));
        }
        if signal.provenance != SignalProvenance::Authored || signal.frame.is_some() {
            return Err(Error::Unsupported(
                "mass flow requires an authored scalar source without a frame",
            ));
        }
        let sampled =
            crate::excitation::sample_signal(signal, input.sample, |v| v.canonical_value())
                .and_then(crate::excitation::expect_scalar)
                .map_err(|e| Error::InputRealization(e.to_string()))?;
        if sampled.imaginary != 0.0 {
            return Err(Error::Unsupported("complex fluid mass flow"));
        }
        let faces = select_faces(input.source, input.topology, target)?;
        let area = faces
            .iter()
            .map(|&i| input.topology.faces[i].area_m2.canonical_value())
            .sum::<f64>();
        if !area.is_finite() || area <= 0.0 {
            return Err(Error::Invalid("mass-flow boundary area"));
        }
        // Divide separately: rho*A may overflow even for a finite representable speed.
        let normal_speed = -(sampled.real / input.density) / area;
        if !normal_speed.is_finite() {
            return Err(Error::Invalid("mass-flow normal speed"));
        }
        let mut realized = 0.0;
        for &i in &faces {
            let face = &input.topology.faces[i];
            let value = scale(
                face.owner_area_vector_m2,
                normal_speed / face.area_m2.canonical_value(),
            );
            if value.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("mass-flow boundary velocity"));
            }
            realized -= input.density * dot(value, face.owner_area_vector_m2);
            bind_face(
                input.topology,
                out,
                i,
                FaceBoundary::Velocity {
                    authority_id: a.excitation_id.0.as_str(),
                    value,
                    moving_wall: false,
                },
            )?;
        }
        let defect = (realized - sampled.real).abs();
        if !realized.is_finite() || defect > tolerance * sampled.real.abs() {
            return Err(Error::Invalid(
                "mass-flow total projection does not meet authored tolerance",
            ));
        }
    }
    Ok(())
}
