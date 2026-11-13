//! Authored inlet velocity and wall motion: a prescribed velocity vector on exact faces, from a
//! vector signal or a scalar speed history with a unit direction. Wall motion is tangential;
//! normal wall motion belongs to ALE geometry (fluid–structure interaction).
use super::{bind_face, select_faces, BoundaryInput, Error, FaceBoundary};
use runmat_analysis_core::{
    CoordinateFrame, ExcitationAction, ExcitationDistribution, ExcitationKinematics,
    ExcitationTarget, PhysicalSignal, SignalProvenance,
};

pub(super) fn compile<'a>(
    input: &BoundaryInput<'a, '_>,
    out: &mut [FaceBoundary<'a>],
) -> Result<(), Error> {
    for action in input.excitations {
        let (a, moving_wall) = match action {
            ExcitationAction::InletVelocity(a) => (a, false),
            ExcitationAction::WallMotion(a) => (a, true),
            _ => continue,
        };
        if a.frame != CoordinateFrame::Global
            || a.kinematics != ExcitationKinematics::FixedFrame
            || a.distribution != ExcitationDistribution::UniformKinematic
            || !a.active_step_ids.is_empty()
            || a.combination_group.is_some()
        {
            return Err(Error::Unsupported(
                "fluid kinematic input requires a fixed-frame uniform velocity",
            ));
        }
        let ExcitationTarget::Region(target) = &a.target else {
            return Err(Error::Unsupported(
                "fluid kinematic input requires an exact boundary region",
            ));
        };
        let mut matching = input
            .signals
            .iter()
            .filter(|s| s.signal_id() == &a.signal_id);
        let Some(PhysicalSignal::Velocity(signal)) = matching.next() else {
            return Err(Error::Invalid(
                "exact velocity signal quantity and identity",
            ));
        };
        if matching.next().is_some() {
            return Err(Error::Invalid("duplicate velocity signal identity"));
        }
        if signal.provenance != SignalProvenance::Authored
            || signal
                .frame
                .as_ref()
                .is_some_and(|frame| frame != &CoordinateFrame::Global)
        {
            return Err(Error::Unsupported(
                "fluid kinematic input requires an authored global velocity source",
            ));
        }
        if let Some(direction) = a.direction {
            if (super::super::geometry_ops::norm(direction.vector) - 1.0).abs() > 1e-12 {
                return Err(Error::Invalid("fluid kinematic unit direction"));
            }
        }
        let sampled =
            crate::excitation::sample_signal(signal, input.sample, |v| v.canonical_value())
                .and_then(|value| {
                    crate::excitation::expect_vector3_or_direction(value, a.direction)
                })
                .map_err(|e| Error::InputRealization(e.to_string()))?;
        if sampled.imaginary != [0.0; 3] || sampled.real.iter().any(|v| !v.is_finite()) {
            return Err(Error::Unsupported("nonreal or nonfinite fluid velocity"));
        }
        let law = FaceBoundary::Velocity {
            authority_id: a.excitation_id.0.as_str(),
            value: sampled.real,
            moving_wall,
        };
        for i in select_faces(input.source, input.topology, target)? {
            bind_face(input.topology, out, i, law)?;
        }
    }
    Ok(())
}
