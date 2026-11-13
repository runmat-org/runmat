use super::CanonicalFluidError as Error;
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, EntityDimension, ExcitationAction, ExpectedTopology,
    FluidConstraint, FluidConstraintKind, PhysicalSignal, RegionTarget, SignalProvenance,
};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) enum FaceBoundary<'a> {
    Interior,
    Velocity {
        authority_id: &'a str,
        value: [f64; 3],
        moving_wall: bool,
    },
    PressureOutlet {
        authority_id: &'a str,
        pressure: f64,
    },
    /// Stationary impermeable plane without shear: zero normal velocity and zero
    /// tangential traction. Authored slip walls and symmetry planes share this law.
    Symmetry {
        authority_id: &'a str,
    },
}

mod kinematic;
mod mass_flow;

pub(super) struct BoundaryInput<'a, 's> {
    pub source: &'a AnalysisProjection,
    pub topology: &'a FiniteVolumeTopology,
    pub constraints: &'a [FluidConstraint],
    pub operators: &'a [AnalysisBoundary],
    pub signals: &'a [PhysicalSignal],
    pub excitations: &'a [ExcitationAction],
    pub density: f64,
    pub sample: crate::excitation::ExcitationSamplePoint<'s>,
}

pub(super) fn compile<'a>(input: BoundaryInput<'a, '_>) -> Result<Vec<FaceBoundary<'a>>, Error> {
    let BoundaryInput {
        source,
        topology: t,
        constraints,
        operators,
        signals,
        sample,
        ..
    } = input;
    let mut ids = std::collections::BTreeSet::new();
    for id in constraints
        .iter()
        .map(|c| c.constraint_id.as_str())
        .chain(operators.iter().map(|b| b.boundary_id.as_str()))
        .chain(input.excitations.iter().filter_map(|a| match a {
            ExcitationAction::MassFlow(a)
            | ExcitationAction::InletVelocity(a)
            | ExcitationAction::WallMotion(a) => Some(a.excitation_id.0.as_str()),
            _ => None,
        }))
    {
        if id.is_empty() || id.trim() != id || id.chars().any(char::is_control) || !ids.insert(id) {
            return Err(Error::Invalid("duplicate or malformed fluid authority ID"));
        }
    }
    let mut out = vec![FaceBoundary::Interior; t.faces.len()];
    for c in constraints {
        let (target, law) = match &c.kind {
            FluidConstraintKind::PrescribedVelocity { target, value }
            | FluidConstraintKind::MovingWall {
                target,
                velocity: value,
            } => {
                if value
                    .signal(signals)
                    .is_some_and(|s| s.provenance() != &SignalProvenance::Authored)
                {
                    return Err(Error::Unsupported("unauthenticated velocity signal source"));
                }
                let v = crate::excitation::sample_prescribed_velocity(value, signals, sample)
                    .map_err(|e| Error::InputRealization(e.to_string()))?;
                if v.imaginary != [0.0; 3] {
                    return Err(Error::Unsupported("complex fluid velocity"));
                }
                (
                    target,
                    FaceBoundary::Velocity {
                        authority_id: c.constraint_id.as_str(),
                        value: v.real,
                        moving_wall: matches!(&c.kind, FluidConstraintKind::MovingWall { .. }),
                    },
                )
            }
            FluidConstraintKind::NoSlipWall { target } => (
                target,
                FaceBoundary::Velocity {
                    authority_id: c.constraint_id.as_str(),
                    value: [0.0; 3],
                    moving_wall: false,
                },
            ),
            FluidConstraintKind::SlipWall { target } | FluidConstraintKind::Symmetry { target } => {
                (
                    target,
                    FaceBoundary::Symmetry {
                        authority_id: c.constraint_id.as_str(),
                    },
                )
            }
            // Gauge authority is compiled separately; it supplies no face law.
            FluidConstraintKind::PressureReference { .. } => continue,
            FluidConstraintKind::Periodic(_) => {
                return Err(Error::Unsupported(crate::admission::FLUID_PERIODIC.message))
            }
        };
        bind(source, t, &mut out, target, law)?;
    }
    for b in operators {
        let BoundaryOperator::FluidOutletPressure { value } = &b.operator else {
            return Err(Error::Unsupported("nonfluid boundary operator"));
        };
        if value
            .signal(signals)
            .is_some_and(|s| s.provenance() != &SignalProvenance::Authored)
        {
            return Err(Error::Unsupported("unauthenticated pressure signal source"));
        }
        let p = crate::excitation::sample_prescribed_pressure(value, signals, sample)
            .map_err(|e| Error::InputRealization(e.to_string()))?;
        if p.imaginary != 0.0 {
            return Err(Error::Unsupported("complex fluid pressure"));
        }
        bind(
            source,
            t,
            &mut out,
            &b.target,
            FaceBoundary::PressureOutlet {
                authority_id: b.boundary_id.as_str(),
                pressure: p.real,
            },
        )?;
    }
    mass_flow::compile(&input, &mut out)?;
    kinematic::compile(&input, &mut out)?;
    for (i, f) in t.faces.iter().enumerate() {
        if f.neighbor.is_none() && matches!(out[i], FaceBoundary::Interior) {
            return Err(Error::Invalid("incomplete exterior boundary coverage"));
        }
    }
    Ok(out)
}

fn bind<'a>(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    out: &mut [FaceBoundary<'a>],
    target: &RegionTarget,
    law: FaceBoundary<'a>,
) -> Result<(), Error> {
    let finite = match &law {
        FaceBoundary::Velocity { value, .. } => value.iter().all(|x| x.is_finite()),
        FaceBoundary::PressureOutlet { pressure, .. } => pressure.is_finite(),
        FaceBoundary::Symmetry { .. } => true,
        FaceBoundary::Interior => false,
    };
    if !finite {
        return Err(Error::Invalid("boundary value"));
    }
    for i in select_faces(source, t, target)? {
        bind_face(t, out, i, law)?;
    }
    Ok(())
}

fn bind_face<'a>(
    t: &FiniteVolumeTopology,
    out: &mut [FaceBoundary<'a>],
    i: usize,
    law: FaceBoundary<'a>,
) -> Result<(), Error> {
    let f = &t.faces[i];
    if let FaceBoundary::Velocity {
        value,
        moving_wall: true,
        ..
    } = &law
    {
        let normal = super::geometry_ops::dot(*value, f.owner_area_vector_m2).abs();
        let roundoff =
            128.0 * f64::EPSILON * super::geometry_ops::norm(*value) * f.area_m2.canonical_value();
        if normal > roundoff {
            return Err(Error::Unsupported(
                "normal moving-wall motion requires ALE geometry",
            ));
        }
    }
    if !matches!(out[i], FaceBoundary::Interior) {
        return Err(Error::Invalid("overlapping fluid boundary authorities"));
    }
    out[i] = law;

    Ok(())
}

pub(super) fn select_faces(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    target: &RegionTarget,
) -> Result<Vec<usize>, Error> {
    if target.entity_dimension != EntityDimension::Face
        || target.expected_topology != ExpectedTopology::BoundaryFace
    {
        return Err(Error::Invalid("fluid boundary target kind"));
    }
    if target.resolved_entities().is_empty() {
        return Err(Error::Invalid(
            "fluid boundary requires exact resolved entity authority",
        ));
    }
    target
        .validate_resolution()
        .map_err(|_| Error::Invalid("fluid boundary resolved entity authority"))?;
    let mut covered = vec![false; target.resolved_entities().len()];
    let mut matched = Vec::new();
    for (i, f) in t.faces.iter().enumerate() {
        let Some(b) = f.boundary_source else { continue };
        let original = &source.boundary_faces[b.projection_boundary_index];
        if !original
            .provenance
            .iter()
            .any(|e| target.matches_persistent_entity(e))
        {
            continue;
        }
        if f.neighbor.is_some() {
            return Err(Error::Invalid("fluid boundary selects interior interface"));
        }
        for entity in &original.provenance {
            if let Ok(j) = target.resolved_entities().binary_search(entity) {
                covered[j] = true;
            }
        }
        matched.push(i);
    }
    if matched.is_empty() || covered.iter().any(|seen| !*seen) {
        return Err(Error::Invalid("empty fluid boundary target"));
    }
    Ok(matched)
}
