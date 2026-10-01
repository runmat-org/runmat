use super::CanonicalFluidError as Error;
use crate::{
    algebra::connected_components,
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, EntityDimension, ExpectedTopology, FluidConstraint,
    FluidConstraintKind, PhysicalSignal, RegionTarget, SignalProvenance,
};

#[derive(Clone, Debug, serde::Serialize)]
pub(super) enum FaceBoundary {
    Interior,
    Velocity {
        authority_id: String,
        value: [f64; 3],
        moving_wall: bool,
    },
    PressureOutlet {
        authority_id: String,
        pressure: f64,
    },
}

pub(super) fn compile(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    constraints: &[FluidConstraint],
    operators: &[AnalysisBoundary],
    signals: &[PhysicalSignal],
) -> Result<Vec<FaceBoundary>, Error> {
    let mut ids = std::collections::BTreeSet::new();
    for id in constraints
        .iter()
        .map(|c| c.constraint_id.as_str())
        .chain(operators.iter().map(|b| b.boundary_id.as_str()))
    {
        if id.trim() != id || id.chars().any(char::is_control) || !ids.insert(id) {
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
                let v = value.study_constant(signals).ok_or(Error::Unsupported(
                    "unresolved steady velocity signal/frame",
                ))?;
                (
                    target,
                    FaceBoundary::Velocity {
                        authority_id: c.constraint_id.clone(),
                        value: v.map(|v| v.meters_per_second),
                        moving_wall: matches!(&c.kind, FluidConstraintKind::MovingWall { .. }),
                    },
                )
            }
            FluidConstraintKind::NoSlipWall { target } => (
                target,
                FaceBoundary::Velocity {
                    authority_id: c.constraint_id.clone(),
                    value: [0.0; 3],
                    moving_wall: false,
                },
            ),
            FluidConstraintKind::PressureReference { .. } => {
                return Err(Error::Unsupported(
                    "closed-component pressure gauge requires shared cell nullspace evidence",
                ))
            }
            _ => {
                return Err(Error::Unsupported(
                    "slip, symmetry or periodic fluid boundary",
                ))
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
        let p = value.study_constant(signals).ok_or(Error::Unsupported(
            "unresolved outlet pressure signal/frame",
        ))?;
        bind(
            source,
            t,
            &mut out,
            &b.target,
            FaceBoundary::PressureOutlet {
                authority_id: b.boundary_id.clone(),
                pressure: p.pascals,
            },
        )?;
    }
    for (i, f) in t.faces.iter().enumerate() {
        if f.neighbor.is_none() && matches!(out[i], FaceBoundary::Interior) {
            return Err(Error::Invalid("incomplete exterior boundary coverage"));
        }
    }
    pressure_references(t, &out)?;
    Ok(out)
}

fn bind(
    source: &AnalysisProjection,
    t: &FiniteVolumeTopology,
    out: &mut [FaceBoundary],
    target: &RegionTarget,
    law: FaceBoundary,
) -> Result<(), Error> {
    if target.entity_dimension != EntityDimension::Face
        || target.expected_topology != ExpectedTopology::BoundaryFace
    {
        return Err(Error::Invalid("fluid boundary target kind"));
    }
    let finite = match &law {
        FaceBoundary::Velocity { value, .. } => value.iter().all(|x| x.is_finite()),
        FaceBoundary::PressureOutlet { pressure, .. } => pressure.is_finite(),
        _ => false,
    };
    if !finite {
        return Err(Error::Invalid("boundary value"));
    }
    if target.resolved_entities().is_empty() {
        return Err(Error::Invalid(
            "fluid boundary requires exact resolved entity authority",
        ));
    }
    target
        .validate_resolution()
        .map_err(|_| Error::Invalid("fluid boundary resolved entity authority"))?;
    let mut matched = 0usize;
    let mut covered = vec![false; target.resolved_entities().len()];
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
        if let FaceBoundary::Velocity {
            value,
            moving_wall: true,
            ..
        } = &law
        {
            let normal = super::geometry_ops::dot(*value, f.owner_area_vector_m2).abs();
            let roundoff = 128.0 * f64::EPSILON * super::geometry_ops::norm(*value) * f.area_m2;
            if normal > roundoff {
                return Err(Error::Unsupported(
                    "normal moving-wall motion requires ALE geometry",
                ));
            }
        }
        for entity in &original.provenance {
            if let Ok(j) = target.resolved_entities().binary_search(entity) {
                covered[j] = true;
            }
        }
        if !matches!(out[i], FaceBoundary::Interior) {
            return Err(Error::Invalid("overlapping fluid boundary authorities"));
        }
        out[i] = law.clone();
        matched += 1;
    }
    if matched == 0 || covered.iter().any(|seen| !*seen) {
        return Err(Error::Invalid("empty fluid boundary target"));
    }
    Ok(())
}

pub(super) fn pressure_references(
    t: &FiniteVolumeTopology,
    boundaries: &[FaceBoundary],
) -> Result<Vec<f64>, Error> {
    let mut neighbors = vec![Vec::new(); t.cells.len()];
    for f in &t.faces {
        if let Some(n) = f.neighbor {
            neighbors[f.owner].push(n);
            neighbors[n].push(f.owner);
        }
    }
    let components = connected_components(&neighbors)
        .map_err(|_| Error::Invalid("pressure reference adjacency"))?;
    let mut labels = vec![0; t.cells.len()];
    for (label, component) in components.iter().enumerate() {
        for &c in component {
            labels[c] = label;
        }
    }
    // One traversal, independent of the number of disconnected components.
    // First canonical outlet remains the deterministic component reference.
    let mut outlets = vec![None; components.len()];
    for (i, face) in t.faces.iter().enumerate() {
        if let FaceBoundary::PressureOutlet { pressure, .. } = boundaries[i] {
            outlets[labels[face.owner]].get_or_insert(pressure);
        }
    }
    let outlets = outlets
        .into_iter()
        .map(|p| p.ok_or(Error::Unsupported("physical outlet reference required")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(labels.into_iter().map(|label| outlets[label]).collect())
}
