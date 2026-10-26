//! Original exterior-face terms grouped by exact authored boundary authority.
//! This is conservation attribution, never an independent spatial-error estimate.
use super::{
    boundary::FaceBoundary, geometry_ops::*, CanonicalFluidError as Error,
    CanonicalFluidFaceForces, CanonicalFluidProblem,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidBoundaryBalance {
    pub authority_id: String,
    pub face_count: u64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub area_m2: runmat_types::physical::Area,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// Outward from the original fluid domain; inlet flux is negative.
    pub volume_flow_rate_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub mass_flow_rate_kg_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::MassFlowRate, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub advective_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, V>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub viscous_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, V>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub pressure_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, V>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub pressure_fluctuation_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, V>,
}

pub(super) struct BoundaryBalancePlan {
    /// Exact original boundary face and its report index, in authority/face order.
    faces: Vec<(usize, usize)>,
    /// Representative original face and total face count. No copied ID registry.
    groups: Vec<(usize, usize)>,
}

fn authority<'a>(boundary: &FaceBoundary<'a>) -> Option<&'a str> {
    match boundary {
        FaceBoundary::Interior => None,
        FaceBoundary::Velocity { authority_id, .. }
        | FaceBoundary::PressureOutlet { authority_id, .. }
        | FaceBoundary::Symmetry { authority_id } => Some(authority_id),
    }
}

impl BoundaryBalancePlan {
    pub(super) fn derive(boundaries: &[FaceBoundary]) -> Self {
        let mut faces = boundaries
            .iter()
            .enumerate()
            .filter_map(|(i, b)| authority(b).map(|_| (i, 0)))
            .collect::<Vec<_>>();
        faces.sort_unstable_by(|(a, _), (b, _)| {
            authority(&boundaries[*a])
                .cmp(&authority(&boundaries[*b]))
                .then(a.cmp(b))
        });
        let mut groups: Vec<(usize, usize)> = Vec::new();
        for (face, group) in &mut faces {
            if groups.last().is_none_or(|(previous, _)| {
                authority(&boundaries[*previous]) != authority(&boundaries[*face])
            }) {
                groups.push((*face, 0));
            }
            *group = groups.len() - 1;
            groups[*group].1 += 1;
        }
        Self { faces, groups }
    }

    pub(super) fn authorities<'a>(
        &'a self,
        boundaries: &'a [FaceBoundary],
    ) -> impl Iterator<Item = &'a str> + Clone {
        self.groups
            .iter()
            .map(move |(face, _)| authority(&boundaries[*face]).unwrap())
    }

    pub(super) fn evaluate(
        &self,
        p: &CanonicalFluidProblem<'_>,
        q: &[f64],
        forces: &CanonicalFluidFaceForces,
    ) -> Result<Vec<CanonicalFluidBoundaryBalance>, Error> {
        let mut reports = self
            .groups
            .iter()
            .map(|(face, count)| CanonicalFluidBoundaryBalance {
                authority_id: authority(&p.boundaries[*face]).unwrap().to_owned(),
                face_count: *count as u64,
                area_m2: runmat_types::physical::Area::new(0.0),
                volume_flow_rate_m3_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::VolumeFlowRate,
                    _,
                >::new(0.0),
                mass_flow_rate_kg_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::MassFlowRate,
                    _,
                >::new(0.0),
                advective_momentum_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                viscous_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                pressure_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
                pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Force,
                    _,
                >::new([0.0; 3]),
            })
            .collect::<Vec<_>>();
        for &(face, group) in &self.faces {
            let r = &mut reports[group];
            r.area_m2 = runmat_types::physical::Quantity::new(
                r.area_m2.canonical_value() + (p.topology.faces[face].area_m2.canonical_value()),
            );
            r.volume_flow_rate_m3_per_s += runmat_types::physical::VolumeFlowRate::new(q[face]);
            r.mass_flow_rate_kg_per_s +=
                runmat_types::physical::MassFlowRate::new(p.density * q[face]);
            r.advective_momentum_n = runmat_types::physical::Quantity::new(add(
                r.advective_momentum_n.canonical_value(),
                forces.advective_momentum_n.canonical_values()[face],
            ));
            r.viscous_traction_n = runmat_types::physical::Quantity::new(add(
                r.viscous_traction_n.canonical_value(),
                forces.viscous_traction_n.canonical_values()[face],
            ));
            r.pressure_traction_n = runmat_types::physical::Quantity::new(add(
                r.pressure_traction_n.canonical_value(),
                forces.pressure_traction_n.canonical_values()[face],
            ));
            r.pressure_fluctuation_traction_n = runmat_types::physical::Quantity::new(add(
                r.pressure_fluctuation_traction_n.canonical_value(),
                forces.pressure_fluctuation_traction_n.canonical_values()[face],
            ));
        }
        if reports.iter().any(|r| {
            [
                r.area_m2.canonical_value(),
                r.volume_flow_rate_m3_per_s.canonical_value(),
                r.mass_flow_rate_kg_per_s.canonical_value(),
            ]
            .iter()
            .chain(r.advective_momentum_n.canonical_values())
            .chain(r.viscous_traction_n.canonical_values())
            .chain(r.pressure_traction_n.canonical_values())
            .chain(r.pressure_fluctuation_traction_n.canonical_values())
            .any(|v| !v.is_finite())
        }) {
            return Err(Error::Invalid("nonfinite original boundary balance"));
        }
        Ok(reports)
    }
}
