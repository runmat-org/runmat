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
    pub area_m2: f64,
    /// Outward from the original fluid domain; inlet flux is negative.
    pub volume_flow_rate_m3_per_s: f64,
    pub mass_flow_rate_kg_per_s: f64,
    pub advective_momentum_n: V,
    pub viscous_traction_n: V,
    pub pressure_traction_n: V,
    pub pressure_fluctuation_traction_n: V,
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
        | FaceBoundary::PressureOutlet { authority_id, .. } => Some(authority_id),
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
                area_m2: 0.0,
                volume_flow_rate_m3_per_s: 0.0,
                mass_flow_rate_kg_per_s: 0.0,
                advective_momentum_n: [0.0; 3],
                viscous_traction_n: [0.0; 3],
                pressure_traction_n: [0.0; 3],
                pressure_fluctuation_traction_n: [0.0; 3],
            })
            .collect::<Vec<_>>();
        for &(face, group) in &self.faces {
            let r = &mut reports[group];
            r.area_m2 += p.topology.faces[face].area_m2;
            r.volume_flow_rate_m3_per_s += q[face];
            r.mass_flow_rate_kg_per_s += p.density * q[face];
            r.advective_momentum_n = add(r.advective_momentum_n, forces.advective_momentum_n[face]);
            r.viscous_traction_n = add(r.viscous_traction_n, forces.viscous_traction_n[face]);
            r.pressure_traction_n = add(r.pressure_traction_n, forces.pressure_traction_n[face]);
            r.pressure_fluctuation_traction_n = add(
                r.pressure_fluctuation_traction_n,
                forces.pressure_fluctuation_traction_n[face],
            );
        }
        if reports.iter().any(|r| {
            [
                r.area_m2,
                r.volume_flow_rate_m3_per_s,
                r.mass_flow_rate_kg_per_s,
            ]
            .iter()
            .chain(&r.advective_momentum_n)
            .chain(&r.viscous_traction_n)
            .chain(&r.pressure_traction_n)
            .chain(&r.pressure_fluctuation_traction_n)
            .any(|v| !v.is_finite())
        }) {
            return Err(Error::Invalid("nonfinite original boundary balance"));
        }
        Ok(reports)
    }
}
