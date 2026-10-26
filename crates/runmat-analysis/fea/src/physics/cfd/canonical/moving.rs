//! Swept face volumes between two configurations of one cell complex, for moving-mesh
//! intervals. Corners move linearly in time, so the swept volume of each affine triangle is
//! integrated exactly and the cell volume change equals the sum of its faces' swept volumes.
use super::CanonicalFluidError as Error;
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};
use std::collections::BTreeMap;

/// Swept volume of every face of `arrival` over the interval, divided by `duration_s`,
/// outward from the face owner. `departure` and `arrival` project the same nodes.
pub fn swept_face_flux(
    departure: &AnalysisProjection,
    arrival: &AnalysisProjection,
    topology: &FiniteVolumeTopology,
    duration_s: f64,
) -> Result<Vec<f64>, Error> {
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return Err(Error::Invalid("moving-mesh interval duration"));
    }
    if departure.nodes.len() != arrival.nodes.len() {
        return Err(Error::Invalid("moving-mesh node inventories differ"));
    }
    let index = arrival
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.stable_identity, i))
        .collect::<BTreeMap<_, _>>();
    let mut flux = Vec::with_capacity(topology.faces.len());
    for face in &topology.faces {
        let mut old = [[0.0; 3]; 3];
        let mut new = [[0.0; 3]; 3];
        for (k, identity) in face.corner_identities.iter().enumerate() {
            let &i = index
                .get(identity)
                .ok_or(Error::Invalid("moving-mesh face corner"))?;
            if departure.nodes[i].stable_identity != *identity {
                return Err(Error::Invalid("moving-mesh node order differs"));
            }
            old[k] = departure.nodes[i].coordinates_m.canonical_value();
            new[k] = arrival.nodes[i].coordinates_m.canonical_value();
        }
        let a0 = sub(old[1], old[0]);
        let b0 = sub(old[2], old[0]);
        let a1 = sub(new[1], new[0]);
        let b1 = sub(new[2], new[0]);
        let n1 = cross(a1, b1);
        let mixed = add(cross(a0, b1), cross(a1, b0));
        let normal = std::array::from_fn::<f64, 3, _>(|k| {
            0.5 * (cross(a0, b0)[k] / 3.0 + n1[k] / 3.0 + mixed[k] / 6.0)
        });
        let mean = std::array::from_fn::<f64, 3, _>(|k| {
            (new[0][k] - old[0][k] + new[1][k] - old[1][k] + new[2][k] - old[2][k]) / 3.0
        });
        let orientation = if dot(n1, face.owner_area_vector_m2) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let swept = orientation * dot(mean, normal) / duration_s;
        if !swept.is_finite() {
            return Err(Error::Invalid("nonfinite swept face volume"));
        }
        flux.push(swept);
    }
    Ok(flux)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
