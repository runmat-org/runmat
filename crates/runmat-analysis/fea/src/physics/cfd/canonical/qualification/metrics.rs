//! Original volume-weighted cell-centroid and boundary traction comparisons.
use super::super::{
    geometry_ops::{norm, sub},
    CanonicalFluidResidual, CanonicalFluidResult,
};
use super::{fixture, reference};
pub fn cell_centroid_velocity_error(f: &fixture::Fixture, r: &CanonicalFluidResult) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (c, u) in f
        .topology
        .cells
        .iter()
        .zip(r.velocity_m_per_s.canonical_values())
    {
        let exact = reference::velocity(f.reference, c.centroid_m, f.rotated);
        numerator += c.volume_m3 * norm(sub(*u, exact)).powi(2);
        denominator += c.volume_m3 * norm(exact).powi(2);
    }
    (numerator / denominator).sqrt()
}

pub fn wall_shear_error(f: &fixture::Fixture, r: &CanonicalFluidResidual) -> f64 {
    let mut error = 0.0;
    let mut exact = 0.0;
    for (i, face) in f.topology.faces.iter().enumerate() {
        if face.neighbor.is_some() || face.owner_area_vector_m2[0] != 0.0 {
            continue;
        }
        let gradient = reference::velocity_gradient(f.reference, face.centroid_m);
        let traction = std::array::from_fn(|k| {
            (0..3)
                .map(|j| (gradient[k][j] + gradient[j][k]) * face.owner_area_vector_m2[j])
                .sum::<f64>()
        });
        error += norm(sub(
            r.face_forces.viscous_traction_n.canonical_values()[i],
            traction,
        ))
        .powi(2)
            / face.area_m2.canonical_value();
        exact += norm(traction).powi(2) / face.area_m2.canonical_value();
    }
    (error / exact).sqrt()
}
