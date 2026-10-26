//! Lowest-order Raviart–Thomas lift of the retained finite-volume face fluxes.
//!
//! For a tetrahedron of volume V, the basis of face f with opposite vertex x_f is
//! (x − x_f) / (3V): it carries unit outward flux through f and none through the other faces.
//! The lifted velocity Σ Q_f (x − x_f) / (3V) therefore reproduces every retained face flux,
//! has divergence Σ Q_f / V (zero when the cell conserves mass) and continuous normal
//! components between neighboring cells.
use super::ConjugateHeatError as Error;
use crate::topology::{AnalysisProjection, FiniteVolumeTopology};

#[derive(Clone, Copy, Debug)]
pub(super) struct CellVelocity {
    inverse_three_volume: f64,
    faces: [([f64; 3], f64); 4],
}

impl CellVelocity {
    pub(super) fn at(&self, x: [f64; 3]) -> [f64; 3] {
        let mut u = [0.0; 3];
        for (vertex, flux) in self.faces {
            for k in 0..3 {
                u[k] += flux * (x[k] - vertex[k]) * self.inverse_three_volume;
            }
        }
        u
    }
}

/// Per projection element: the lifted velocity of fluid cells, `None` for solid elements.
pub(super) fn lift(
    projection: &AnalysisProjection,
    topology: &FiniteVolumeTopology,
    flux: &[f64],
) -> Result<Vec<Option<CellVelocity>>, Error> {
    if flux.len() != topology.faces.len() || flux.iter().any(|q| !q.is_finite()) {
        return Err(Error::Invalid("retained face flux inventory"));
    }
    topology
        .validate_for(projection)
        .map_err(|_| Error::Invalid("fluid cell complex differs from the projection"))?;
    let mut cells = vec![None; projection.elements.len()];
    for cell in &topology.cells {
        let element = &projection.elements[cell.source_element_index];
        let corners = element
            .node_indices
            .get(..4)
            .ok_or(Error::Invalid("fluid cell corners"))?;
        let mut faces = [([0.0; 3], 0.0); 4];
        for (slot, incidence) in faces.iter_mut().zip(cell.faces) {
            let face = &topology.faces[incidence.face_index];
            let mut opposite = corners.iter().filter(|&&n| {
                !face
                    .corner_identities
                    .contains(&projection.nodes[n].stable_identity)
            });
            let (Some(&vertex), None) = (opposite.next(), opposite.next()) else {
                return Err(Error::Invalid(
                    "fluid cell face has no unique opposite vertex",
                ));
            };
            *slot = (
                projection.nodes[vertex].coordinates_m.canonical_value(),
                f64::from(incidence.sign) * flux[incidence.face_index],
            );
        }
        if !cell.volume_m3.is_finite() || cell.volume_m3 <= 0.0 {
            return Err(Error::Invalid("fluid cell volume"));
        }
        cells[cell.source_element_index] = Some(CellVelocity {
            inverse_three_volume: 1.0 / (3.0 * cell.volume_m3),
            faces,
        });
    }
    Ok(cells)
}
