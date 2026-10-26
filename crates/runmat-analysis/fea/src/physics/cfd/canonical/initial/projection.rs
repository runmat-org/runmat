//! Exact affine Tet4 cell means and oriented triangle flux from ORIGINAL nodal data.
use super::*;
use std::collections::BTreeMap;
pub(super) struct ProjectedInitialFields {
    pub velocity_m_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Velocity, Vec<[f64; 3]>>,
    pub pressure_fluctuation_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    pub face_volume_flux_m3_per_s:
        runmat_types::physical::Quantity<runmat_types::physical::kind::VolumeFlowRate, Vec<f64>>,
}
pub(super) fn integrate(
    p: &CanonicalFluidProblem<'_>,
    velocity: &[[f64; 3]],
    pressure: &[Option<f64>],
) -> Result<ProjectedInitialFields, Error> {
    let index = p
        .source
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.stable_identity, i))
        .collect::<BTreeMap<_, _>>();
    let mut u = Vec::with_capacity(p.topology.cells.len());
    let mut fluctuation = Vec::with_capacity(p.topology.cells.len());
    for (row, cell) in p.topology.cells.iter().enumerate() {
        let element = p
            .source
            .elements
            .get(cell.source_element_index)
            .ok_or(Error::Invalid("initial cell source"))?;
        if element.order != runmat_meshing_core::ElementOrder::Tet4
            || element.node_indices.len() != 4
        {
            return Err(Error::Unsupported(
                "initial physical interpolation requires affine Tet4",
            ));
        }
        let mut mean = [0.0_f64; 3];
        let mut absolute = 0.;
        let mut assigned = 0;
        for &n in &element.node_indices {
            let v = velocity
                .get(n)
                .ok_or(Error::Invalid("initial nodal velocity row"))?;
            for k in 0..3 {
                mean[k] += 0.25 * v[k];
            }
            if let Some(v) = pressure[n] {
                absolute += 0.25 * v;
                assigned += 1;
            }
        }
        if assigned != 0 && assigned != 4 {
            return Err(Error::Invalid("partially assigned initial pressure cell"));
        }
        let value = if assigned == 0 {
            0.
        } else {
            absolute - p.pressure_reference_pa.canonical_values()[row]
        };
        if !value.is_finite() || mean.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("nonfinite original initial cell mean"));
        }
        u.push(mean);
        fluctuation.push(value);
    }
    let mut q = Vec::with_capacity(p.topology.faces.len());
    for face in &p.topology.faces {
        let mut mean = [0.0_f64; 3];
        for identity in face.corner_identities {
            let &n = index
                .get(&identity)
                .ok_or(Error::Invalid("initial face corner identity"))?;
            for (component, value) in mean.iter_mut().zip(velocity[n]) {
                *component += value / 3.;
            }
        }
        let flux = super::super::geometry_ops::dot(mean, face.owner_area_vector_m2);
        if !flux.is_finite() {
            return Err(Error::Invalid("nonfinite original initial face flux"));
        }
        q.push(flux);
    }
    Ok(ProjectedInitialFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(u),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(fluctuation),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(q),
    })
}
