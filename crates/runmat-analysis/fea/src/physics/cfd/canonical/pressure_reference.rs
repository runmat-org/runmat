//! Gauge authority is separate from physical wall/outlet laws.
use super::{
    boundary::{self, FaceBoundary},
    geometry_ops::*,
    CanonicalFluidError as Error, CanonicalFluidProblem,
};
use crate::{
    algebra::connected_components,
    topology::{AnalysisProjection, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    FluidConstraint, FluidConstraintKind, PhysicalSignal, SignalProvenance,
};

#[derive(Clone, serde::Serialize)]
struct GaugeComponent {
    authority_id: String,
    cells: Vec<usize>,
    faces: Vec<(usize, f64)>,
}
#[derive(Default, serde::Serialize)]
pub(super) struct PressureReferencePlan {
    closed: Vec<GaugeComponent>,
}
impl PressureReferencePlan {
    pub(super) fn derive(
        source: &AnalysisProjection,
        t: &FiniteVolumeTopology,
        b: &[FaceBoundary],
        constraints: &[FluidConstraint],
        signals: &[PhysicalSignal],
        sample: crate::excitation::ExcitationSamplePoint<'_>,
    ) -> Result<(Self, Vec<f64>), Error> {
        let mut neighbors = vec![Vec::new(); t.cells.len()];
        for f in &t.faces {
            if let Some(n) = f.neighbor {
                neighbors[f.owner].push(n);
                neighbors[n].push(f.owner);
            }
        }
        let components = connected_components(&neighbors)
            .map_err(|_| Error::Invalid("pressure component adjacency"))?;
        let mut labels = vec![0; t.cells.len()];
        for (i, cells) in components.iter().enumerate() {
            for &c in cells {
                labels[c] = i;
            }
        }
        let mut outlets = vec![None; components.len()];
        for (i, f) in t.faces.iter().enumerate() {
            if let FaceBoundary::PressureOutlet { pressure, .. } = b[i] {
                outlets[labels[f.owner]].get_or_insert(pressure);
            }
        }
        let mut gauges = vec![None; components.len()];
        for constraint in constraints {
            let FluidConstraintKind::PressureReference { target, value } = &constraint.kind else {
                continue;
            };
            if value
                .signal(signals)
                .is_some_and(|s| s.provenance() != &SignalProvenance::Authored)
            {
                return Err(Error::Unsupported("unauthenticated pressure gauge signal"));
            }
            let value = crate::excitation::sample_prescribed_pressure(value, signals, sample)
                .map_err(|e| Error::InputRealization(e.to_string()))?;
            if value.imaginary != 0.0 {
                return Err(Error::Unsupported("complex fluid gauge"));
            }
            let value = value.real;
            let selected = boundary::select_faces(source, t, target)?;
            let mut by_component = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for i in selected {
                by_component
                    .entry(labels[t.faces[i].owner])
                    .or_default()
                    .push(i);
            }
            for (label, faces) in by_component {
                if outlets[label].is_some() || gauges[label].is_some() {
                    return Err(Error::Invalid(
                        "duplicate gauge or gauge on physical outlet component",
                    ));
                }
                let area = faces
                    .iter()
                    .map(|i| t.faces[*i].area_m2.canonical_value())
                    .sum::<f64>();
                if !area.is_finite() || area <= 0.0 {
                    return Err(Error::Invalid("pressure gauge area"));
                }
                gauges[label] = Some((
                    value,
                    GaugeComponent {
                        authority_id: constraint.constraint_id.clone(),
                        cells: components[label].clone(),
                        faces: faces
                            .into_iter()
                            .map(|i| (i, t.faces[i].area_m2.canonical_value() / area))
                            .collect(),
                    },
                ));
            }
        }
        // One exterior traversal across all components, including disconnected
        // closed regions; never rescan every face for every gauge.
        let mut flux_sum = vec![0.0; components.len()];
        let mut flux_scale = vec![0.0; components.len()];
        for (i, f) in t.faces.iter().enumerate() {
            if f.neighbor.is_none() && outlets[labels[f.owner]].is_none() {
                // Velocity faces carry their prescribed flux; symmetry planes and slip walls
                // are impermeable.
                let flux = match b[i] {
                    FaceBoundary::Velocity { value, .. } => dot(value, f.owner_area_vector_m2),
                    FaceBoundary::Symmetry { .. } => 0.0,
                    _ => return Err(Error::Invalid("closed pressure component face law")),
                };
                flux_sum[labels[f.owner]] += flux;
                flux_scale[labels[f.owner]] += flux.abs();
            }
        }
        let mut references = vec![0.0; t.cells.len()];
        let mut plan = Self::default();
        for (label, cells) in components.iter().enumerate() {
            let reference = if let Some(outlet) = outlets[label] {
                outlet
            } else {
                let (value, gauge) = gauges[label].take().ok_or(Error::Unsupported(
                    "closed fluid component requires authored pressure gauge",
                ))?;
                // Closed flux is fixed by physical velocity faces, independently
                // of the gauge. Reject incompatible data before operator reduction.
                if !flux_sum[label].is_finite()
                    || !flux_scale[label].is_finite()
                    || flux_sum[label].abs() > 128.0 * f64::EPSILON * flux_scale[label]
                {
                    return Err(Error::Invalid(
                        "incompatible closed-component boundary volume flux",
                    ));
                }
                plan.closed.push(gauge);
                value
            };
            for &c in cells {
                references[c] = reference;
            }
        }
        Ok((plan, references))
    }
    pub(super) fn roots(&self) -> Vec<usize> {
        self.closed.iter().map(|c| c.cells[0]).collect()
    }
    pub(super) fn normalize(
        &self,
        p: &CanonicalFluidProblem<'_>,
        values: &mut [f64],
    ) -> Result<(), Error> {
        if self.closed.is_empty() {
            return Ok(());
        }
        let gradients = pressure_gradient(p, values)?;
        for component in &self.closed {
            let mut mean = 0.0;
            for &(face, weight) in &component.faces {
                mean += weight * face_pressure_offset(p, face, values, &gradients)?;
            }
            if !mean.is_finite() {
                return Err(Error::Invalid("nonfinite pressure gauge mean"));
            }
            for &c in &component.cells {
                values[c] -= mean;
            }
        }
        Ok(())
    }
    pub(super) fn validate(
        &self,
        p: &CanonicalFluidProblem<'_>,
        values: &[f64],
    ) -> Result<(), Error> {
        if self.closed.is_empty() {
            return Ok(());
        }
        let gradients = pressure_gradient(p, values)?;
        for component in &self.closed {
            let mut mean = 0.0;
            let mut scale = component
                .cells
                .iter()
                .map(|c| values[*c].abs())
                .fold(0.0_f64, f64::max);
            for &(face, weight) in &component.faces {
                let v = face_pressure_offset(p, face, values, &gradients)?;
                mean += weight * v;
                scale += weight * v.abs();
            }
            if !mean.is_finite()
                || mean.abs() > 1024.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE)
            {
                return Err(Error::Invalid("retained original pressure gauge"));
            }
        }
        Ok(())
    }
}
