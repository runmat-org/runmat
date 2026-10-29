use std::collections::BTreeSet;

use super::{
    model::RealizedExcitation, ProjectionScalar, ProjectionVector3, SpatialIntegrationRule,
    SpatialProjectionBalance, SpatialProjectionEvidence,
};
use crate::topology::AnalysisProjection;

#[cfg(test)]
mod tests;

enum ProjectionCoordinates<'a> {
    Canonical(&'a AnalysisProjection),
    Explicit(&'a [[f64; 3]]),
}

impl ProjectionCoordinates<'_> {
    fn at(&self, node_index: usize) -> Option<[f64; 3]> {
        match self {
            Self::Canonical(projection) => projection
                .nodes
                .get(node_index)
                .map(|node| node.coordinates_m.canonical_value()),
            Self::Explicit(coordinates) => coordinates.get(node_index).copied(),
        }
    }
}

pub(crate) struct SpatialProjectionAccumulator<'a> {
    load: &'a RealizedExcitation,
    projection_digest: runmat_execution::Digest,
    coordinates: ProjectionCoordinates<'a>,
    integration_rule: SpatialIntegrationRule,
    resolved_entity_count: usize,
    resolved_nodes: BTreeSet<usize>,
    reference_point_m: [f64; 3],
    requested_resultant: [f64; 3],
    projected_resultant: [f64; 3],
    requested_moment: [f64; 3],
    projected_moment: [f64; 3],
    requested_resultant_scale: f64,
    projected_resultant_scale: f64,
    requested_moment_scale: f64,
    projected_moment_scale: f64,
}

pub(crate) struct ScalarProjectionAccumulator<'a> {
    load: &'a RealizedExcitation,
    projection: &'a AnalysisProjection,
    integration_rule: SpatialIntegrationRule,
    resolved_entity_count: usize,
    resolved_nodes: Vec<bool>,
    resolved_node_count: usize,
    /// Compensated (real, imaginary) sums: the conservation check compares two integrals of
    /// many contributions, so their rounding must not grow with the entity count.
    requested: [crate::algebra::CompensatedSum; 2],
    projected: [crate::algebra::CompensatedSum; 2],
    requested_scale: f64,
    projected_scale: f64,
}

impl<'a> ScalarProjectionAccumulator<'a> {
    pub(crate) fn new(
        load: &'a RealizedExcitation,
        projection: &'a AnalysisProjection,
        integration_rule: SpatialIntegrationRule,
        resolved_entity_count: usize,
    ) -> Result<Self, std::collections::TryReserveError> {
        let mut resolved_nodes = Vec::new();
        resolved_nodes.try_reserve_exact(projection.nodes.len())?;
        resolved_nodes.resize(projection.nodes.len(), false);
        Ok(Self {
            load,
            projection,
            integration_rule,
            resolved_entity_count,
            resolved_nodes,
            resolved_node_count: 0,
            requested: Default::default(),
            projected: Default::default(),
            requested_scale: 0.0,
            projected_scale: 0.0,
        })
    }

    pub(crate) fn add_requested(&mut self, real: f64, imaginary: f64) {
        self.requested[0].add(real);
        self.requested[1].add(imaginary);
        self.requested_scale += real.hypot(imaginary);
    }

    pub(crate) fn add_projected(&mut self, node_index: usize, real: f64, imaginary: f64) {
        if !self.resolved_nodes[node_index] {
            self.resolved_nodes[node_index] = true;
            self.resolved_node_count += 1;
        }
        self.projected[0].add(real);
        self.projected[1].add(imaginary);
        self.projected_scale += real.hypot(imaginary);
    }

    pub(crate) fn finish(self) -> SpatialProjectionEvidence {
        let requested = ProjectionScalar {
            real: self.requested[0].total(),
            imaginary: self.requested[1].total(),
        };
        let projected = ProjectionScalar {
            real: self.projected[0].total(),
            imaginary: self.projected[1].total(),
        };
        // Contribution norms bound the integral in exact arithmetic. Include the
        // retained integral norms as well so independent rounding cannot violate
        // that bound. This retains cancellation scales without a dimensional floor.
        let normalization_scale = self
            .requested_scale
            .max(self.projected_scale)
            .max(requested.real.hypot(requested.imaginary))
            .max(projected.real.hypot(projected.imaginary));
        let error =
            (requested.real - projected.real).hypot(requested.imaginary - projected.imaginary);
        let relative_error = if normalization_scale == 0.0 {
            0.0
        } else {
            error / normalization_scale
        };
        let requirements = self.load.policy.projection;
        SpatialProjectionEvidence {
            excitation_id: self.load.excitation_id.0.clone(),
            signal_id: self.load.signal_id.0.clone(),
            quantity: self.load.quantity,
            sample: self.load.sample.clone(),
            analysis_projection_digest: self.projection.projection_digest,
            integration_rule: self.integration_rule,
            resolved_entity_count: self.resolved_entity_count,
            resolved_node_count: self.resolved_node_count,
            reference_point_m: [0.0; 3],
            balance: SpatialProjectionBalance::ScalarIntegral {
                requested_integral: requested,
                projected_integral: projected,
                normalization_scale,
                relative_error,
            },
            requirements,
            accepted: !requirements.preserve_resultant
                || relative_error <= requirements.relative_tolerance,
        }
    }
}

impl<'a> SpatialProjectionAccumulator<'a> {
    pub(crate) fn new(
        load: &'a RealizedExcitation,
        projection: &'a AnalysisProjection,
        integration_rule: SpatialIntegrationRule,
        resolved_entity_count: usize,
        reference_point_m: [f64; 3],
    ) -> Self {
        Self {
            load,
            projection_digest: projection.projection_digest,
            coordinates: ProjectionCoordinates::Canonical(projection),
            integration_rule,
            resolved_entity_count,
            resolved_nodes: BTreeSet::new(),
            reference_point_m,
            requested_resultant: [0.0; 3],
            projected_resultant: [0.0; 3],
            requested_moment: [0.0; 3],
            projected_moment: [0.0; 3],
            requested_resultant_scale: 0.0,
            projected_resultant_scale: 0.0,
            requested_moment_scale: 0.0,
            projected_moment_scale: 0.0,
        }
    }

    pub(crate) fn for_explicit_topology(
        load: &'a RealizedExcitation,
        projection_digest: runmat_execution::Digest,
        node_coordinates_m: &'a [[f64; 3]],
        integration_rule: SpatialIntegrationRule,
        resolved_entity_count: usize,
        reference_point_m: [f64; 3],
    ) -> Self {
        Self {
            load,
            projection_digest,
            coordinates: ProjectionCoordinates::Explicit(node_coordinates_m),
            integration_rule,
            resolved_entity_count,
            resolved_nodes: BTreeSet::new(),
            reference_point_m,
            requested_resultant: [0.0; 3],
            projected_resultant: [0.0; 3],
            requested_moment: [0.0; 3],
            projected_moment: [0.0; 3],
            requested_resultant_scale: 0.0,
            projected_resultant_scale: 0.0,
            requested_moment_scale: 0.0,
            projected_moment_scale: 0.0,
        }
    }

    pub(crate) fn add_requested_force(&mut self, position_m: [f64; 3], force: [f64; 3]) {
        add(&mut self.requested_resultant, force);
        let arm = subtract(position_m, self.reference_point_m);
        let moment = cross(arm, force);
        add(&mut self.requested_moment, moment);
        self.requested_resultant_scale += norm(force);
        self.requested_moment_scale += norm(moment);
    }

    pub(crate) fn add_requested_moment(&mut self, moment: [f64; 3]) {
        add(&mut self.requested_moment, moment);
        self.requested_moment_scale += norm(moment);
    }

    pub(crate) fn add_projected_moment(&mut self, moment: [f64; 3]) {
        add(&mut self.projected_moment, moment);
        self.projected_moment_scale += norm(moment);
    }

    pub(crate) fn register_nodes(&mut self, nodes: impl IntoIterator<Item = usize>) {
        self.resolved_nodes.extend(nodes);
    }

    pub(crate) fn add_projected_force(&mut self, node_index: usize, force: [f64; 3]) {
        self.resolved_nodes.insert(node_index);
        self.add_projected_generalized_force(force);
        let coordinate = self
            .coordinates
            .at(node_index)
            .expect("projection accumulator nodes must belong to its topology");
        let arm = subtract(coordinate, self.reference_point_m);
        let moment = cross(arm, force);
        add(&mut self.projected_moment, moment);
        self.projected_moment_scale += norm(moment);
    }

    pub(crate) fn add_projected_generalized_force(&mut self, force: [f64; 3]) {
        add(&mut self.projected_resultant, force);
        self.projected_resultant_scale += norm(force);
    }

    pub(crate) fn finish(self) -> SpatialProjectionEvidence {
        let resultant_scale = self
            .requested_resultant_scale
            .max(self.projected_resultant_scale)
            .max(norm(self.requested_resultant))
            .max(norm(self.projected_resultant));
        let moment_scale = self
            .requested_moment_scale
            .max(self.projected_moment_scale)
            .max(norm(self.requested_moment))
            .max(norm(self.projected_moment));
        let resultant_relative_error = relative_vector_error(
            self.requested_resultant,
            self.projected_resultant,
            resultant_scale,
        );
        let moment_relative_error =
            relative_vector_error(self.requested_moment, self.projected_moment, moment_scale);
        let requirements = self.load.policy.projection;
        let accepted = (!requirements.preserve_resultant
            || resultant_relative_error <= requirements.relative_tolerance)
            && (!requirements.preserve_moment
                || moment_relative_error <= requirements.relative_tolerance);
        SpatialProjectionEvidence {
            excitation_id: self.load.excitation_id.0.clone(),
            signal_id: self.load.signal_id.0.clone(),
            quantity: self.load.quantity,
            sample: self.load.sample.clone(),
            analysis_projection_digest: self.projection_digest,
            integration_rule: self.integration_rule,
            resolved_entity_count: self.resolved_entity_count,
            resolved_node_count: self.resolved_nodes.len(),
            reference_point_m: self.reference_point_m,
            balance: SpatialProjectionBalance::VectorResultant {
                requested_resultant: ProjectionVector3 {
                    real: self.requested_resultant,
                    imaginary: [0.0; 3],
                },
                projected_resultant: ProjectionVector3 {
                    real: self.projected_resultant,
                    imaginary: [0.0; 3],
                },
                requested_moment: ProjectionVector3 {
                    real: self.requested_moment,
                    imaginary: [0.0; 3],
                },
                projected_moment: ProjectionVector3 {
                    real: self.projected_moment,
                    imaginary: [0.0; 3],
                },
                resultant_scale,
                moment_scale,
                resultant_relative_error,
                moment_relative_error,
            },
            requirements,
            accepted,
        }
    }
}

fn relative_vector_error(requested: [f64; 3], projected: [f64; 3], scale: f64) -> f64 {
    if scale == 0.0 {
        0.0
    } else {
        norm(subtract(requested, projected)) / scale
    }
}

fn add(target: &mut [f64; 3], value: [f64; 3]) {
    for axis in 0..3 {
        target[axis] += value[axis];
    }
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}
