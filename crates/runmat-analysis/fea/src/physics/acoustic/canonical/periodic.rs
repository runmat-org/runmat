//! Pressure phase relations reuse canonical geometry pairing and shared affine algebra.

use super::{assembly::AcousticAssembly, AcousticHarmonicError};
use crate::constraints::{
    build_phase_reduction, pair_targets, resolve_nodal_target, UnitPhaseRelation,
};
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations};
use runmat_analysis_core::{
    AcousticConstraintKind, AnalysisConstraint, AnalysisModel, PairingPolicy, PeriodicTransform,
    ProcedureStep, RegionTarget, StepObjectRef,
};
use runmat_meshing_core::{BoundaryFaceRole, SolverMeshArtifact};
use std::collections::{BTreeMap, BTreeSet};

impl AcousticAssembly {
    pub(super) fn with_periodic_constraints(
        mut self,
        mesh: &SolverMeshArtifact,
        model: &AnalysisModel,
        step: &ProcedureStep,
        excitations: &RealizedExcitations,
    ) -> Result<Self, AcousticHarmonicError> {
        if self.projection.mesh_digest != mesh.canonical_digest {
            return Err(AcousticHarmonicError::Topology(
                "periodic pairing mesh differs from the pressure projection".into(),
            ));
        }
        let node_indices = self
            .projection
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.stable_identity, index))
            .collect::<BTreeMap<_, _>>();
        let mut relations = Vec::new();
        let mut paired_faces = BTreeSet::new();
        for authored in &model.constraints {
            if !step.object_is_active(&StepObjectRef::Constraint(authored.constraint_id().into())) {
                continue;
            }
            let AnalysisConstraint::Acoustic(constraint) = authored else {
                continue;
            };
            let (periodic, phase_radians) = match &constraint.kind {
                AcousticConstraintKind::Periodic(periodic) => (periodic, 0.0),
                AcousticConstraintKind::PhaseShiftedPeriodic(bloch) => {
                    (&bloch.periodic, bloch.phase.canonical_value())
                }
                AcousticConstraintKind::PrescribedPressure { .. } => continue,
            };
            if !matches!(periodic.pair.pairing, PairingPolicy::Matching)
                || !matches!(
                    periodic.transform,
                    PeriodicTransform::Identity | PeriodicTransform::Translation(_)
                )
            {
                return Err(AcousticHarmonicError::Unsupported(
                    crate::admission::ACOUSTIC_ROTATED_PERIODIC.message.into(),
                ));
            }
            for target in [&periodic.pair.primary, &periodic.pair.secondary] {
                let faces = self
                    .projection
                    .boundary_faces_for_target(target)
                    .collect::<Vec<_>>();
                if faces.is_empty()
                    || faces
                        .iter()
                        .any(|face| face.role != BoundaryFaceRole::Exterior)
                {
                    return Err(AcousticHarmonicError::Boundary(
                        "periodic pressure requires exterior face targets".into(),
                    ));
                }
                for face in faces {
                    if !paired_faces.insert(face.face_id) {
                        return Err(AcousticHarmonicError::Boundary(
                            "overlapping periodic pressure faces".into(),
                        ));
                    }
                }
            }
            let primary =
                resolve_nodal_target(mesh, &periodic.pair.primary, &constraint.constraint_id)
                    .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
            let secondary =
                resolve_nodal_target(mesh, &periodic.pair.secondary, &constraint.constraint_id)
                    .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
            for pairing in pair_targets(
                mesh,
                &primary,
                &secondary,
                &periodic.pair.pairing,
                &periodic.transform,
                &constraint.constraint_id,
            )
            .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?
            {
                let [weight] = pairing.primary_weights.as_slice() else {
                    return Err(AcousticHarmonicError::Boundary(
                        "periodic pressure pairing is not bijective".into(),
                    ));
                };
                if weight.weight != 1.0 {
                    return Err(AcousticHarmonicError::Boundary(
                        "periodic pressure pairing is not matching".into(),
                    ));
                }
                relations.push(UnitPhaseRelation {
                    primary: *node_indices
                        .get(&weight.primary_node_identity)
                        .ok_or_else(|| {
                            AcousticHarmonicError::Topology(
                                "periodic primary node is missing".into(),
                            )
                        })?,
                    secondary: *node_indices
                        .get(&pairing.secondary_node_identity)
                        .ok_or_else(|| {
                            AcousticHarmonicError::Topology(
                                "periodic secondary node is missing".into(),
                            )
                        })?,
                    phase_radians: runmat_types::physical::Angle::new(phase_radians),
                });
            }
        }
        if relations.is_empty() {
            return Ok(self);
        }
        let overlaps = |target: &RegionTarget| {
            self.projection
                .boundary_faces_for_target(target)
                .any(|face| paired_faces.contains(&face.face_id))
        };
        if model.boundaries.iter().any(|boundary| {
            step.object_is_active(&StepObjectRef::Boundary(boundary.boundary_id.clone()))
                && overlaps(&boundary.target)
        }) {
            return Err(AcousticHarmonicError::Boundary(
                "periodic pressure overlaps natural boundary".into(),
            ));
        }
        for load in excitations.iter() {
            if matches!(
                load.kind,
                RealizedExcitationKind::NormalVelocity(_)
                    | RealizedExcitationKind::VolumeVelocity(_)
                    | RealizedExcitationKind::AcousticIncidentPressure(_)
            ) {
                if let RealizedExcitationTarget::Region(target) = &load.target {
                    if overlaps(target) {
                        return Err(AcousticHarmonicError::Boundary(
                            "periodic pressure overlaps natural acoustic source".into(),
                        ));
                    }
                }
            }
        }
        let reduction = build_phase_reduction(
            self.projection.nodes.len(),
            &relations,
            self.prescribed.canonical_values(),
        )
        .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
        // A prescribed representative fixes its entire phase component. Mark those nodes
        // for reaction power recovery; reactions on free paired nodes cancel in aggregate.
        let size = self.projection.nodes.len();
        for node in 0..size {
            let fixed = [node, node + size].into_iter().all(|row| {
                reduction.recovery_row_offsets[row] == reduction.recovery_row_offsets[row + 1]
            });
            if fixed {
                self.prescribed
                    .try_replace(
                        node,
                        runmat_types::physical::Quantity::new(Some(
                            crate::excitation::ProjectionScalar {
                                real: reduction.offset[node],
                                imaginary: reduction.offset[node + size],
                            },
                        )),
                    )
                    .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))?;
            }
        }
        self.reduction = Some(reduction);
        Ok(self)
    }
}
