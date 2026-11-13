//! Typed scalar-potential sources, integrated with the shared scalar projection owner:
//! volume charge (C) for electrostatics and terminal current injected through boundary
//! faces (A) for steady conduction. Cold validation may rederive the source, but never a
//! potential/field or stiffness solve.
use super::ElectricalStaticError;
use crate::{
    excitation::{
        realize_step_excitations, ExcitationSamplePoint, RealizedExcitationKind,
        RealizedExcitationTarget, ScalarProjectionAccumulator, SpatialIntegrationRule,
        SpatialProjectionEvidence,
    },
    topology::AnalysisProjection,
};
use runmat_analysis_core::{
    AnalysisModel, CoordinateFrame, EntityDimension, ExcitationAction, ExcitationDistribution,
    ExcitationKinematics, ProcedureStep,
};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalElectricalSource {
    SourceFree,
    VolumeCharge,
    /// Total current injected through exterior boundary faces, uniform normal density.
    BoundaryCurrent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalChargeSources {
    /// Consistent nodal source: charge in C (electrostatics) or injected current in A
    /// (steady conduction). Positive values enter the domain.
    pub nodal_source: Vec<f64>,
    pub integrated_source: f64,
    /// Sum of absolute quadrature contributions, preserving cancellation scale.
    pub absolute_source: f64,
    pub projection_evidence: Vec<SpatialProjectionEvidence>,
}

pub(crate) fn kind(model: &AnalysisModel) -> CanonicalElectricalSource {
    if model.excitations.iter().any(|action| {
        matches!(
            action,
            ExcitationAction::ElectricCurrent(_) | ExcitationAction::CurrentDensity(_)
        )
    }) {
        CanonicalElectricalSource::BoundaryCurrent
    } else if model
        .excitations
        .iter()
        .any(|action| matches!(action, ExcitationAction::ElectricChargeDensity(_)))
    {
        CanonicalElectricalSource::VolumeCharge
    } else {
        CanonicalElectricalSource::SourceFree
    }
}

pub(crate) fn assemble(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    step: &ProcedureStep,
    projection: &AnalysisProjection,
    sample: ExcitationSamplePoint<'_>,
) -> Result<Option<ElectricalChargeSources>, ElectricalStaticError> {
    if kind(model) == CanonicalElectricalSource::SourceFree {
        return Ok(None);
    }
    let loads = realize_step_excitations(model, step, sample)
        .map_err(|e| ElectricalStaticError::InvalidInput(e.to_string()))?;
    let raw = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| (element.stable_identity, element))
        .collect::<BTreeMap<_, _>>();
    let mut source = ElectricalChargeSources {
        nodal_source: vec![0.0; projection.nodes.len()],
        integrated_source: 0.0,
        absolute_source: 0.0,
        projection_evidence: Vec::with_capacity(loads.len()),
    };
    let source_kind = kind(model);
    for load in loads.iter() {
        // Terminal voltages are essential conditions, applied with the constraints.
        if matches!(load.kind, RealizedExcitationKind::TerminalVoltage(_)) {
            continue;
        }
        if source_kind == CanonicalElectricalSource::BoundaryCurrent {
            if let RealizedExcitationKind::CurrentDensity(value) = load.kind {
                add_boundary_current_density(load, value, projection, &mut source)?;
                continue;
            }
            let RealizedExcitationKind::ElectricCurrent(value) = load.kind else {
                return Err(ElectricalStaticError::Unsupported(
                    "terminal current cannot be combined with volume charge".into(),
                ));
            };
            add_boundary_current(load, value, projection, &mut source)?;
            continue;
        }
        let RealizedExcitationKind::ElectricChargeDensity(value) = load.kind else {
            return Err(ElectricalStaticError::Unsupported(
                "non-volume-charge realization".into(),
            ));
        };
        if value.imaginary != 0.0
            || !value.real.is_finite()
            || load.policy.distribution != ExcitationDistribution::IntensityPerVolume
            || load.policy.kinematics != ExcitationKinematics::FixedFrame
            || load.policy.frame != CoordinateFrame::Global
            || load.policy.projection.preserve_moment
        {
            return Err(ElectricalStaticError::Unsupported(
                "finite real scalar fixed volume charge required".into(),
            ));
        }
        let matches = |element: &&crate::topology::ProjectedElement| match &load.target {
            RealizedExcitationTarget::WholeModel => true,
            RealizedExcitationTarget::Region(target)
                if target.entity_dimension == EntityDimension::Volume =>
            {
                let original = raw[&element.stable_identity];
                target.matches_persistent_entity(&original.region_id)
                    || original
                        .provenance
                        .iter()
                        .any(|id| target.matches_persistent_entity(id))
            }
            _ => false,
        };
        let count = projection.elements.iter().filter(matches).count();
        if count == 0 {
            return Err(ElectricalStaticError::Boundary(format!(
                "charge source {} has no matched volume elements",
                load.excitation_id.0
            )));
        }
        let mut evidence = ScalarProjectionAccumulator::new(
            load,
            projection,
            SpatialIntegrationRule::VolumeKeastFifteen,
            count,
        )
        .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
        for element in projection.elements.iter().filter(matches) {
            projection
                .visit_element_quadrature(element, |point| {
                    let charge = value.real * point.volume_weight_m3;
                    source.integrated_source += charge;
                    source.absolute_source += charge.abs();
                    evidence.add_requested(charge, 0.0);
                    for (&node, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                        let nodal = charge * shape;
                        source.nodal_source[node] += nodal;
                        evidence.add_projected(node, nodal, 0.0);
                    }
                })
                .map_err(|e| ElectricalStaticError::Topology(e.to_string()))?;
        }
        let evidence = evidence.finish();
        evidence
            .validate()
            .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
        if !evidence.accepted {
            return Err(ElectricalStaticError::Operator(
                "charge projection conservation failed".into(),
            ));
        }
        source.projection_evidence.push(evidence);
    }
    if source.nodal_source.iter().any(|v| !v.is_finite())
        || !source.integrated_source.is_finite()
        || !source.absolute_source.is_finite()
    {
        return Err(ElectricalStaticError::Operator(
            "nonfinite integrated charge".into(),
        ));
    }
    Ok(Some(source))
}

/// Injects the authored total current through the selected exterior faces with a uniform
/// normal current density I/A, integrated consistently against the face shape functions.
fn add_boundary_current(
    load: &crate::excitation::RealizedExcitation,
    value: crate::excitation::ComplexScalar,
    projection: &AnalysisProjection,
    source: &mut ElectricalChargeSources,
) -> Result<(), ElectricalStaticError> {
    if value.imaginary != 0.0
        || !value.real.is_finite()
        || load.policy.distribution != ExcitationDistribution::PreserveResultant
        || load.policy.kinematics != ExcitationKinematics::FixedFrame
        || load.policy.projection.preserve_moment
    {
        return Err(ElectricalStaticError::Unsupported(
            "terminal current requires a finite real total current".into(),
        ));
    }
    let RealizedExcitationTarget::Region(target) = &load.target else {
        return Err(ElectricalStaticError::Boundary(
            "terminal current requires a face region".into(),
        ));
    };
    if target.entity_dimension != EntityDimension::Face {
        return Err(ElectricalStaticError::Boundary(
            "terminal current requires a face region".into(),
        ));
    }
    let faces = projection
        .boundary_faces_for_target(target)
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(ElectricalStaticError::Boundary(format!(
            "terminal current {} matches no boundary face",
            load.excitation_id.0
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut area = 0.0;
    for face in &faces {
        if !seen.insert(face.face_id)
            || face.role != runmat_meshing_core::BoundaryFaceRole::Exterior
        {
            return Err(ElectricalStaticError::Boundary(
                "terminal current requires unique exterior faces".into(),
            ));
        }
        projection
            .visit_boundary_quadrature(face, |point| area += point.area_weight_m2)
            .map_err(|e| ElectricalStaticError::Topology(e.to_string()))?;
    }
    if !area.is_finite() || area <= 0.0 {
        return Err(ElectricalStaticError::Boundary(
            "terminal current face area must be positive".into(),
        ));
    }
    let density = value.real / area;
    let mut evidence = ScalarProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::BoundaryDunavantSeven,
        faces.len(),
    )
    .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
    evidence.add_requested(value.real, 0.0);
    source.integrated_source += value.real;
    source.absolute_source += value.real.abs();
    for face in faces {
        projection
            .visit_boundary_quadrature(face, |point| {
                for (i, &node) in face.node_indices.iter().enumerate() {
                    let nodal = density * point.area_weight_m2 * point.shape_values()[i];
                    source.nodal_source[node] += nodal;
                    evidence.add_projected(node, nodal, 0.0);
                }
            })
            .map_err(|e| ElectricalStaticError::Topology(e.to_string()))?;
    }
    let evidence = evidence.finish();
    evidence
        .validate()
        .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
    if !evidence.accepted {
        return Err(ElectricalStaticError::Operator(
            "terminal current projection conservation failed".into(),
        ));
    }
    source.projection_evidence.push(evidence);
    Ok(())
}

/// A current density J on terminal faces injects the normal flux -J.n (n outward) at every
/// quadrature point; the injected current is its integral over the faces.
fn add_boundary_current_density(
    load: &crate::excitation::RealizedExcitation,
    value: crate::excitation::ComplexVector3,
    projection: &AnalysisProjection,
    source: &mut ElectricalChargeSources,
) -> Result<(), ElectricalStaticError> {
    if value.imaginary != [0.0; 3]
        || value.real.iter().any(|v| !v.is_finite())
        || load.policy.kinematics != ExcitationKinematics::FixedFrame
        || load.policy.frame != CoordinateFrame::Global
        || load.policy.projection.preserve_moment
    {
        return Err(ElectricalStaticError::Unsupported(
            "current density requires a finite real global-frame vector".into(),
        ));
    }
    let RealizedExcitationTarget::Region(target) = &load.target else {
        return Err(ElectricalStaticError::Boundary(
            "current density requires a face region".into(),
        ));
    };
    if target.entity_dimension != EntityDimension::Face {
        return Err(ElectricalStaticError::Boundary(
            "current density requires a face region".into(),
        ));
    }
    let faces = projection
        .boundary_faces_for_target(target)
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Err(ElectricalStaticError::Boundary(format!(
            "current density {} matches no boundary face",
            load.excitation_id.0
        )));
    }
    let mut evidence = ScalarProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::BoundaryDunavantSeven,
        faces.len(),
    )
    .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
    let mut injected = 0.0;
    for face in &faces {
        if face.role != runmat_meshing_core::BoundaryFaceRole::Exterior
            || !(face.area_m2.canonical_value() > 0.0)
        {
            return Err(ElectricalStaticError::Boundary(
                "current density requires exterior faces".into(),
            ));
        }
        let normal = face
            .oriented_area_vector_m2
            .map(|a| a / face.area_m2.canonical_value());
        let flux =
            -(value.real[0] * normal[0] + value.real[1] * normal[1] + value.real[2] * normal[2]);
        injected += flux * face.area_m2.canonical_value();
    }
    evidence.add_requested(injected, 0.0);
    source.integrated_source += injected;
    source.absolute_source += injected.abs();
    for face in faces {
        let normal = face
            .oriented_area_vector_m2
            .map(|a| a / face.area_m2.canonical_value());
        let flux =
            -(value.real[0] * normal[0] + value.real[1] * normal[1] + value.real[2] * normal[2]);
        projection
            .visit_boundary_quadrature(face, |point| {
                for (i, &node) in face.node_indices.iter().enumerate() {
                    let nodal = flux * point.area_weight_m2 * point.shape_values()[i];
                    source.nodal_source[node] += nodal;
                    evidence.add_projected(node, nodal, 0.0);
                }
            })
            .map_err(|e| ElectricalStaticError::Topology(e.to_string()))?;
    }
    let evidence = evidence.finish();
    evidence
        .validate()
        .map_err(|e| ElectricalStaticError::Operator(e.to_string()))?;
    if !evidence.accepted {
        return Err(ElectricalStaticError::Operator(
            "current density projection conservation failed".into(),
        ));
    }
    source.projection_evidence.push(evidence);
    Ok(())
}
