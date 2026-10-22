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
    if model
        .excitations
        .iter()
        .any(|action| matches!(action, ExcitationAction::ElectricCurrent(_)))
    {
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
