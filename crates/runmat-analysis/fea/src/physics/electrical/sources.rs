//! Typed volume charge, integrated with the shared scalar projection owner.
//! Cold validation may rederive the source, but never a potential/field or stiffness solve.
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalChargeSources {
    pub nodal_charge_c: Vec<f64>,
    pub integrated_charge_c: f64,
    /// Sum of absolute quadrature contributions, preserving cancellation scale.
    pub absolute_charge_c: f64,
    pub projection_evidence: Vec<SpatialProjectionEvidence>,
}

pub(crate) fn kind(model: &AnalysisModel) -> CanonicalElectricalSource {
    if model
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
) -> Result<Option<ElectricalChargeSources>, ElectricalStaticError> {
    if kind(model) == CanonicalElectricalSource::SourceFree {
        return Ok(None);
    }
    let loads = realize_step_excitations(model, step, ExcitationSamplePoint::Static)
        .map_err(|e| ElectricalStaticError::InvalidInput(e.to_string()))?;
    let raw = mesh
        .topology
        .volume_elements
        .iter()
        .map(|element| (element.stable_identity, element))
        .collect::<BTreeMap<_, _>>();
    let mut source = ElectricalChargeSources {
        nodal_charge_c: vec![0.0; projection.nodes.len()],
        integrated_charge_c: 0.0,
        absolute_charge_c: 0.0,
        projection_evidence: Vec::with_capacity(loads.len()),
    };
    for load in loads.iter() {
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
                    source.integrated_charge_c += charge;
                    source.absolute_charge_c += charge.abs();
                    evidence.add_requested(charge, 0.0);
                    for (&node, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                        let nodal = charge * shape;
                        source.nodal_charge_c[node] += nodal;
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
    if source.nodal_charge_c.iter().any(|v| !v.is_finite())
        || !source.integrated_charge_c.is_finite()
        || !source.absolute_charge_c.is_finite()
    {
        return Err(ElectricalStaticError::Operator(
            "nonfinite integrated charge".into(),
        ));
    }
    Ok(Some(source))
}
