use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    BodyMassProperties, CurveEvaluatorId, ExactCurveImplementation, ExactEvaluatorRegistry,
    ExactMassPropertiesImplementation, ExactPcurveImplementation, ExactSurfaceImplementation,
    ExactTrimClassifierImplementation, GeometryEvaluationError, GeometryEvaluationErrorKind,
    MassPropertiesEvaluatorId, PcurveEvaluatorId, SurfaceEvaluatorId, TrimClassifierId,
};

use crate::exact::ImportedExactCad;

pub(super) struct EvaluatorBindings {
    pub curves: BTreeMap<CurveEvaluatorId, u64>,
    pub pcurves: BTreeMap<PcurveEvaluatorId, PcurveKey>,
    pub surfaces: BTreeMap<SurfaceEvaluatorId, u64>,
    pub trims: BTreeMap<TrimClassifierId, u64>,
    pub mass_properties: BTreeMap<MassPropertiesEvaluatorId, MassPropertiesBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PcurveKey {
    pub face: u64,
    pub wire: u64,
    pub position: u64,
    pub seam_image: i8,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum MassPropertiesBinding {
    Kernel {
        shape_keys: Vec<u64>,
        is_sheet_body: bool,
    },
    Validated(BodyMassProperties),
}

impl EvaluatorBindings {
    pub fn from_import(imported: &ImportedExactCad) -> Result<Self, GeometryEvaluationError> {
        Self::from_closure(
            &imported.representation,
            &imported.topology,
            &imported.evaluators,
            Some(&imported.kernel_body_shapes),
        )
    }

    pub fn from_closure(
        representation: &[u8],
        topology: &runmat_geometry_core::ExactBRepTopology,
        evaluators: &ExactEvaluatorRegistry,
        kernel_body_shapes: Option<&BTreeMap<MassPropertiesEvaluatorId, Vec<u64>>>,
    ) -> Result<Self, GeometryEvaluationError> {
        let representation_digest = crate::exact::exact_representation_digest(representation);
        let mut curves = BTreeMap::new();
        for record in &evaluators.curves {
            let ExactCurveImplementation::Kernel { reference } = &record.implementation else {
                return Err(inconsistent(
                    "an OCCT import cannot contain a portable curve evaluator",
                ));
            };
            require_digest(
                reference.representation_digest,
                representation_digest,
                "curve",
            )?;
            let shape_key = parse_edge_token(&reference.entity_token)?;
            if curves.insert(record.id.clone(), shape_key).is_some() {
                return Err(inconsistent("duplicate OCCT curve evaluator identity"));
            }
        }
        if curves.is_empty() {
            return Err(inconsistent(
                "OCCT exact geometry contains no curve evaluators",
            ));
        }
        require_inventory(
            "curve",
            topology
                .edges
                .iter()
                .map(|edge| &edge.curve_evaluator_id)
                .collect(),
            curves.keys().collect(),
        )?;

        let mut pcurves = BTreeMap::new();
        for record in &evaluators.pcurves {
            let ExactPcurveImplementation::Kernel { reference } = &record.implementation else {
                return Err(inconsistent(
                    "an OCCT import cannot contain a portable pcurve evaluator",
                ));
            };
            require_digest(
                reference.representation_digest,
                representation_digest,
                "pcurve",
            )?;
            if pcurves
                .insert(
                    record.id.clone(),
                    parse_pcurve_token(&reference.entity_token)?,
                )
                .is_some()
            {
                return Err(inconsistent("duplicate OCCT pcurve evaluator identity"));
            }
        }
        require_inventory(
            "pcurve",
            topology
                .coedges
                .iter()
                .map(|coedge| &coedge.pcurve_evaluator_id)
                .collect(),
            pcurves.keys().collect(),
        )?;

        let mut surfaces = BTreeMap::new();
        for record in &evaluators.surfaces {
            let ExactSurfaceImplementation::Kernel { reference } = &record.implementation else {
                return Err(inconsistent(
                    "an OCCT import cannot contain a portable surface evaluator",
                ));
            };
            require_digest(
                reference.representation_digest,
                representation_digest,
                "surface",
            )?;
            if surfaces
                .insert(
                    record.id.clone(),
                    parse_face_token(&reference.entity_token, "surface evaluator")?,
                )
                .is_some()
            {
                return Err(inconsistent("duplicate OCCT surface evaluator identity"));
            }
        }
        require_inventory(
            "surface",
            topology
                .faces
                .iter()
                .map(|face| &face.surface_evaluator_id)
                .collect(),
            surfaces.keys().collect(),
        )?;

        let mut trims = BTreeMap::new();
        for record in &evaluators.trim_classifiers {
            let ExactTrimClassifierImplementation::Kernel { reference } = &record.implementation
            else {
                return Err(inconsistent(
                    "an OCCT import cannot contain a portable trim classifier",
                ));
            };
            require_digest(
                reference.representation_digest,
                representation_digest,
                "trim classifier",
            )?;
            if trims
                .insert(
                    record.id.clone(),
                    parse_face_token(&reference.entity_token, "trim classifier")?,
                )
                .is_some()
            {
                return Err(inconsistent("duplicate OCCT trim classifier identity"));
            }
        }
        require_inventory(
            "trim classifier",
            topology
                .faces
                .iter()
                .map(|face| &face.trim_classifier_id)
                .collect(),
            trims.keys().collect(),
        )?;

        let mut mass_properties = BTreeMap::new();
        let bodies_by_evaluator = topology
            .bodies
            .iter()
            .map(|body| (&body.mass_properties_evaluator_id, body))
            .collect::<BTreeMap<_, _>>();
        for record in &evaluators.mass_properties {
            let body = bodies_by_evaluator
                .get(&record.id)
                .ok_or_else(|| inconsistent("OCCT mass-properties evaluator has no body"))?;
            let binding = match &record.implementation {
                ExactMassPropertiesImplementation::Kernel { reference } => {
                    require_digest(
                        reference.representation_digest,
                        representation_digest,
                        "mass-properties",
                    )?;
                    let expected_token = if body.is_sheet_body {
                        "body:sheet"
                    } else {
                        "body:solid"
                    };
                    if reference.entity_token != expected_token {
                        return Err(inconsistent(
                            "OCCT mass-properties evaluator has an invalid body token",
                        ));
                    }
                    MassPropertiesBinding::Kernel {
                        shape_keys: kernel_body_shapes
                            .and_then(|shapes| shapes.get(&record.id))
                            .filter(|keys| !keys.is_empty())
                            .cloned()
                            .ok_or_else(|| {
                                inconsistent("OCCT body has no kernel shape inventory")
                            })?,
                        is_sheet_body: body.is_sheet_body,
                    }
                }
                ExactMassPropertiesImplementation::KernelValidated {
                    properties,
                    validation_digest,
                } => {
                    if *validation_digest
                        != super::exact_projection::mass_validation_digest(
                            representation_digest,
                            properties,
                        )
                    {
                        return Err(inconsistent(
                            "OCCT mass-properties validation evidence does not match the representation",
                        ));
                    }
                    MassPropertiesBinding::Validated(*properties)
                }
            };
            if mass_properties.insert(record.id.clone(), binding).is_some() {
                return Err(inconsistent(
                    "duplicate OCCT mass-properties evaluator identity",
                ));
            }
        }
        require_inventory(
            "mass-properties",
            topology
                .bodies
                .iter()
                .map(|body| &body.mass_properties_evaluator_id)
                .collect(),
            mass_properties.keys().collect(),
        )?;
        Ok(Self {
            curves,
            pcurves,
            surfaces,
            trims,
            mass_properties,
        })
    }
}

fn require_digest(
    actual: [u8; 32],
    expected: [u8; 32],
    role: &str,
) -> Result<(), GeometryEvaluationError> {
    if actual != expected {
        return Err(inconsistent(format!(
            "{role} evaluator does not bind the supplied OCCT representation"
        )));
    }
    Ok(())
}

fn require_inventory<T: Ord>(
    role: &str,
    topology: BTreeSet<&T>,
    evaluators: BTreeSet<&T>,
) -> Result<(), GeometryEvaluationError> {
    if topology != evaluators {
        return Err(inconsistent(format!(
            "OCCT {role} evaluator inventory does not match exact topology"
        )));
    }
    Ok(())
}

fn parse_edge_token(token: &str) -> Result<u64, GeometryEvaluationError> {
    token
        .strip_prefix("edge:")
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|key| format!("edge:{key:020}") == token && *key != 0)
        .ok_or_else(|| inconsistent("OCCT curve evaluator has an invalid edge token"))
}

fn parse_pcurve_token(token: &str) -> Result<PcurveKey, GeometryEvaluationError> {
    let parts = token.split(':').collect::<Vec<_>>();
    if let ["face", face, "wire", wire, "coedge", position, "seam", seam_image] = parts.as_slice() {
        let parsed = PcurveKey {
            face: face.parse().unwrap_or(0),
            wire: wire.parse().unwrap_or(0),
            position: position.parse().unwrap_or(0),
            seam_image: seam_image.parse().unwrap_or(-2),
        };
        if parsed.face != 0
            && parsed.wire != 0
            && parsed.position != 0
            && (-1..=1).contains(&parsed.seam_image)
            && format!(
                "face:{:020}:wire:{:020}:coedge:{:020}:seam:{}",
                parsed.face, parsed.wire, parsed.position, parsed.seam_image
            ) == token
        {
            return Ok(parsed);
        }
    }
    Err(inconsistent(
        "OCCT pcurve evaluator has an invalid face-use token",
    ))
}

fn parse_face_token(token: &str, role: &str) -> Result<u64, GeometryEvaluationError> {
    token
        .strip_prefix("face:")
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|key| *key != 0 && format!("face:{key:020}") == token)
        .ok_or_else(|| inconsistent(format!("OCCT {role} has an invalid face token")))
}

fn inconsistent(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}
