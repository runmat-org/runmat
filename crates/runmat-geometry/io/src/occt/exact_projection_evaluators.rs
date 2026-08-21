use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactCurveEvaluatorRecord, ExactCurveImplementation, ExactEvaluatorRegistry,
    ExactMassPropertiesRecord, ExactPcurveEvaluatorRecord, ExactPcurveImplementation,
    ExactSurfaceEvaluatorRecord, ExactSurfaceImplementation, ExactTrimClassifierImplementation,
    ExactTrimClassifierRecord, EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
};

use super::{exact_persistent_names::PersistentNameIndex, ffi::bridge};
use crate::import::GeometryImportError;

pub(super) fn project_evaluators(
    payload: &bridge::OcctExactShapePayload,
    names: &PersistentNameIndex,
    representation_digest: [u8; 32],
    mass_properties: Vec<ExactMassPropertiesRecord>,
) -> Result<ExactEvaluatorRegistry, GeometryImportError> {
    let evaluator_ref = |entity_token: String| runmat_geometry_core::KernelEvaluatorRef {
        entity_token,
        representation_digest,
    };
    let curves = payload
        .edges
        .iter()
        .map(|edge| {
            Ok(ExactCurveEvaluatorRecord {
                id: names.curve_id(edge.shape_key)?,
                implementation: ExactCurveImplementation::Kernel {
                    reference: evaluator_ref(format!("edge:{:020}", edge.shape_key)),
                },
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let pcurves = payload
        .coedges
        .iter()
        .map(|coedge| {
            Ok(ExactPcurveEvaluatorRecord {
                id: names.pcurve_id(coedge.wire_key, coedge.coedge_key)?,
                implementation: ExactPcurveImplementation::Kernel {
                    reference: evaluator_ref(format!(
                        "face:{:020}:wire:{:020}:coedge:{:020}:seam:{}",
                        coedge.face_key, coedge.wire_key, coedge.coedge_key, coedge.seam_image
                    )),
                },
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let surfaces = payload
        .faces
        .iter()
        .map(|face| {
            Ok(ExactSurfaceEvaluatorRecord {
                id: names.surface_id(face.shape_key)?,
                implementation: ExactSurfaceImplementation::Kernel {
                    reference: evaluator_ref(format!("face:{:020}", face.shape_key)),
                },
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let trim_classifiers = payload
        .faces
        .iter()
        .map(|face| {
            Ok(ExactTrimClassifierRecord {
                id: names.trim_id(face.shape_key)?,
                implementation: ExactTrimClassifierImplementation::Kernel {
                    reference: evaluator_ref(format!("face:{:020}", face.shape_key)),
                },
            })
        })
        .collect::<Result<Vec<_>, GeometryImportError>>()?;
    let mut registry = ExactEvaluatorRegistry {
        schema_version: EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
        kernel_abi: payload.kernel_abi.clone(),
        curves: canonical_by_id(curves, |record| record.id.as_str()),
        pcurves: canonical_by_id(pcurves, |record| record.id.as_str()),
        surfaces: canonical_by_id(surfaces, |record| record.id.as_str()),
        trim_classifiers: canonical_by_id(trim_classifiers, |record| record.id.as_str()),
        mass_properties,
    };
    registry
        .mass_properties
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(registry)
}

fn canonical_by_id<T, F>(records: Vec<T>, id: F) -> Vec<T>
where
    F: Fn(&T) -> &str,
{
    records
        .into_iter()
        .map(|record| (id(&record).to_owned(), record))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect()
}
