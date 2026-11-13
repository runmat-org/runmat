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
    // A coedge's position and seam image are defined on the face use the topology traversed
    // (its orientation in the body), which can differ from the shape table's stored
    // orientation. The token names that use so the evaluator walks the same wire.
    let face_reversed = payload
        .faces
        .iter()
        .map(|face| (face.shape_key, face.reversed))
        .collect::<BTreeMap<_, _>>();
    let pcurves = payload
        .coedges
        .iter()
        .map(|coedge| {
            let reversed = face_reversed
                .get(&coedge.face_key)
                .copied()
                .ok_or_else(|| {
                    GeometryImportError::InvalidGeometry(
                        "OCCT exact coedge names a face absent from the topology".to_string(),
                    )
                })?;
            Ok(ExactPcurveEvaluatorRecord {
                id: names.pcurve_id(coedge.wire_key, coedge.coedge_key)?,
                implementation: ExactPcurveImplementation::Kernel {
                    reference: evaluator_ref(pcurve_token(
                        coedge.face_key,
                        reversed,
                        coedge.wire_key,
                        coedge.coedge_key,
                        coedge.seam_image,
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

/// Kernel reference of one coedge's pcurve: the face use (shape key and orientation), its wire,
/// the coedge's position in that use's traversal and its seam image (-1 off a seam).
pub(super) fn pcurve_token(
    face_key: u64,
    face_reversed: bool,
    wire_key: u64,
    position: u64,
    seam_image: i8,
) -> String {
    format!(
        "face:{face_key:020}:use:{}:wire:{wire_key:020}:coedge:{position:020}:seam:{seam_image}",
        if face_reversed { "reversed" } else { "forward" }
    )
}
