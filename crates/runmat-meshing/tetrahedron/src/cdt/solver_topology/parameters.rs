use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{SolverNodeExactParameter, StableDigest};

use super::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput,
};

type CurveParameters = BTreeMap<PersistentEntityId, f64>;
type SurfaceParameters = BTreeMap<(PersistentEntityId, StableDigest), [f64; 2]>;

pub(super) fn build_node_exact_parameters(
    input: &DelaunaySolverTopologyInput<'_>,
    indices: &BTreeMap<StableDigest, u32>,
) -> Result<Vec<Vec<SolverNodeExactParameter>>, DelaunaySolverTopologyError> {
    let mut curves = vec![CurveParameters::new(); indices.len()];
    let mut surfaces = vec![SurfaceParameters::new(); indices.len()];

    for segment in &input.volume_mesh.provenance.segments {
        let edge = segment
            .entity_ids
            .first()
            .ok_or_else(|| invalid("exact segment has no source edge"))?;
        for (identity, parameter) in segment.node_identities.iter().zip(segment.edge_parameters) {
            let index = resolve_index(indices, identity)?;
            insert_curve(&mut curves[index], edge, parameter)?;
        }
    }

    for node in &input.exact_surface.nodes {
        let index = resolve_index(indices, &node.node_id)?;
        for use_record in &node.uses {
            insert_surface(
                &mut surfaces[index],
                &use_record.source_face_id,
                use_record.chart_id,
                use_record.evaluator_uv,
            )?;
            for parameter in &use_record.exact_edge_parameters {
                insert_curve(
                    &mut curves[index],
                    &parameter.source_edge_id,
                    parameter.parameter,
                )?;
            }
        }
    }

    Ok(curves
        .into_iter()
        .zip(surfaces)
        .map(|(curves, surfaces)| {
            curves
                .into_iter()
                .map(
                    |(source_edge_id, parameter)| SolverNodeExactParameter::Curve {
                        source_edge_id,
                        parameter,
                    },
                )
                .chain(
                    surfaces
                        .into_iter()
                        .map(|((source_face_id, chart_id), evaluator_uv)| {
                            SolverNodeExactParameter::Surface {
                                source_face_id,
                                chart_id,
                                evaluator_uv,
                            }
                        }),
                )
                .collect()
        })
        .collect())
}

fn resolve_index(
    indices: &BTreeMap<StableDigest, u32>,
    identity: &StableDigest,
) -> Result<usize, DelaunaySolverTopologyError> {
    indices
        .get(identity)
        .map(|index| *index as usize)
        .ok_or_else(|| invalid("exact parameter references a missing volume node"))
}

fn insert_curve(
    target: &mut CurveParameters,
    edge: &PersistentEntityId,
    parameter: f64,
) -> Result<(), DelaunaySolverTopologyError> {
    if target
        .insert(edge.clone(), parameter)
        .is_some_and(|existing| existing.to_bits() != parameter.to_bits())
    {
        return Err(invalid(
            "one solver node has conflicting parameters on an exact edge",
        ));
    }
    Ok(())
}

fn insert_surface(
    target: &mut SurfaceParameters,
    face: &PersistentEntityId,
    chart: StableDigest,
    evaluator_uv: [f64; 2],
) -> Result<(), DelaunaySolverTopologyError> {
    if target
        .insert((face.clone(), chart), evaluator_uv)
        .is_some_and(|existing| existing.map(f64::to_bits) != evaluator_uv.map(f64::to_bits))
    {
        return Err(invalid(
            "one solver node has conflicting parameters in an exact surface chart",
        ));
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
