use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, SolverNodeExactParameter, StableDigest};

use super::{
    checkpoint, error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

type CurveParameters = BTreeMap<PersistentEntityId, BTreeMap<u64, f64>>;
type SurfaceParameters = BTreeMap<(PersistentEntityId, StableDigest), BTreeMap<[u64; 2], [f64; 2]>>;

pub(super) fn build_node_exact_parameters(
    input: &DelaunaySolverTopologyInput<'_>,
    indices: &BTreeMap<StableDigest, u32>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<Vec<SolverNodeExactParameter>>, DelaunaySolverTopologyError> {
    let mut curves = vec![CurveParameters::new(); indices.len()];
    let mut surfaces = vec![SurfaceParameters::new(); indices.len()];

    for (work, segment) in input.volume_mesh.provenance.segments.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let edge = segment
            .entity_ids
            .first()
            .ok_or_else(|| invalid("exact segment has no source edge"))?;
        for (identity, parameter) in segment.node_identities.iter().zip(segment.edge_parameters) {
            let index = resolve_index(indices, identity)?;
            insert_curve(&mut curves[index], edge, parameter)?;
        }
    }

    for (work, node) in input.exact_surface.nodes.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        // A mixed exact model may contain independent sheet bodies in addition to solids. Those
        // sheets are valid analysis geometry, but their surface-only nodes are deliberately absent
        // after the volume mesh is carved. Parameters are projected only for nodes retained by the
        // volume topology; parameters on retained solid-boundary nodes remain mandatory below.
        let Some(index) = indices.get(&node.node_id).map(|index| *index as usize) else {
            continue;
        };
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

    let mut result = Vec::with_capacity(indices.len());
    for (work, (curves, surfaces)) in curves.into_iter().zip(surfaces).enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        result.push(
            curves
                .into_iter()
                .flat_map(|(source_edge_id, parameters)| {
                    parameters
                        .into_values()
                        .map(move |parameter| SolverNodeExactParameter::Curve {
                            source_edge_id: source_edge_id.clone(),
                            parameter,
                        })
                })
                .chain(
                    surfaces
                        .into_iter()
                        .flat_map(|((source_face_id, chart_id), parameters)| {
                            parameters.into_values().map(move |evaluator_uv| {
                                SolverNodeExactParameter::Surface {
                                    source_face_id: source_face_id.clone(),
                                    chart_id,
                                    evaluator_uv,
                                }
                            })
                        }),
                )
                .collect(),
        );
    }
    Ok(result)
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
    if !parameter.is_finite() {
        return Err(invalid("exact edge parameter is not finite"));
    }
    target
        .entry(edge.clone())
        .or_default()
        .entry(parameter.to_bits())
        .or_insert(parameter);
    Ok(())
}

fn insert_surface(
    target: &mut SurfaceParameters,
    face: &PersistentEntityId,
    chart: StableDigest,
    evaluator_uv: [f64; 2],
) -> Result<(), DelaunaySolverTopologyError> {
    if evaluator_uv.iter().any(|value| !value.is_finite()) {
        return Err(invalid("exact surface parameter is not finite"));
    }
    target
        .entry((face.clone(), chart))
        .or_default()
        .entry(evaluator_uv.map(f64::to_bits))
        .or_insert(evaluator_uv);
    Ok(())
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
