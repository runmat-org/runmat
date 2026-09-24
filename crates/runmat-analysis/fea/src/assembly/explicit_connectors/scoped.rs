use runmat_analysis_core::AnalysisConnector;

use super::{invalid, SurfaceFoundationSample};
use crate::{
    algebra::RealSparseAssembler,
    assembly::{dofs::StructuralDofKind, dofs::StructuralDofLayout, StructuralAssemblyError},
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError},
};

type SparseEntry = (usize, usize, f64);

// A Tet10 boundary face has six nodes, hence at most 18 translational terms
// and 324 sparse entries per quadrature sample. Four samples retain under
// 32 KiB of entries per worker, below the admitted 128 KiB assembly scratch.
const MAX_SAMPLE_NODES: usize = 6;
const SAMPLES_PER_WORKER_WAVE: usize = 4;

pub(super) fn add_surface_normal_foundation_scoped(
    connector: &AnalysisConnector,
    assembler: &mut RealSparseAssembler,
    layout: &StructuralDofLayout,
    samples: &[SurfaceFoundationSample],
    modulus: f64,
    plan: LocalExecutionPlan,
) -> Result<(), StructuralAssemblyError> {
    validate_surface_samples(connector, samples)?;
    let wave_capacity = usize::from(plan.thread_count().get())
        .checked_mul(SAMPLES_PER_WORKER_WAVE)
        .ok_or_else(|| {
            StructuralAssemblyError::LocalExecution("connector wave size overflow".into())
        })?;
    let cancellation = crate::progress::cancellation_token();
    for wave in samples.chunks(wave_capacity) {
        if cancellation.is_cancelled() {
            return Err(StructuralAssemblyError::Cancelled);
        }
        let contributions = execute_scoped_chunks(plan, wave.len() as u64, |chunk| {
            let mut contributions = Vec::with_capacity(chunk.len() as usize);
            for sample in &wave[chunk.start() as usize..chunk.end_exclusive() as usize] {
                if cancellation.is_cancelled() {
                    return Err(StructuralAssemblyError::Cancelled);
                }
                contributions.push(sample_entries(layout, sample, modulus)?);
            }
            Ok(contributions)
        })
        .map_err(|error| match error {
            ScopedLocalExecutionError::Work { source, .. } => source,
            other => StructuralAssemblyError::LocalExecution(other.to_string()),
        })?;
        // Sparse accumulation and its floating-point order stay authored and
        // sample-major, independent of worker completion or chunk partition.
        for entries in contributions.into_iter().flatten() {
            for (row, column, value) in entries {
                assembler.add(row, column, value);
            }
        }
    }
    Ok(())
}

pub(super) fn validate_surface_samples(
    connector: &AnalysisConnector,
    samples: &[SurfaceFoundationSample],
) -> Result<(), StructuralAssemblyError> {
    for sample in samples {
        if sample.node_indices.len() > MAX_SAMPLE_NODES
            || sample.node_indices.len() != sample.shape_values.len()
        {
            return Err(invalid(
                connector,
                "surface foundation quadrature exceeds the admitted six-node face",
            ));
        }
    }
    Ok(())
}

fn sample_entries(
    layout: &StructuralDofLayout,
    sample: &SurfaceFoundationSample,
    modulus: f64,
) -> Result<Vec<SparseEntry>, StructuralAssemblyError> {
    let mut terms = Vec::with_capacity(sample.node_indices.len() * 3);
    for (&node, &shape) in sample.node_indices.iter().zip(&sample.shape_values) {
        for (dof, normal) in [
            (StructuralDofKind::Ux, sample.unit_normal[0]),
            (StructuralDofKind::Uy, sample.unit_normal[1]),
            (StructuralDofKind::Uz, sample.unit_normal[2]),
        ] {
            let row = layout.index(node, dof).ok_or_else(|| {
                StructuralAssemblyError::LocalExecution(
                    "surface foundation node lacks a translational degree of freedom".into(),
                )
            })?;
            terms.push((row, shape * normal));
        }
    }
    let scale = modulus * sample.area_weight_m2;
    let mut entries = Vec::with_capacity(terms.len() * terms.len());
    for &(row, row_weight) in &terms {
        for &(column, column_weight) in &terms {
            entries.push((row, column, scale * row_weight * column_weight));
        }
    }
    Ok(entries)
}
