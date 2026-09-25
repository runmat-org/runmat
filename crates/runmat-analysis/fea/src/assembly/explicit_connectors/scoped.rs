use runmat_analysis_core::AnalysisConnector;

use super::{
    canonical_node, invalid, provenance, PointConnectorOperator, PreparedPointConnector,
    SurfaceFoundationSample,
};
use crate::{
    algebra::RealSparseAssembler,
    assembly::{dofs::StructuralDofKind, dofs::StructuralDofLayout, StructuralAssemblyError},
    connectors::CanonicalConnectorEntry,
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError},
};

type SparseEntry = (usize, usize, f64);

// A Tet10 boundary face has six nodes, hence at most 18 translational terms
// and 324 sparse entries per quadrature sample. Four samples retain under
// 32 KiB of entries per worker, below the admitted 128 KiB assembly scratch.
const MAX_SAMPLE_NODES: usize = 6;
const SAMPLES_PER_WORKER_WAVE: usize = 4;
// An axial connector contributes at most 6x6 entries. Eight point connectors
// and their transient chunk-to-merge copy fit the admitted 128 KiB fixed
// assembly scratch per worker along with canonical-entry metadata.
const POINT_CONNECTORS_PER_WORKER_WAVE: usize = 8;

pub(super) struct PointConnectorSink<'a> {
    pub stiffness: &'a mut RealSparseAssembler,
    pub damping: &'a mut RealSparseAssembler,
    pub stiffness_count: &'a mut usize,
    pub damping_count: &'a mut usize,
}

pub(super) fn point_wave_capacity(
    plan: LocalExecutionPlan,
) -> Result<usize, StructuralAssemblyError> {
    usize::from(plan.thread_count().get())
        .checked_mul(POINT_CONNECTORS_PER_WORKER_WAVE)
        .ok_or_else(|| {
            StructuralAssemblyError::LocalExecution("point connector wave size overflow".into())
        })
}

pub(super) fn flush_point_wave(
    wave: &mut Vec<PreparedPointConnector<'_>>,
    layout: &StructuralDofLayout,
    plan: LocalExecutionPlan,
    sink: PointConnectorSink<'_>,
) -> Result<(), StructuralAssemblyError> {
    if wave.is_empty() {
        return Ok(());
    }
    let cancellation = crate::progress::cancellation_token();
    if cancellation.is_cancelled() {
        return Err(StructuralAssemblyError::Cancelled);
    }
    let contributions = if plan.is_serial() {
        wave.iter()
            .map(|point| point_entries(layout, point))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        execute_scoped_chunks(plan, wave.len() as u64, |chunk| {
            let mut entries = Vec::with_capacity(chunk.len() as usize);
            for point in &wave[chunk.start() as usize..chunk.end_exclusive() as usize] {
                if cancellation.is_cancelled() {
                    return Err(StructuralAssemblyError::Cancelled);
                }
                entries.push(point_entries(layout, point)?);
            }
            Ok(entries)
        })
        .map_err(|error| match error {
            ScopedLocalExecutionError::Work { source, .. } => source,
            other => StructuralAssemblyError::LocalExecution(other.to_string()),
        })?
        .into_iter()
        .flatten()
        .collect()
    };
    if cancellation.is_cancelled() {
        return Err(StructuralAssemblyError::Cancelled);
    }
    for (point, entries) in wave.drain(..).zip(contributions) {
        let (assembler, count) = match point.operator {
            PointConnectorOperator::AxialDashpot { .. } => {
                (&mut *sink.damping, &mut *sink.damping_count)
            }
            _ => (&mut *sink.stiffness, &mut *sink.stiffness_count),
        };
        for (row, column, value) in entries {
            assembler.add(row, column, value);
        }
        assembler.add_contribution(provenance(point.connector));
        *count += 1;
    }
    Ok(())
}

pub(super) fn canonical_point_entry(
    point: &PreparedPointConnector<'_>,
    node_identities: &[runmat_meshing_core::StableDigest],
) -> Result<CanonicalConnectorEntry, StructuralAssemblyError> {
    let connector = point.connector;
    Ok(match point.operator {
        PointConnectorOperator::AxialSpring {
            endpoints,
            axis,
            coefficient,
        } => CanonicalConnectorEntry::AxialSpring {
            connector_id: connector.connector_id.clone(),
            source_digest: point.source_digest,
            primary: canonical_node(endpoints[0], node_identities, connector)?,
            secondary: canonical_node(endpoints[1], node_identities, connector)?,
            unit_axis: axis,
            stiffness_newtons_per_meter: coefficient,
        },
        PointConnectorOperator::AxialDashpot {
            endpoints,
            axis,
            coefficient,
        } => CanonicalConnectorEntry::AxialDashpot {
            connector_id: connector.connector_id.clone(),
            source_digest: point.source_digest,
            primary: canonical_node(endpoints[0], node_identities, connector)?,
            secondary: canonical_node(endpoints[1], node_identities, connector)?,
            unit_axis: axis,
            damping_newton_seconds_per_meter: coefficient,
        },
        PointConnectorOperator::GroundSpring { node, coefficient } => {
            CanonicalConnectorEntry::IsotropicGroundSpring {
                connector_id: connector.connector_id.clone(),
                source_digest: point.source_digest,
                target: canonical_node(node, node_identities, connector)?,
                stiffness_newtons_per_meter: coefficient,
            }
        }
    })
}

fn point_entries(
    layout: &StructuralDofLayout,
    point: &PreparedPointConnector<'_>,
) -> Result<Vec<SparseEntry>, StructuralAssemblyError> {
    match point.operator {
        PointConnectorOperator::AxialSpring {
            endpoints,
            axis,
            coefficient,
        }
        | PointConnectorOperator::AxialDashpot {
            endpoints,
            axis,
            coefficient,
        } => {
            let mut terms = Vec::with_capacity(6);
            for (node, sign) in [(endpoints[0], -1.0), (endpoints[1], 1.0)] {
                for (dof, component) in [
                    (StructuralDofKind::Ux, axis[0]),
                    (StructuralDofKind::Uy, axis[1]),
                    (StructuralDofKind::Uz, axis[2]),
                ] {
                    let row = layout.index(node, dof).ok_or_else(|| {
                        invalid(
                            point.connector,
                            "axial endpoint lacks a translational degree of freedom",
                        )
                    })?;
                    terms.push((row, sign * component));
                }
            }
            let mut entries = Vec::with_capacity(36);
            for &(row, row_weight) in &terms {
                for &(column, column_weight) in &terms {
                    entries.push((row, column, coefficient * row_weight * column_weight));
                }
            }
            Ok(entries)
        }
        PointConnectorOperator::GroundSpring { node, coefficient } => [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .map(|dof| {
            let row = layout.index(node, dof).ok_or_else(|| {
                invalid(
                    point.connector,
                    "ground spring node lacks a translational degree of freedom",
                )
            })?;
            Ok((row, row, coefficient))
        })
        .collect(),
    }
}

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
