use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;

use super::error::{refinement_error, sliver_error};
use super::{
    error, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshOptions,
    DelaunayVolumeMeshStage,
};
use crate::cdt::{
    volume_refinement::iteration::refine_delaunay_volume_until_stalled,
    volume_sliver::{run_treatment, QualityConvergenceMode},
    DelaunayVolumeNode, DelaunayVolumeProvenance, DelaunayVolumeQuality,
    DelaunayVolumeRefinementInput, DelaunayVolumeSliverMutation, DelaunayVolumeTopology,
};

pub(super) struct ConvergedVolumeQuality {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) provenance: DelaunayVolumeProvenance,
    pub(super) refinement_insertions: Vec<DelaunayVolumeNode>,
    pub(super) optimization_mutations: Vec<DelaunayVolumeSliverMutation>,
}

pub(super) fn converge_volume_quality(
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    metric_request: &MetricFieldRequest,
    provenance: DelaunayVolumeProvenance,
    fixed_node_identities: &[runmat_meshing_core::StableDigest],
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ConvergedVolumeQuality, DelaunayVolumeMeshError> {
    let mut topology = topology;
    let mut quality = quality;
    let mut refinement_insertions = Vec::new();
    let mut optimization_mutations = Vec::new();
    let mut remaining_refinement_insertions = options.refinement.maximum_insertions;
    let mut remaining_optimization_passes = options.sliver.maximum_passes;
    loop {
        let before_refinement = refinement_insertions.len();
        let mut refinement_stalled = false;
        if quality.worst_refinement_tetrahedron.is_some() {
            if remaining_refinement_insertions == 0 {
                return Err(error(
                    DelaunayVolumeMeshStage::Refinement,
                    DelaunayVolumeMeshErrorKind::ResourceLimit,
                    "coupled volume quality convergence exhausted its refinement insertion budget",
                ));
            }
            let mut refinement_options = options.refinement;
            refinement_options.maximum_insertions = remaining_refinement_insertions;
            let progress = refine_delaunay_volume_until_stalled(
                DelaunayVolumeRefinementInput {
                    topology: &topology,
                    metric_request,
                    provenance: &provenance,
                    quality: &quality,
                    quality_options: options.quality,
                    fixed_node_identities,
                },
                refinement_options,
                cancellation,
            )
            .map_err(refinement_error)?;
            refinement_stalled = progress.stalled;
            let refinement = progress.refinement;
            remaining_refinement_insertions =
                remaining_refinement_insertions.saturating_sub(refinement.insertions.len() as u64);
            refinement_insertions.extend(refinement.insertions);
            topology = refinement.topology;
            quality = refinement.quality;
        }

        let before_optimization = optimization_mutations.len();
        if refinement_stalled || quality.worst_optimization_tetrahedron.is_some() {
            if remaining_optimization_passes == 0 {
                return Err(error(
                    DelaunayVolumeMeshStage::Optimization,
                    DelaunayVolumeMeshErrorKind::ResourceLimit,
                    "coupled volume quality convergence exhausted its optimization pass budget",
                ));
            }
            let mut optimization_options = options.sliver;
            optimization_options.maximum_passes = remaining_optimization_passes;
            let mode = if refinement_stalled {
                QualityConvergenceMode::StalledRefinement
            } else {
                QualityConvergenceMode::Coupled
            };
            let optimized = run_treatment(
                DelaunayVolumeRefinementInput {
                    topology: &topology,
                    metric_request,
                    provenance: &provenance,
                    quality: &quality,
                    quality_options: options.quality,
                    fixed_node_identities,
                },
                optimization_options,
                cancellation,
                mode,
            );
            let optimized = match optimized {
                Ok(optimized) => optimized,
                Err(failure) => return Err(sliver_error(failure)),
            };
            remaining_optimization_passes =
                remaining_optimization_passes.saturating_sub(optimized.mutations.len() as u64);
            optimization_mutations.extend(optimized.mutations);
            topology = optimized.topology;
            quality = optimized.quality;
        }

        if quality.worst_refinement_tetrahedron.is_none()
            && quality.worst_optimization_tetrahedron.is_none()
        {
            return Ok(ConvergedVolumeQuality {
                topology,
                quality,
                provenance,
                refinement_insertions,
                optimization_mutations,
            });
        }
        if refinement_insertions.len() == before_refinement
            && optimization_mutations.len() == before_optimization
        {
            return Err(error(
                DelaunayVolumeMeshStage::Optimization,
                DelaunayVolumeMeshErrorKind::InvalidQuality,
                "coupled volume refinement and optimization made no progress",
            ));
        }
    }
}
