//! Global metrics need one derived cell context at a time. Scoped grading
//! retains the complete graph-bound inventory. Neither variant is serialized.
use super::*;
use crate::cdt::volume_quality::{
    provenance_error, DelaunayVolumeProvenance, DelaunayVolumeQualityOptions, MetricFieldRequest,
};
use runmat_meshing_core::MeshingCancellationSignal;

pub(in crate::cdt::volume_quality) trait MetricContextInventory {
    fn len(&self) -> usize;
    fn indexed(&self) -> Option<&[DelaunayVolumeMetricContext]>;
}
impl MetricContextInventory for Vec<DelaunayVolumeMetricContext> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn indexed(&self) -> Option<&[DelaunayVolumeMetricContext]> {
        Some(self)
    }
}
pub(in crate::cdt::volume_quality) enum MetricContexts<'a> {
    Derived {
        topology: &'a DelaunayVolumeTopology,
        provenance: &'a DelaunayVolumeProvenance,
    },
    Indexed(Vec<DelaunayVolumeMetricContext>),
}
impl<'a> MetricContexts<'a> {
    pub(in crate::cdt::volume_quality) fn from_admitted(
        topology: &'a DelaunayVolumeTopology,
        provenance: &'a DelaunayVolumeProvenance,
        request: &MetricFieldRequest,
        options: DelaunayVolumeQualityOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayVolumeQualityError> {
        if request.contributions.is_empty() {
            return Ok(Self::Derived {
                topology,
                provenance,
            });
        }
        let contexts = crate::cdt::volume_provenance::derive_delaunay_volume_metric_contexts_from_admitted_input(topology, provenance, options.provenance, cancellation).map_err(provenance_error)?;
        validate_metric_contexts(topology, &contexts)?;
        Ok(Self::Indexed(contexts))
    }
}
impl MetricContextInventory for MetricContexts<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Derived { topology, .. } => topology.tetrahedra.len(),
            Self::Indexed(contexts) => contexts.len(),
        }
    }
    fn indexed(&self) -> Option<&[DelaunayVolumeMetricContext]> {
        match self {
            Self::Derived { .. } => None,
            Self::Indexed(contexts) => Some(contexts),
        }
    }
}
pub(in crate::cdt::volume_quality) enum MetricContextIterator<'a> {
    Derived {
        topology: &'a DelaunayVolumeTopology,
        provenance: &'a DelaunayVolumeProvenance,
        index: usize,
    },
    Indexed(std::vec::IntoIter<DelaunayVolumeMetricContext>),
}
impl<'a> IntoIterator for MetricContexts<'a> {
    type Item = Result<DelaunayVolumeMetricContext, DelaunayVolumeQualityError>;
    type IntoIter = MetricContextIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Derived {
                topology,
                provenance,
            } => MetricContextIterator::Derived {
                topology,
                provenance,
                index: 0,
            },
            Self::Indexed(contexts) => MetricContextIterator::Indexed(contexts.into_iter()),
        }
    }
}
impl Iterator for MetricContextIterator<'_> {
    type Item = Result<DelaunayVolumeMetricContext, DelaunayVolumeQualityError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed(contexts) => contexts.next().map(Ok),
            Self::Derived {
                topology,
                provenance,
                index,
            } => {
                let cell = topology.tetrahedra.get(*index)?;
                *index += 1;
                Some(
                    crate::cdt::volume_provenance::derive_metric_context_for_cell(
                        cell, topology, provenance,
                    )
                    .map_err(provenance_error),
                )
            }
        }
    }
}
