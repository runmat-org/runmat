use super::super::test_fixtures::{assigned, pair, quality};
use super::*;
use crate::cdt::volume_sliver::admission::{quality_spectrum, QualityObjective};

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Independent full physical quality evaluations of a three-to-two bipyramid and zero or
//   64 unrelated cells reconstruct exactly from three removed and two changed quality records.
//   Every per-cell metric, provenance field, violation ratio and aggregate extreme is preserved;
//   both priority spectra match the original full inventories. Changed metadata on unchanged
//   connectivity is retained, unchanged seeds contain no changed records, and noncanonical cell
//   order rejects. This storage witness does not establish sublinear encoding work, arbitrary-mesh
//   mutation locality, native memory or quality convergence.
#[test]
fn compact_quality_seed_preserves_every_metric_and_rank_without_unrelated_copies() {
    for unrelated in [0, 64] {
        let (base, candidate) = pair(unrelated);
        let base = quality(&assigned(base));
        let candidate = quality(&assigned(candidate));
        let patch = QualityPatch::between(&base, &candidate).unwrap();
        assert_eq!(patch.removed.len(), 3);
        assert_eq!(patch.changed.len(), 2);
        // Full independent physical evaluations supply both inventories. Equality
        // covers metrics, provenance, violation ratios and aggregate extrema.
        assert_eq!(patch.materialize(&base), candidate);
        for objective in [QualityObjective::Shape, QualityObjective::Refinement] {
            assert_eq!(
                patch.spectrum(&base, objective),
                quality_spectrum(&candidate, objective)
            );
        }
        let unchanged = QualityPatch::between(&base, &base).unwrap();
        assert!(unchanged.removed.is_empty());
        assert!(unchanged.changed.is_empty());
        assert_eq!(unchanged.materialize(&base), base);
        // A changed metric on unchanged connectivity must never reuse its old
        // evidence merely because the corner identities are the same.
        let mut changed_metric = base.clone();
        changed_metric.tetrahedra[0].optimization_violation_ratio = 7.;
        let changed = QualityPatch::between(&base, &changed_metric).unwrap();
        assert_eq!(changed.removed.len(), 1);
        assert_eq!(changed.changed.len(), 1);
        assert_eq!(changed.materialize(&base), changed_metric);
        assert_eq!(
            changed.spectrum(&base, QualityObjective::Shape),
            quality_spectrum(&changed_metric, QualityObjective::Shape)
        );
        let mut noncanonical = base.clone();
        noncanonical.tetrahedra.reverse();
        assert_eq!(
            QualityPatch::between(&base, &noncanonical)
                .err()
                .unwrap()
                .kind,
            DelaunayVolumeSliverErrorKind::InvalidQuality
        );
    }
}
