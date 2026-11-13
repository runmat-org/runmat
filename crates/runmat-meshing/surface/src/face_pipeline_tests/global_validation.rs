//! Actual accepted fixture mutations exercise the independent global consumer,
//! rather than comparing a replayed copy of the original partition decision.
use crate::*;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::NeverCancelled;

pub(super) fn verify(surface: &ExactSurfaceMesh, topology: &ExactBRepTopology) {
    use crate::surface_mesh::indexed_face::surface::SurfaceEncoding;
    for mutation in 0..7 {
        let mut changed = surface.clone();
        match mutation {
            0 => {
                let duplicate = changed.nodes[0].uses[0].clone();
                changed.nodes[0].uses.push(duplicate);
            }
            1 => changed.triangles.push(changed.triangles[0].clone()),
            2 => changed
                .boundary_segments
                .push(changed.boundary_segments[0].clone()),
            3 => changed.nodes.swap(0, 1),
            4 => changed.triangles[0].acceptance_sample_count = 0,
            5 => changed.triangles[0].accepted_chordal_deviation_m += 1.0,
            _ => changed.nodes[0].uses[0].chart_id = runmat_meshing_core::StableDigest::ZERO,
        }
        assert!(
            validate_published_exact_surface_mesh(
                &changed,
                topology,
                ExactSurfaceJoinOptions::default()
            )
            .is_err(),
            "global validator accepted mutation {mutation}"
        );
        let prepared = SurfaceEncoding::new(&changed, &NeverCancelled).unwrap();
        let profile = prepared.profile(&NeverCancelled).unwrap();
        let bytes = prepared.encode_profiled(&profile, &NeverCancelled).unwrap();
        assert!(
            decode_published_exact_surface_mesh(
                &bytes,
                topology,
                ExactSurfaceJoinOptions::default()
            )
            .is_err(),
            "cold global consumer accepted mutation {mutation}"
        );
    }
}
