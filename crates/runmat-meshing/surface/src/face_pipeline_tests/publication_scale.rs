//! Representation/resource fixture, deliberately separate from CAD/PDE acceptance.
//! A regular planar inventory reproduces the dense face's node/triangle scale
//! without repeating meshing. It does not claim original boundary provenance.
use crate::*;
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{NeverCancelled, StableDigest};
use sha2::{Digest, Sha256};

fn representative_face(cells: usize) -> ExactFaceMesh {
    let face = PersistentEntityId::new(
        PersistentEntityKind::Face,
        format!("occt:{}", "1".repeat(64)),
        vec!["root".into(), "0:1:1:1".into()],
    );
    let chart = StableDigest::from_bytes(Sha256::digest(b"surface-resource-grid/chart").into());
    let stride = cells + 1;
    let mut nodes = Vec::with_capacity(stride * stride);
    for y in 0..=cells {
        for x in 0..=cells {
            let uv = [x as f64 / cells as f64, y as f64 / cells as f64];
            nodes.push(ExactFaceMeshNode {
                node_id: exact_face_interior_node_id(&face, uv),
                point_m: [uv[0], uv[1], 0.0],
                uses: vec![ExactFaceMeshNodeUse {
                    source_face_id: face.clone(),
                    chart_id: chart,
                    uv,
                    evaluator_uv: uv,
                    exact_edge_parameters: vec![],
                }],
            });
        }
    }
    let mut triangles = Vec::with_capacity(2 * cells * cells);
    let h = 1.0 / cells as f64;
    for y in 0..cells {
        for x in 0..cells {
            let a = y * stride + x;
            for indices in [[a, a + 1, a + stride], [a + 1, a + stride + 1, a + stride]] {
                let node_ids = indices.map(|index| nodes[index].node_id);
                let (node_ids, rotation) = crate::face_mesh::canonical_triangle(node_ids);
                let edge_lengths = [0, 1, 2].map(|edge| {
                    let a = nodes[indices[(edge + rotation) % 3]].point_m;
                    let b = nodes[indices[(edge + rotation + 1) % 3]].point_m;
                    (a[0] - b[0]).hypot(a[1] - b[1])
                });
                triangles.push(ExactFaceMeshTriangle {
                    triangle_id: crate::face_mesh::exact_face_triangle_id(chart, node_ids),
                    chart_id: chart,
                    source_face_id: face.clone(),
                    node_ids,
                    unit_normal: [0.0, 0.0, 1.0],
                    physical_area_m2: h * h * 0.5,
                    metric_edge_lengths: edge_lengths,
                    minimum_metric_angle_rad: std::f64::consts::FRAC_PI_4,
                    physical_aspect_ratio: (1.0 + 2.0_f64.sqrt()) / 3.0_f64.sqrt(),
                    chordal_deviation_m: 0.0,
                    normal_deviation_rad: 0.0,
                    acceptance_sample_count: 84,
                    accepted_chordal_deviation_m: 0.0,
                    accepted_normal_deviation_rad: 0.0,
                });
            }
        }
    }
    nodes.sort_by_key(|node| node.node_id);
    triangles.sort_by_key(|triangle| triangle.triangle_id);
    ExactFaceMesh {
        schema_version: EXACT_FACE_MESH_SCHEMA_VERSION,
        source_face_id: face,
        nodes,
        triangles,
        boundary_segments: vec![],
        joined_chart_cuts: vec![],
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.0),
        maximum_normal_deviation_rad: 0.0,
    }
}

// witness: FEA-MESH-DISCRETIZATION-001/surface_discretization @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A dense face of 90,601 nodes and 180,000 triangles has a bounded publication
//   representation inside the 512 MiB default component memory and round-trips through the record
//   and global codecs; seams, poles and many trims are not asserted here.
#[test]
#[ignore = "representative dense surface publication/resource gate"]
fn representative_dense_face_publication_fits_default_component_memory() {
    let face = representative_face(300);
    assert_eq!(face.nodes.len(), 90_601);
    assert_eq!(face.triangles.len(), 180_000);
    let profile =
        crate::surface_mesh::profile_accepted_face(&face, 256 * 1024 * 1024, &NeverCancelled)
            .unwrap();
    eprintln!(
        "DENSE_SURFACE_PUBLICATION_PROFILE nodes={} triangles={} profile={profile:?}",
        face.nodes.len(),
        face.triangles.len()
    );
    let profile = profile.expect("dense face must have a bounded production representation");
    // Typed input is separate from encoder workspace. Real CAD provenance and
    // complete workflow checks remain mandatory after this representation gate.
    let typed = face.retained_storage_bytes(&super::Control).unwrap();
    eprintln!(
        "DENSE_SURFACE_PUBLICATION_STORAGE typed={typed} wire={} encoder={}",
        profile.encoded_bytes_upper_bound, profile.resident_workspace_bytes_estimate
    );
    assert!(
        typed + profile.resident_workspace_bytes_estimate <= 512 * 1024 * 1024,
        "dense face publication exceeds unchanged default component memory"
    );
    dense_round_trip(&face, typed);
    dense_record_batch(&face, typed);
    dense_global_round_trip(face);
}

struct MemoryControl {
    live: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
    limit: u64,
}
impl runmat_geometry_core::GeometryEvaluationControl for MemoryControl {
    fn checkpoint(&self) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(
        &self,
        _: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(
        &self,
        _: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(
        &self,
        bytes: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        use std::sync::atomic::Ordering::Relaxed;
        let previous = self
            .live
            .fetch_update(Relaxed, Relaxed, |live| {
                live.checked_add(bytes).filter(|sum| *sum <= self.limit)
            })
            .map_err(|_| {
                runmat_geometry_core::GeometryEvaluationError::new(
                    runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "dense publication exceeds its original component ceiling",
                )
            })?;
        self.peak.fetch_max(previous + bytes, Relaxed);
        Ok(())
    }
    fn release_allocation_bytes(&self, bytes: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.live
            .fetch_update(Relaxed, Relaxed, |live| live.checked_sub(bytes))
            .expect("balanced owned publication reservations");
    }
}

fn dense_record_batch(face: &ExactFaceMesh, typed: u64) {
    use runmat_geometry_core::{
        GeometryAllocationReservation, GeometryAllocationScope, GeometryEvaluationControl,
    };
    let memory = MemoryControl {
        live: 0.into(),
        peak: 0.into(),
        limit: 512 * 1024 * 1024,
    };
    {
        let scope = GeometryAllocationScope::new(&memory);
        scope
            .consume_allocation_bytes(typed + (8 * std::mem::size_of::<Vec<u8>>()) as u64)
            .unwrap();
        let mut records = Vec::with_capacity(8);
        for _ in 0..8 {
            let _preparation = GeometryAllocationReservation::new(
                &scope,
                ExactFacePartitionEncoding::single_preparation_storage_bytes(face, &NeverCancelled)
                    .unwrap(),
            )
            .unwrap();
            let prepared = ExactFacePartitionEncoding::single(face, &NeverCancelled).unwrap();
            let profile = prepared.encoding_profile(&NeverCancelled).unwrap();
            let mut encoding = GeometryAllocationReservation::new(
                &scope,
                profile.resident_workspace_bytes_estimate,
            )
            .unwrap();
            let bytes = prepared.encode_profiled(&profile, &NeverCancelled).unwrap();
            assert!(bytes.capacity() as u64 <= profile.encoded_bytes_upper_bound);
            encoding.retain_bytes(bytes.capacity() as u64);
            records.push(bytes);
        }
        assert_eq!(records.len(), 8);
        eprintln!(
            "DENSE_SURFACE_EIGHT_RECORD_PRODUCER wire={} retained_capacity={} admitted_peak={}",
            records.iter().map(Vec::len).sum::<usize>(),
            records.iter().map(Vec::capacity).sum::<usize>(),
            memory.peak.load(std::sync::atomic::Ordering::Relaxed)
        );
        // Repeated representation only: these are not eight independently
        // accepted CAD faces, and no joined/PDE qualification is claimed.
    }
    assert_eq!(memory.live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

fn dense_round_trip(face: &ExactFaceMesh, typed: u64) {
    use crate::surface_mesh::indexed_face::partition;
    use runmat_canonical_codec::{CanonicalCodecError, CanonicalCodecErrorKind, CanonicalLimits};
    use runmat_geometry_core::{
        GeometryAllocationReservation, GeometryAllocationScope, GeometryEvaluationControl,
    };
    let memory = MemoryControl {
        live: 0.into(),
        peak: 0.into(),
        limit: 512 * 1024 * 1024,
    };
    {
        let scope = GeometryAllocationScope::new(&memory);
        scope.consume_allocation_bytes(typed).unwrap();
        let preparation = GeometryAllocationReservation::new(
            &scope,
            ExactFacePartitionEncoding::single_preparation_storage_bytes(face, &NeverCancelled)
                .unwrap(),
        )
        .unwrap();
        let prepared = ExactFacePartitionEncoding::single(face, &NeverCancelled).unwrap();
        let profile = prepared.encoding_profile(&NeverCancelled).unwrap();
        let mut encoding =
            GeometryAllocationReservation::new(&scope, profile.resident_workspace_bytes_estimate)
                .unwrap();
        let bytes = prepared.encode_profiled(&profile, &NeverCancelled).unwrap();
        assert!(bytes.len() as u64 <= profile.encoded_bytes_upper_bound);
        encoding.retain_bytes(bytes.capacity() as u64);
        drop(encoding);
        drop(prepared);
        drop(preparation);
        let requested =
            ExactFacePartitionEncoding::decoding_storage_bytes(&bytes, &NeverCancelled).unwrap();
        let decoded = partition::decode_admitted(
            &bytes,
            CanonicalLimits::new(256 * 1024 * 1024, 10_000_000, 1024 * 1024, 64),
            &NeverCancelled,
            |bytes| {
                assert_eq!(bytes, requested);
                scope
                    .consume_allocation_bytes(bytes)
                    .map_err(|error| CanonicalCodecError {
                        kind: CanonicalCodecErrorKind::ResourceLimit,
                        field: "dense round trip",
                        reason: error.to_string(),
                    })
            },
        )
        .unwrap();
        let ExactFacePartitionOutcome::Converged { faces } = decoded.outcome else {
            panic!("original dense inventory");
        };
        assert_eq!(faces.as_slice(), std::slice::from_ref(face));
        eprintln!(
            "DENSE_SURFACE_ROUND_TRIP wire={} typed_decode={} admitted_peak={}",
            bytes.len(),
            requested,
            memory.peak.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    assert_eq!(memory.live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Lossless indexed surface transport preserves original identities, quality and
//   signed-zero values, inventories typed reconstruction before decoding, and rejects denied
//   admission. This finite representation fixture does not establish CAD or PDE acceptance,
//   whole-task RSS, or arbitrary model capacity.
#[test]
fn indexed_surface_preserves_original_values_and_admits_before_decode() {
    use crate::surface_mesh::indexed_face::partition;
    use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
    let mut face = representative_face(3);
    face.nodes[0].uses[0].evaluator_uv[0] = -0.0;
    face.triangles[0].accepted_chordal_deviation_m = -0.0;
    face.triangles[1].unit_normal[0] = -0.0;
    let prep = ExactFacePartitionEncoding::single_preparation_storage_bytes(&face, &NeverCancelled)
        .unwrap();
    let prepared = ExactFacePartitionEncoding::single(&face, &NeverCancelled).unwrap();
    assert!(prepared.preparation_bytes() <= prep);
    let encoded = prepared.encode_prepared(&NeverCancelled).unwrap();
    let storage =
        ExactFacePartitionEncoding::decoding_storage_bytes(&encoded, &NeverCancelled).unwrap();
    let limits = CanonicalLimits::new(256 * 1024 * 1024, 10_000_000, 1024 * 1024, 64);
    let mut admitted = None;
    let denied = partition::decode_admitted(&encoded, limits, &NeverCancelled, |bytes| {
        admitted = Some(bytes);
        assert!(bytes > storage - 1);
        Err(CanonicalCodecError {
            kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
            field: "test decode admission",
            reason: "one byte below typed demand".into(),
        })
    })
    .unwrap_err();
    assert_eq!(admitted, Some(storage));
    assert_eq!(denied.kind, ExactSurfaceMeshErrorKind::ResourceLimit);
    let decoded = partition::decode_admitted(&encoded, limits, &NeverCancelled, |bytes| {
        assert_eq!(bytes, storage);
        Ok(())
    })
    .unwrap();
    let ExactFacePartitionOutcome::Converged { faces } = decoded.outcome else {
        panic!("converged original inventory");
    };
    assert_eq!(faces, vec![face.clone()]);
    assert_eq!(
        faces[0].nodes[0].uses[0].evaluator_uv[0].to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(
        faces[0].triangles[0].accepted_chordal_deviation_m.to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(
        faces[0].triangles[1].unit_normal[0].to_bits(),
        (-0.0f64).to_bits()
    );
    assert!(faces[0].retained_storage_bytes(&super::Control).unwrap() <= storage);
}

pub(super) fn verify_global_admission(
    bytes: &[u8],
    surface: &ExactSurfaceMesh,
    topology: &runmat_geometry_core::ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
) {
    let control = MemoryControl {
        live: 0.into(),
        peak: 0.into(),
        limit: 512 * 1024 * 1024,
    };
    {
        let scope = runmat_geometry_core::GeometryAllocationScope::new(&control);
        let decoded = crate::decode_published_exact_surface_mesh_admitted(
            bytes,
            topology,
            options,
            &NeverCancelled,
            &scope,
        )
        .unwrap();
        assert_eq!(&decoded, surface);
        let retained = decoded.retained_storage_bytes(&scope).unwrap();
        let storage = scope.transfer_reservation(retained).unwrap();
        drop(scope);
        assert_eq!(
            control.live.load(std::sync::atomic::Ordering::Relaxed),
            retained
        );
        drop(decoded);
        drop(storage);
    }
    assert_eq!(control.live.load(std::sync::atomic::Ordering::Relaxed), 0);
    let denied = MemoryControl {
        live: 0.into(),
        peak: 0.into(),
        limit: 1,
    };
    {
        let scope = runmat_geometry_core::GeometryAllocationScope::new(&denied);
        assert_eq!(
            crate::decode_published_exact_surface_mesh_admitted(
                bytes,
                topology,
                options,
                &NeverCancelled,
                &scope
            )
            .unwrap_err()
            .kind,
            ExactSurfaceMeshErrorKind::ResourceLimit
        );
    }
    assert_eq!(denied.live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

fn dense_global_round_trip(face: ExactFaceMesh) {
    use crate::surface_mesh::indexed_face::surface::{self, SurfaceEncoding};
    use runmat_geometry_core::{
        GeometryAllocationReservation, GeometryAllocationScope, GeometryEvaluationControl,
    };
    let mesh = ExactSurfaceMesh {
        schema_version: EXACT_SURFACE_MESH_SCHEMA_VERSION,
        face_ids: vec![face.source_face_id],
        nodes: face.nodes,
        triangles: face.triangles,
        boundary_segments: face.boundary_segments,
        shells: vec![],
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            face.maximum_chordal_deviation_m.canonical_value(),
        ),
        maximum_normal_deviation_rad: face.maximum_normal_deviation_rad,
    };
    let control = MemoryControl {
        live: 0.into(),
        peak: 0.into(),
        limit: 512 * 1024 * 1024,
    };
    {
        let scope = GeometryAllocationScope::new(&control);
        let typed = mesh.retained_storage_bytes(&scope).unwrap();
        scope.consume_allocation_bytes(typed).unwrap();
        let preparation = GeometryAllocationReservation::new(
            &scope,
            SurfaceEncoding::preparation_storage_bytes(&mesh, &NeverCancelled).unwrap(),
        )
        .unwrap();
        let prepared = SurfaceEncoding::new(&mesh, &NeverCancelled).unwrap();
        let profile = prepared.profile(&NeverCancelled).unwrap();
        let mut storage =
            GeometryAllocationReservation::new(&scope, profile.resident_workspace_bytes_estimate)
                .unwrap();
        let bytes = prepared.encode_profiled(&profile, &NeverCancelled).unwrap();
        storage.retain_bytes(bytes.capacity() as u64);
        drop(storage);
        drop(prepared);
        drop(preparation);
        let decoded =
            surface::decode_admitted(&bytes, surface::LIMITS, &NeverCancelled, Some(&scope))
                .unwrap();
        assert!(
            decoded == mesh,
            "global transport must preserve the complete original numerical arrays and metadata"
        );
        eprintln!(
            "DENSE_GLOBAL_SURFACE_ROUND_TRIP typed={} wire={} admitted_peak={}",
            typed,
            bytes.len(),
            control.peak.load(std::sync::atomic::Ordering::Relaxed)
        );
        // This tests representation and storage. Empty shell evidence and the
        // synthetic geometry intentionally grant no global CAD/PDE acceptance.
    }
    assert_eq!(control.live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// Exercise actual accepted partitions, including the native ownership transfer.
pub(super) fn verify_admitted_join(
    topology: &runmat_geometry_core::ExactBRepTopology,
    curves: &runmat_meshing_curve::SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    expected: &ExactSurfacePassResult,
) {
    use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let control = MemoryControl {
        live: AtomicU64::new(0),
        peak: AtomicU64::new(0),
        limit: 512 * 1024 * 1024,
    };
    let scope = GeometryAllocationScope::new(&control);
    let inputs = partitions.to_vec();
    let bytes = inputs
        .iter()
        .map(|part| part.retained_storage_bytes(&scope).unwrap())
        .sum();
    scope.consume_allocation_bytes(bytes).unwrap();
    let accepted = decide_exact_surface_pass_owned_admitted(
        curves,
        inputs,
        topology,
        ExactSurfaceJoinOptions::default(),
        &NeverCancelled,
        Some(&scope),
    )
    .unwrap();
    assert_eq!(accepted.result(), expected);
    if let ExactSurfacePassOutcome::Converged { surface } = &accepted.result().outcome {
        let bytes = surface.retained_storage_bytes(&scope).unwrap();
        let retained = scope.transfer_reservation(bytes).unwrap();
        drop(scope);
        assert_eq!(
            control.live.load(Relaxed),
            bytes,
            "only the final joined surface stays charged"
        );
        drop(accepted);
        drop(retained);
    } else {
        drop(accepted);
        drop(scope);
    }
    assert_eq!(control.live.load(Relaxed), 0);
    if matches!(expected.outcome, ExactSurfacePassOutcome::Converged { .. }) {
        let denied = MemoryControl {
            live: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            limit: 1,
        };
        let scope = GeometryAllocationScope::new(&denied);
        let failure = decide_exact_surface_pass_owned_admitted(
            curves,
            partitions.to_vec(),
            topology,
            ExactSurfaceJoinOptions::default(),
            &NeverCancelled,
            Some(&scope),
        )
        .err()
        .unwrap();
        assert_eq!(
            failure.kind,
            ExactSurfaceConvergenceErrorKind::Surface(ExactSurfaceMeshErrorKind::ResourceLimit)
        );
        drop(scope);
        assert_eq!(
            denied.live.load(Relaxed),
            0,
            "denied join must release all admitted scratch"
        );
    }
}

#[test]
fn compact_local_node_references_preserve_and_admit_foreign_identity_storage() {
    use crate::surface_mesh::indexed_face::partition;
    use runmat_canonical_codec::CanonicalLimits;
    let original = representative_face(2);
    let bytes = ExactFacePartitionEncoding::single(&original, &NeverCancelled)
        .unwrap()
        .encode_prepared(&NeverCancelled)
        .unwrap();
    let base = ExactFacePartitionEncoding::decoding_storage_bytes(&bytes, &NeverCancelled).unwrap();
    let mut foreign = original;
    let missing = StableDigest::from_bytes([241; 32]);
    assert!(foreign.nodes.iter().all(|node| node.node_id != missing));
    foreign.triangles[0].node_ids[0] = missing;
    let bytes = ExactFacePartitionEncoding::single(&foreign, &NeverCancelled)
        .unwrap()
        .encode_prepared(&NeverCancelled)
        .unwrap();
    let requested =
        ExactFacePartitionEncoding::decoding_storage_bytes(&bytes, &NeverCancelled).unwrap();
    assert_eq!(
        requested,
        base + std::mem::size_of::<StableDigest>() as u64,
        "the foreign digest allocation must be admitted before decoding"
    );
    let decoded = partition::decode_admitted(
        &bytes,
        CanonicalLimits::new(256 * 1024 * 1024, 10_000_000, 1024 * 1024, 64),
        &NeverCancelled,
        |bytes| {
            assert_eq!(bytes, requested);
            Ok(())
        },
    )
    .unwrap();
    let ExactFacePartitionOutcome::Converged { faces } = decoded.outcome else {
        unreachable!()
    };
    assert_eq!(
        faces[0], foreign,
        "foreign identities must survive for independent semantic rejection"
    );
}
