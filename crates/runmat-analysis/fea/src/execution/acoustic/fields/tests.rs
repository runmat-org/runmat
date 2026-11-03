use super::*;
use crate::{
    results::field::{canonical_field_partitions, reconstruct_recovered_field},
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
};

#[test]
fn sampled_native_catalog_profiles_cover_partition_boundaries() {
    use crate::results::field::{
        RecoveredFieldArtifactManifest, RecoveredFieldPartition,
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
    };
    let mesh = slab_mesh(1);
    let result = super::super::tests::workload().execute(&mesh).unwrap();
    let source = result.native_field_request_from_result().unwrap();
    // Shape-only synthetic values, not independent numerical/physics evidence.
    for count in [
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT - 1,
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
        RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1,
        2 * RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 1,
    ] {
        let request = FieldRecoveryRequest::new(
            source.producer,
            source.source,
            source.topology,
            native_acoustic_field_plan(count, count).unwrap(),
        )
        .unwrap();
        let definitions = acoustic_native_field_definitions(&request).unwrap();
        let counts = vec![count; definitions.len()];
        for frequency in [0.1, 0.2, 0.3] {
            let sample = RecoveredFieldSample::FrequencyHertz(frequency);
            let fields = definitions
                .iter()
                .map(|definition| {
                    let shape = definition.storage_shape(count).unwrap();
                    let width = definition.storage_component_count().unwrap();
                    RecoveredField::new(
                        definition.clone(),
                        sample,
                        AnalysisField::host_f64(
                            definition.field_id(),
                            definition.layout(),
                            shape,
                            vec![f64::MAX; count * width],
                        ),
                    )
                    .unwrap()
                })
                .collect();
            let artifact = RecoveredFieldArtifact::new(
                request.identity(),
                request.source,
                request.topology.discretization_digest(),
                RecoveredFieldEntityCounts {
                    nodes: count,
                    volume_elements: count,
                    ..native_acoustic_entity_counts(&result)
                },
                fields,
            )
            .unwrap();
            let profile = RecoveredFieldArtifactManifest::profile_sampled_shape(
                &request,
                &definitions,
                &counts,
                sample,
            )
            .unwrap();
            let manifest = artifact.manifest().unwrap().canonical_bytes().unwrap();
            assert!(manifest.len() as u64 <= profile.encoded_bytes_upper_bound);
            assert!(
                runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                    &manifest
                )
                .unwrap()
                    <= profile.materialization_resident_bytes_estimate
            );
            for field in artifact.fields() {
                let partitions = canonical_field_partitions(field).unwrap();
                assert_eq!(
                    partitions.len(),
                    count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
                );
                for partition in partitions {
                    let projected = RecoveredFieldPartition::profile_shape(
                        field.definition(),
                        sample,
                        partition.source_indices.len(),
                    )
                    .unwrap();
                    let bytes = partition.canonical_bytes().unwrap();
                    assert!(bytes.len() as u64 <= projected.encoded_bytes_upper_bound);
                    assert!(runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap()
                        <= projected.materialization_resident_bytes_estimate);
                }
            }
        }
    }
}

#[test]
fn cold_complex_acoustic_fields_use_generic_domains_and_partition_exactly() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(4)
        };
        let workload = super::super::tests::workload();
        let result = workload.execute(&mesh).unwrap();
        let request = result.native_field_request(&workload, &mesh).unwrap();
        assert!(matches!(
            request.topology,
            FieldTopologyIdentity::VolumeMesh { .. }
        ));
        let request =
            FieldRecoveryRequest::from_canonical_bytes(&request.canonical_bytes().unwrap())
                .unwrap();
        let artifact = result
            .recover_native_fields(&workload, &mesh, &request)
            .unwrap();
        let cold =
            RecoveredFieldArtifact::from_canonical_bytes(&artifact.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(cold, artifact);
        assert_eq!(
            cold.source,
            FieldRecoverySource::SolverResult(result.source_result_identity().unwrap())
        );
        assert_eq!(cold.entity_counts.nodes, mesh.topology.nodes.len());
        assert_eq!(
            cold.entity_counts.volume_elements,
            mesh.topology.volume_elements.len()
        );
        assert_eq!(cold.entity_counts.volume_integration_points, 0);
        let pressure = &cold.fields()[0];
        let velocity = &cold.fields()[1];
        assert_eq!(
            pressure.definition().entity_domain(),
            RecoveredFieldEntityDomain::Nodes
        );
        assert_eq!(
            velocity.definition().entity_domain(),
            RecoveredFieldEntityDomain::VolumeElements
        );
        assert_eq!(pressure.sample(), RecoveredFieldSample::FrequencyHertz(0.2));
        assert_eq!(pressure.storage().shape, vec![mesh.topology.nodes.len(), 2]);
        assert_eq!(
            velocity.storage().shape,
            vec![mesh.topology.volume_elements.len(), 3, 2]
        );
        let AnalysisFieldValues::HostF64(values) = &pressure.storage().values else {
            panic!("host field");
        };
        // Independent matched-duct pressure, exp(+i omega t), rho*c=6.
        let mean_square = mesh
            .topology
            .nodes
            .iter()
            .zip(values.chunks_exact(2))
            .map(|(node, p)| {
                let phase = std::f64::consts::PI * 0.2 * node.coordinates_m.canonical_value()[0];
                (p[0] - phase.cos()).powi(2) + (p[1] + phase.sin()).powi(2)
            })
            .sum::<f64>()
            / mesh.topology.nodes.len() as f64;
        assert!(mean_square.sqrt() < 0.01);
        let AnalysisFieldValues::HostF64(values) = &velocity.storage().values else {
            panic!("host field");
        };
        for (stored, native) in values.chunks_exact(6).zip(result.velocity_m_per_s()) {
            assert_eq!(
                stored,
                &[
                    native.real[0],
                    native.imaginary[0],
                    native.real[1],
                    native.imaginary[1],
                    native.real[2],
                    native.imaginary[2]
                ]
            );
        }
        // Independently integrate the exact plane wave over each straight physical
        // tetrahedron using simplex polynomial moments, not FE recovery quadrature.
        let mut velocity_error = 0.0;
        for (element, stored) in mesh
            .topology
            .volume_elements
            .iter()
            .zip(values.chunks_exact(6))
        {
            let xs = std::array::from_fn(|corner| {
                mesh.topology
                    .nodes
                    .iter()
                    .find(|node| node.node_id == element.node_ids[corner])
                    .unwrap()
                    .coordinates_m
                    .canonical_value()[0]
            });
            let exact = mean_plane_wave(xs, std::f64::consts::PI * 0.2);
            velocity_error += (stored[0] - exact[0] / 6.0).powi(2)
                + (stored[1] - exact[1] / 6.0).powi(2)
                + stored[2..].iter().map(|value| value * value).sum::<f64>();
        }
        assert!((velocity_error / mesh.topology.volume_elements.len() as f64).sqrt() < 0.01);
        for field in cold.fields() {
            let partitions = canonical_field_partitions(field).unwrap();
            let restored = reconstruct_recovered_field(
                field.identity(),
                field.definition().clone(),
                field.sample(),
                partitions,
                field.storage().shape[0],
            )
            .unwrap();
            assert_eq!(&restored, field);
        }
    }
}

// For a uniform tetrahedron, E[x^n]=6*h_n(x0,x1,x2,x3)/((n+1)(n+2)(n+3)),
// where h_n is the complete homogeneous symmetric polynomial. This analytic
// moment formula integrates the exponential series without a solver reference.
fn mean_plane_wave(xs: [f64; 4], k: f64) -> [f64; 2] {
    let mut homogeneous = [0.0; 25];
    homogeneous[0] = 1.0;
    for x in xs {
        for n in 1..homogeneous.len() {
            homogeneous[n] += x * homogeneous[n - 1];
        }
    }
    let mut coefficient = [1.0, 0.0];
    let mut mean = [0.0, 0.0];
    for (n, h) in homogeneous.into_iter().enumerate() {
        if n > 0 {
            coefficient = [
                coefficient[1] * k / n as f64,
                -coefficient[0] * k / n as f64,
            ];
        }
        let moment = 6.0 * h / ((n + 1) * (n + 2) * (n + 3)) as f64;
        mean[0] += coefficient[0] * moment;
        mean[1] += coefficient[1] * moment;
    }
    mean
}

#[test]
fn independent_simplex_moments_reproduce_constant_coordinate_wave() {
    assert_eq!(mean_plane_wave([0.0; 4], 0.628), [1.0, 0.0]);
    let phase: f64 = 0.628 * 0.5;
    let mean = mean_plane_wave([0.5; 4], 0.628);
    assert!((mean[0] - phase.cos()).abs() < 1e-14);
    assert!((mean[1] + phase.sin()).abs() < 1e-14);
}

#[test]
fn acoustic_field_publication_rejects_foreign_source_plan_topology_and_order() {
    let mesh = slab_mesh(2);
    let workload = super::super::tests::workload();
    let result = workload.execute(&mesh).unwrap();
    let request = result.native_field_request(&workload, &mesh).unwrap();
    let wrong_variant = FieldRecoveryRequest::new(
        request.producer,
        request.source,
        FieldTopologyIdentity::Structural {
            topology:
                crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: mesh.canonical_digest,
                },
        },
        request.plan.clone(),
    )
    .unwrap();
    assert_eq!(
        wrong_variant.topology.discretization_digest(),
        request.topology.discretization_digest()
    );
    assert!(result
        .recover_native_fields(&workload, &mesh, &wrong_variant)
        .is_err());
    let mut other = workload.clone();
    other.solver.relative_tolerance = 5e-10;
    assert!(result
        .recover_native_fields(&other, &mesh, &request)
        .is_err());
    let mut plan = request.plan.clone();
    plan.requirements.pop();
    let wrong_plan =
        FieldRecoveryRequest::new(request.producer, request.source, request.topology, plan)
            .unwrap();
    assert!(result
        .recover_native_fields(&workload, &mesh, &wrong_plan)
        .is_err());
    let mut reordered = serde_json::to_value(&result).unwrap();
    reordered["ordered_node_ids"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    let reordered: AcousticHarmonicSolverResult = serde_json::from_value(reordered).unwrap();
    assert!(reordered.native_field_request(&workload, &mesh).is_err());
    let mut malformed = serde_json::to_value(&result).unwrap();
    malformed["velocity_m_per_s"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<AcousticHarmonicSolverResult>(malformed).is_err());
}
