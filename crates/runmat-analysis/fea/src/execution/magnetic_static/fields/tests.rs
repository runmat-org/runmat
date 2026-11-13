use super::*;
use crate::physics::magnetic::test_support::authored_model;
use crate::results::field::{RecoveredFieldArtifact, VolumeFieldSamplingRule};
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::ObservationValueShape;

fn workload(current: f64) -> MagnetostaticWorkload {
    MagnetostaticWorkload {
        model: authored_model(current),
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
}

#[test]
fn magnetic_detached_native_fields_copy_exact_signed_edges_and_fifteen_point_vectors() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        for current in [-2.0, 0.0, 2.0] {
            let w = workload(current);
            let result = w.execute(&mesh).unwrap();
            let bytes = result.canonical_bytes().unwrap();
            let expected = native_magnetic_fields(&result)
                .map(|f| (f.field_id, f.shape, f.values.values().collect::<Vec<_>>()));
            let request = result.native_field_request(&w, &mesh).unwrap();
            drop(result);
            let retained = MagnetostaticSolverResult::from_canonical_bytes_for(
                &bytes,
                Digest::sha256(&bytes),
                &w,
                &mesh,
            )
            .unwrap();
            assert_eq!(
                retained.sampling_topology().rule(),
                VolumeFieldSamplingRule::TetrahedronKeastFifteen
            );
            let fields = retained
                .recover_native_fields_from_result(&request)
                .unwrap();
            assert_eq!(
                fields.entity_counts.volume_integration_points,
                mesh.topology.volume_elements.len() * 15
            );
            assert_eq!(
                fields.entity_counts.oriented_edges,
                retained
                    .response()
                    .edge_circulations_wb
                    .canonical_values()
                    .len()
            );
            assert_eq!(
                fields.topology_digest,
                request.topology.discretization_digest()
            );
            for ((field, requirement), (name, shape, values)) in fields
                .fields()
                .iter()
                .zip(&request.plan.requirements)
                .zip(expected)
            {
                assert_eq!(field.storage().field_id, name);
                assert_eq!(field.storage().shape, shape);
                assert_eq!(field.storage().as_host_f64().unwrap(), values);
                assert_eq!(field.definition().quantity(), requirement.quantity);
                assert_eq!(
                    field.definition().derivation(),
                    crate::results::field::RecoveredFieldDerivation::Native
                );
            }
            assert_eq!(
                request.plan.requirements[0].value_shape,
                ObservationValueShape::Scalar
            );
            assert_eq!(
                request.plan.requirements[0].quantity,
                ObservationQuantity::MagneticEdgeCirculation
            );
            assert_eq!(
                request.plan.requirements[0].output_location,
                FieldLocation::Edge
            );
            assert_eq!(
                request.plan.requirements[1].value_shape,
                ObservationValueShape::Vector3
            );
            assert_eq!(
                request.plan.requirements[1].output_location,
                FieldLocation::IntegrationPoint
            );
            // Element means and H come from the retained B, the mesh weights and the element
            // permeability, exactly as the per-sample values they summarize.
            let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
            let samples = retained.response().samples(&projection).unwrap();
            let element_flux = fields.fields()[4].storage().as_host_f64().unwrap();
            let intensity = fields.fields()[3].storage().as_host_f64().unwrap();
            for (element, group) in samples.chunks(15).enumerate() {
                let volume: f64 = group.iter().map(|s| s.volume_weight_m3).sum();
                for axis in 0..3 {
                    let mean: f64 = group
                        .iter()
                        .map(|s| {
                            s.flux_density_t.canonical_value()[axis] * s.volume_weight_m3 / volume
                        })
                        .sum();
                    assert_eq!(element_flux[3 * element + axis], mean);
                }
            }
            for (index, sample) in samples.iter().enumerate() {
                assert_eq!(
                    &intensity[3 * index..3 * index + 3],
                    &sample.magnetic_field_a_per_m.canonical_value()[..]
                );
            }
            let wire = fields.canonical_bytes().unwrap();
            assert_eq!(
                fields,
                RecoveredFieldArtifact::from_canonical_bytes(&wire).unwrap()
            );
        }
    }
}

#[test]
fn magnetic_native_fields_reject_wrong_sampling_source_and_edge_vector_relabeling() {
    let mesh = slab_mesh(2);
    let w = workload(2.0);
    let result = w.execute(&mesh).unwrap();
    let request = result.native_field_request(&w, &mesh).unwrap();
    let mut altered = request.clone();
    altered.topology = crate::results::field::SampledVolumeFieldTopology::derive(
        &mesh,
        VolumeFieldSamplingRule::TetrahedronFourPoint,
    )
    .unwrap()
    .identity()
    .unwrap();
    assert!(result.recover_native_fields_from_result(&altered).is_err());
    let mut altered = request.clone();
    altered.plan.requirements[0].quantity = ObservationQuantity::MagneticVectorPotential;
    altered.plan.requirements[0].value_shape = ObservationValueShape::Vector3;
    assert!(result.recover_native_fields_from_result(&altered).is_err());
    let foreign = workload(-2.0)
        .execute(&mesh)
        .unwrap()
        .native_field_request_from_result()
        .unwrap();
    assert!(result.recover_native_fields_from_result(&foreign).is_err());
    let mut wire = serde_json::to_value(&result).unwrap();
    wire["response"]["flux_density_t"] = serde_json::json!([]);
    assert!(serde_json::from_value::<MagnetostaticSolverResult>(wire).is_err());
    let mut wire = serde_json::to_value(&result).unwrap();
    wire["response"]["edge_circulations_wb"] = serde_json::json!([]);
    assert!(serde_json::from_value::<MagnetostaticSolverResult>(wire).is_err());
}
