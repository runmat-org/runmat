use super::*;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Solver projection validates the original volume against its original derived metric
//   while preserving the authored request. Substitution with the authored base metric rejects. This
//   kernel check does not establish whole native execution or a physical accuracy envelope.
#[test]
fn projection_preserves_the_original_volume_metric_without_rewriting_the_authored_request() {
    let (exact_topology, exact_surface) = crate::cdt::constraints::tests::tetrahedron();
    let authored = request(ElementOrder::Tet4);
    let original = authored.clone();
    let mut derived = authored.metric.clone();
    derived.global_metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let options = volume_options();
    let volume = crate::cdt::construct_delaunay_volume_mesh(
        &exact_topology,
        &exact_surface,
        &derived,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let project = |volume_metric| {
        build_delaunay_solver_topology(
            DelaunaySolverTopologyInput {
                exact_topology: &exact_topology,
                exact_surface: &exact_surface,
                volume_mesh: &volume,
                volume_options: options,
                volume_metric,
                request: &authored,
                exact_evaluation: None,
            },
            DelaunaySolverTopologyOptions::default(),
            &NeverCancelled,
        )
    };
    let topology = project(&derived).unwrap();
    runmat_meshing_core::validate_solver_mesh_topology(&topology, &authored).unwrap();
    assert_eq!(authored, original);
    assert_eq!(
        project(&authored.metric).unwrap_err().kind,
        DelaunaySolverTopologyErrorKind::InvalidMesh
    );
}
