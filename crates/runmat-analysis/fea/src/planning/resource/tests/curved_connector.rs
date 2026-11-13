use super::*;

#[test]
fn curved_foundation_preflight_budgets_the_compiled_surface_samples() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    {
        let quantity_change = 0.1;
        let quantity_index = 2;
        let quantity_slot = &mut mesh.topology.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "curved_foundation".into(),
            kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:0"),
                ),
                modulus: runmat_analysis_core::FoundationModulus::new(1.0),
            },
        });

    let compiled =
        crate::assembly::explicit_connectors::compile_model_connector_plan(&model, Some(&mesh), 0)
            .unwrap();
    let crate::connectors::CanonicalConnectorEntry::SurfaceNormalFoundation { quadrature, .. } =
        &compiled.entries[0]
    else {
        panic!("compiled connector must retain surface quadrature");
    };
    assert!(quadrature.len() > 7);
    let samples = quadrature.len() as u64;
    let estimate = connector_buffers::estimate_connector_buffers(
        &model,
        Some(&projection),
        projection.nodes.len() as u64,
    )
    .unwrap();
    let metadata = 4_096 + 4 * "curved_foundation".len() as u64;
    let canonical = samples * 1_024;
    let transient = samples * 512 + 1_024;
    assert_eq!(
        estimate.assembly_peak_bytes,
        metadata + canonical + transient
    );
    assert_eq!(
        estimate.recovery_retained_bytes,
        2 * (metadata + canonical) + 1_024
    );
}
