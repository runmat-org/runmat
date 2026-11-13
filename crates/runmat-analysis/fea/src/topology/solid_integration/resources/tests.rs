use super::*;
use runmat_meshing_core::fixtures;

#[test]
fn reference_workspace_increases_with_rule_but_retained_inventory_does_not() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let mut previous_derivation = 0;
        let mut retained = None;
        for rule in [
            CanonicalSolidIntegrationRule::Keast15,
            CanonicalSolidIntegrationRule::Duffy5,
            CanonicalSolidIntegrationRule::Duffy8,
            CanonicalSolidIntegrationRule::Duffy12,
            CanonicalSolidIntegrationRule::Duffy16,
        ] {
            let estimate = derivation_resident_bytes_estimate(&mesh, rule).unwrap();
            assert!(estimate > previous_derivation);
            previous_derivation = estimate;
            let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            let actual_retained = retained_resident_bytes_estimate(&plan).unwrap();
            assert!(estimate > actual_retained);
            assert_eq!(*retained.get_or_insert(actual_retained), actual_retained);
        }
    }
}

#[test]
fn malformed_raw_connectivity_is_counted_without_physical_preparation() {
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let rule = CanonicalSolidIntegrationRule::Keast15;
    let baseline = derivation_resident_bytes_estimate(&mesh, rule).unwrap();
    mesh.topology.volume_elements[0].node_ids.extend(1000..1100);
    assert!(CanonicalSolidIntegrationPlan::derive(&mesh, rule).is_err());
    assert!(derivation_resident_bytes_estimate(&mesh, rule).unwrap() > baseline);
    {
        let quantity_change = f64::NAN;
        let quantity_index = 0;
        let quantity_slot = &mut mesh.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(derivation_resident_bytes_estimate(&mesh, rule).is_err());
}

#[test]
fn inventory_profile_is_the_same_shared_codec_shape_as_owned_records() {
    let mesh = fixtures::slab_mesh(1);
    let rule = CanonicalSolidIntegrationRule::Duffy12;
    let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
    assert_eq!(
        profile(&ElementInventory { mesh: &mesh, rule }).unwrap(),
        profile(&plan.elements).unwrap()
    );
    let owned_reference = rule
        .reference_points()
        .map(|(position, weight)| (position.map(f64::to_bits), weight.to_bits()))
        .collect::<Vec<_>>();
    assert_eq!(
        profile(&ReferenceBits(rule)).unwrap(),
        profile(&owned_reference).unwrap()
    );
}

#[test]
fn checked_storage_overflow_does_not_mutate_accumulator() {
    assert_eq!(
        slots::<CanonicalSolidIntegrationElement>(usize::MAX),
        Err(SolidIntegrationPlanError::CountOverflow)
    );
    let mut total = u64::MAX;
    assert_eq!(
        add(&mut total, 1),
        Err(SolidIntegrationPlanError::CountOverflow)
    );
    assert_eq!(total, u64::MAX);
}
