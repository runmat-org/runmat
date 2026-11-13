use super::*;
use runmat_meshing_core::fixtures;

const RULES: [CanonicalSolidIntegrationRule; 5] = [
    CanonicalSolidIntegrationRule::Keast15,
    CanonicalSolidIntegrationRule::Duffy5,
    CanonicalSolidIntegrationRule::Duffy8,
    CanonicalSolidIntegrationRule::Duffy12,
    CanonicalSolidIntegrationRule::Duffy16,
];

#[test]
fn both_orders_have_exact_preallocation_and_deterministic_identity() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        for rule in RULES {
            let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            let second = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            assert_eq!(plan, second);
            plan.validate(&mesh, rule).unwrap();
            let nodes = if order == ElementOrder::Tet4 { 4 } else { 10 };
            assert_eq!(plan.elements().len(), mesh.topology.volume_elements.len());
            assert_eq!(
                plan.total_point_count(),
                rule.point_count() * plan.elements().len()
            );
            assert_eq!(
                plan.total_point_node_count(),
                plan.total_point_count() * nodes
            );
            for (index, element) in plan.elements().iter().enumerate() {
                assert_eq!(element.order, order);
                assert_eq!(element.node_count, nodes);
                assert_eq!(element.point_count, rule.point_count());
                assert_eq!(element.point_offset, index * rule.point_count());
            }
            let decoded: CanonicalSolidIntegrationPlan =
                serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
            decoded.validate(&mesh, rule).unwrap();
        }
    }
}

#[test]
fn exact_rule_ordered_reference_data_has_positive_weights_and_affine_moments() {
    for rule in RULES {
        let points = rule.reference_points().collect::<Vec<_>>();
        assert_eq!(points.len(), rule.point_count());
        let mut volume = 0.0;
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        for (position, weight) in points {
            assert!(weight.is_finite() && weight > 0.0);
            assert!((position.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert!(position
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0));
            volume += weight;
            for axis in 0..4 {
                first[axis] += weight * position[axis];
                second[axis] += weight * position[axis] * position[axis];
            }
        }
        assert!((volume - 1.0 / 6.0).abs() < 1e-13, "{rule:?}: {volume}");
        for axis in 0..4 {
            assert!((first[axis] - 1.0 / 24.0).abs() < 1e-13);
            assert!((second[axis] - 1.0 / 60.0).abs() < 1e-13);
        }
    }
    for (index, rule) in RULES.into_iter().enumerate() {
        for other in RULES.into_iter().skip(index + 1) {
            assert_ne!(
                rule.reference_identity().unwrap(),
                other.reference_identity().unwrap()
            );
        }
    }
}

#[test]
fn changed_rule_mesh_or_decoded_inventory_is_rejected() {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let rule = CanonicalSolidIntegrationRule::Keast15;
    let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
    assert_eq!(
        plan.validate(&mesh, CanonicalSolidIntegrationRule::Duffy5),
        Err(SolidIntegrationPlanError::AuthorityMismatch)
    );
    let mut other_mesh = mesh.clone();
    for node in &mut other_mesh.topology.nodes {
        {
            let quantity_change = 0.25;
            let quantity_index = 0;
            let quantity_slot = &mut node.coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
    }
    other_mesh.seal_canonical_digest().unwrap();
    assert_eq!(
        plan.validate(&other_mesh, rule),
        Err(SolidIntegrationPlanError::AuthorityMismatch)
    );

    let mut variants = Vec::new();
    let mut altered = plan.clone();
    altered.schema_version += 1;
    variants.push(altered);
    let mut altered = plan.clone();
    altered.elements.clear();
    variants.push(altered);
    let mut altered = plan.clone();
    altered.elements.push(altered.elements[0].clone());
    variants.push(altered);
    let mut altered = plan.clone();
    altered.elements[0].point_offset += 1;
    variants.push(altered);
    let mut altered = plan.clone();
    altered.elements[0].point_count += 1;
    variants.push(altered);
    let mut altered = plan.clone();
    altered.elements[0].node_count += 1;
    variants.push(altered);
    let mut altered = plan.clone();
    altered.reference_identity = Digest::sha256(b"wrong reference ordering");
    variants.push(altered);
    let mut altered = plan.clone();
    altered.identity = Digest::sha256(b"wrong plan");
    variants.push(altered);
    let mut altered = plan.clone();
    altered.total_point_node_count += 1;
    variants.push(altered);
    for altered in variants {
        assert_eq!(
            altered.validate(&mesh, rule),
            Err(SolidIntegrationPlanError::AuthorityMismatch)
        );
    }
}

#[test]
fn ordered_element_inventory_is_not_interchangeable() {
    let mesh = fixtures::slab_mesh(1);
    let rule = CanonicalSolidIntegrationRule::Duffy8;
    let mut plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
    assert!(plan.elements.len() > 1);
    plan.elements.swap(0, 1);
    assert_eq!(
        plan.validate(&mesh, rule),
        Err(SolidIntegrationPlanError::AuthorityMismatch)
    );
}

#[test]
fn count_overflow_fails_without_partial_accumulation() {
    for (mut points, mut point_nodes, added, nodes) in [
        (usize::MAX, 0, 1, 4),
        (0, 0, usize::MAX, 10),
        (0, usize::MAX, 1, 4),
    ] {
        let before = (points, point_nodes);
        assert_eq!(
            accumulate_counts(&mut points, &mut point_nodes, added, nodes),
            Err(SolidIntegrationPlanError::CountOverflow)
        );
        assert_eq!((points, point_nodes), before);
    }
}
