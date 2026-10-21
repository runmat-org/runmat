use super::*;

fn demand(
    order: ElementOrder,
    rule: CanonicalSolidIntegrationRule,
    increments: usize,
    line_search: bool,
) -> NonlinearSolidStorageDemand {
    let (_, mesh, initial) = super::super::tests::fixture(order, true);
    let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
    let dofs = mesh.topology.nodes.len() * 3;
    NonlinearSolidStorageDemand::derive(
        &plan,
        initial.structural_property_map_identity.as_ref().unwrap(),
        dofs,
        dofs * dofs,
        increments,
        line_search,
    )
    .unwrap()
}

#[test]
fn point_history_and_gradients_follow_exact_rule_and_order() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let coarse = demand(order, CanonicalSolidIntegrationRule::Keast15, 12, true);
        let fine = demand(order, CanonicalSolidIntegrationRule::Duffy16, 12, true);
        assert!(fine.prepared_basis_bytes > coarse.prepared_basis_bytes);
        assert!(fine.point_history_bytes > coarse.point_history_bytes);
        assert!(fine.evaluation_bytes > coarse.evaluation_bytes);
        assert!(fine.authority_construction_bytes > coarse.authority_construction_bytes);
        assert_eq!(fine.retained_output_bytes, coarse.retained_output_bytes);
        assert_eq!(
            fine.additional_tangent_bytes,
            coarse.additional_tangent_bytes
        );
        assert_eq!(
            fine.tangent_contribution_bytes,
            coarse.tangent_contribution_bytes
        );
    }
    let linear = demand(
        ElementOrder::Tet4,
        CanonicalSolidIntegrationRule::Keast15,
        12,
        true,
    );
    let quadratic = demand(
        ElementOrder::Tet10,
        CanonicalSolidIntegrationRule::Keast15,
        12,
        true,
    );
    assert!(quadratic.prepared_basis_bytes > linear.prepared_basis_bytes);
    assert!(quadratic.additional_tangent_bytes > linear.additional_tangent_bytes);
    assert_eq!(quadratic.point_history_bytes, linear.point_history_bytes);
}

#[test]
fn full_accepted_cap_and_line_search_are_not_nominal_increment_proxies() {
    let one = demand(
        ElementOrder::Tet10,
        CanonicalSolidIntegrationRule::Keast15,
        1,
        false,
    );
    let many = demand(
        ElementOrder::Tet10,
        CanonicalSolidIntegrationRule::Keast15,
        4096,
        false,
    );
    assert!(many.retained_output_bytes > one.retained_output_bytes);
    assert_eq!(many.prepared_basis_bytes, one.prepared_basis_bytes);
    assert_eq!(many.evaluation_bytes, one.evaluation_bytes);
    let searched = demand(
        ElementOrder::Tet10,
        CanonicalSolidIntegrationRule::Keast15,
        1,
        true,
    );
    assert_eq!(
        searched.nonlinear_peak_bytes - one.nonlinear_peak_bytes,
        one.evaluation_bytes
    );
}

#[test]
fn nonlinear_peak_includes_original_point_state_and_more_than_field_snapshots() {
    let value = demand(
        ElementOrder::Tet10,
        CanonicalSolidIntegrationRule::Duffy8,
        12,
        true,
    );
    assert!(value.nonlinear_peak_bytes > value.retained_output_bytes);
    assert!(value.nonlinear_peak_bytes > value.prepared_basis_bytes);
    assert!(value.evaluation_bytes > value.point_history_bytes);
    assert!(value.additional_tangent_bytes > 0);
    assert!(value.tangent_contribution_bytes > 0);
}

#[test]
fn invalid_and_overflowing_caps_fail_before_allocation() {
    let (_, mesh, initial) = super::super::tests::fixture(ElementOrder::Tet4, true);
    let properties = initial.structural_property_map_identity.as_ref().unwrap();
    let plan = CanonicalSolidIntegrationPlan::derive(&mesh, CanonicalSolidIntegrationRule::Keast15)
        .unwrap();
    assert!(NonlinearSolidStorageDemand::derive(&plan, properties, 0, 144, 12, true).is_err());
    assert!(NonlinearSolidStorageDemand::derive(&plan, properties, 12, 144, 0, true).is_err());
    assert!(
        NonlinearSolidStorageDemand::derive(&plan, properties, usize::MAX, 144, 12, true).is_err()
    );
    assert!(
        NonlinearSolidStorageDemand::derive(&plan, properties, 12, usize::MAX, 12, true).is_err()
    );
    assert!(
        NonlinearSolidStorageDemand::derive(&plan, properties, 12, 144, usize::MAX, true).is_err()
    );
}

#[test]
fn point_schema_composition_covers_actual_full_basis_authority_for_every_rule() {
    use runmat_canonical_codec::CanonicalLimits;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = super::super::tests::fixture(order, true);
        for rule in [
            CanonicalSolidIntegrationRule::Keast15,
            CanonicalSolidIntegrationRule::Duffy5,
            CanonicalSolidIntegrationRule::Duffy8,
            CanonicalSolidIntegrationRule::Duffy12,
            CanonicalSolidIntegrationRule::Duffy16,
        ] {
            let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            let basis = CanonicalNonlinearSolidBasis::prepare_with_integration(
                &model, &mesh, &initial, &plan,
            )
            .unwrap();
            let authority = super::super::authority::BasisAuthority {
                integration_identity: basis.integration_identity,
                tangent_pattern_identity: basis.tangent_pattern.identity().unwrap(),
                properties: &basis.property_identity,
                layout: &initial.structural_dof_layout,
                elements: basis
                    .elements
                    .iter()
                    .map(|e| (e.identity, e.dofs.as_slice(), e.points.as_slice()))
                    .collect(),
            };
            // The authority is encoded directly; its witness is the direct encoder's workspace.
            let actual = runmat_canonical_codec::profile_canonical_encoding(
                &authority,
                CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64),
            )
            .unwrap();
            let demand = NonlinearSolidStorageDemand::derive(
                &plan,
                &basis.property_identity,
                initial.dof_count,
                initial.dof_count * initial.dof_count,
                8,
                true,
            )
            .unwrap();
            assert!(
                demand.authority_construction_bytes >= actual.resident_workspace_bytes_estimate,
                "{order:?}/{rule:?}: {} < {}",
                demand.authority_construction_bytes,
                actual.resident_workspace_bytes_estimate
            );
            assert!(demand.recovery_peak_bytes < demand.nonlinear_peak_bytes);
            // Extraction preserved the exact original authority encoding and identity.
            let encoded = encode_contract(
                b"runmat-fea-nonlinear-solid-basis/v1\0",
                "nonlinear-solid-basis",
                &authority,
                CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64),
            )
            .unwrap();
            assert_eq!(Digest::sha256(encoded), basis.identity());
        }
    }
}
