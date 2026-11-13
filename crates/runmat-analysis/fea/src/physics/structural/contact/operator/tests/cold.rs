use super::*;
#[test]
fn cold_global_configuration_revalidates_original_point_order_frames_work_and_full_coordinates() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, true);
        let operator = f.operator(true);
        let initial = operator.initial_configuration(&f.positions).unwrap();
        let trial = operator
            .assemble_trial(&initial, 0, &f.moved(0.06, 0.02))
            .unwrap();
        let value = serde_json::to_value(&trial.configuration).unwrap();
        let cold: ContactConfigurationState = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(cold, trial.configuration);
        operator.validate_configuration(&cold, 1).unwrap();
        assert!(operator.validate_configuration(&cold, 0).is_err());
        assert!(value["points"][0].get("primary_coordinates_m").is_none());
        assert!(value["points"][0].get("secondary_coordinates_m").is_none());
        let mut reordered = value.clone();
        reordered["points"].as_array_mut().unwrap().swap(0, 1);
        assert!(operator
            .validate_configuration(&serde_json::from_value(reordered).unwrap(), 1)
            .is_err());
        let mut tampered = value.clone();
        tampered["points"][0]["point"]["tangential_traction_pa"][0] = serde_json::json!(1e12);
        assert!(operator
            .validate_configuration(&serde_json::from_value(tampered).unwrap(), 1)
            .is_err());
        let mut coordinates = value.clone();
        let node = f
            .projection
            .nodes
            .iter()
            .position(|n| n.node_id == 101)
            .unwrap();
        coordinates["positions_m"][node][2] =
            serde_json::json!(cold.positions_m()[node][2] + 0.001);
        assert!(operator
            .validate_configuration(&serde_json::from_value(coordinates).unwrap(), 1)
            .is_err());
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("positions_m");
        assert!(serde_json::from_value::<ContactConfigurationState>(missing).is_err());
        let mut overflow = value;
        overflow["ordinal"] = serde_json::json!(u64::MAX);
        let overflow: ContactConfigurationState = serde_json::from_value(overflow).unwrap();
        assert!(operator
            .assemble_trial(&overflow, u64::MAX, overflow.positions_m())
            .is_err());
        let replay = operator
            .assemble_trial(&initial, 0, &f.moved(0.06, 0.02))
            .unwrap();
        assert_eq!(trial.configuration, replay.configuration);
        assert_eq!(trial.residual, replay.residual);
    }
}

#[test]
fn changing_primary_face_rejects_without_transferring_or_mutating_original_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, true);
        let mut interfaces = f.interfaces.clone();
        let mut entities = interfaces[0].primary.resolved_entities().to_vec();
        entities.push(
            f.projection
                .boundary_faces
                .iter()
                .flat_map(|p| &p.provenance)
                .find(|p| p.source_topology_id == "contact-primary-side")
                .unwrap()
                .clone(),
        );
        interfaces[0]
            .primary
            .bind_resolved_entities(entities)
            .unwrap();
        let operator = CanonicalContactOperator::prepare(
            &f.projection,
            &interfaces,
            &f.rows,
            3 * f.rows.len(),
            &[Fixture::control(true)],
            Fixture::budget(),
        )
        .unwrap();
        let initial = operator.initial_configuration(&f.positions).unwrap();
        let retained = initial.clone();
        let current: Vec<_> = f
            .positions
            .iter()
            .zip(&f.projection.nodes)
            .map(|(p, n)| {
                if (100..200).contains(&n.node_id) {
                    [p[0], p[1] - 0.3, p[2] + 0.35]
                } else {
                    *p
                }
            })
            .collect();
        assert!(matches!(
            operator.assemble_trial(&initial, 0, &current),
            Err(ContactOperatorError::FeatureTransition)
        ));
        assert_eq!(initial, retained);
    }
}
