//! Local evaluation and distributed selection agree on the original physical endpoint.
use super::*;

#[test]
fn final_frequency_selection_preserves_coordinates_and_rejects_ambiguous_sources() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&canonical_tetrahedron_solver_mesh(order)).unwrap();
        let mut step = ProcedureStep::acoustic_harmonic("terminal-frequency-selection");
        step.independent_domain =
            runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
        let mut observation = spec("terminal", ObservationTarget::WholeModel);
        observation.quantity = ObservationQuantity::AcousticPressure;
        step.observations = vec![ObservationRequest::Field(observation)];
        let definition = RecoveredFieldDefinition::new(
            "acoustic.pressure.recovered",
            ObservationQuantity::AcousticPressure,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Scalar,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap();
        let plan = resolve_observation_plan(
            &step,
            &projection,
            std::slice::from_ref(&definition),
            ObservationLimits::default(),
        )
        .unwrap();
        for sample in [
            RecoveredFieldSample::FrequencyHertz(20.0),
            RecoveredFieldSample::FrequencyHertz(30.0),
            RecoveredFieldSample::FrequencyHertz(40.0),
            RecoveredFieldSample::FrequencyHertz(60.0),
            RecoveredFieldSample::TimeSeconds(40.0),
        ] {
            assert_eq!(
                resolved_observation_accepts_sample(&plan, &plan.observations[0], sample, None)
                    .unwrap(),
                sample == RecoveredFieldSample::FrequencyHertz(40.0)
            );
        }
        let fields = |sample| {
            catalog_at_sample(
                &projection,
                &definition,
                sample,
                vec![projection.nodes.len()],
                vec![7.0; projection.nodes.len()],
            )
        };
        let terminal = RecoveredFieldSample::FrequencyHertz(40.0);
        let results = evaluate_observations(&plan, &projection, &fields(terminal)).unwrap();
        assert_eq!(results[0].sample, terminal);
        assert_eq!(
            results[0].field.as_host_f64().unwrap(),
            vec![7.0; projection.nodes.len()]
        );
        assert!(evaluate_observations(
            &plan,
            &projection,
            &fields(RecoveredFieldSample::FrequencyHertz(20.0))
        )
        .is_err());
        let ambiguous = RecoveredFieldCatalog::new(
            &projection,
            [terminal, RecoveredFieldSample::Final]
                .into_iter()
                .map(|sample| {
                    RecoveredField::new(
                        definition.clone(),
                        sample,
                        runmat_analysis_core::AnalysisField::host_f64(
                            definition.field_id(), definition.layout(),
                            vec![projection.nodes.len()],
                            vec![7.0; projection.nodes.len()],
                        ),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        assert!(evaluate_observations(&plan, &projection, &ambiguous).is_err());
    }
}
