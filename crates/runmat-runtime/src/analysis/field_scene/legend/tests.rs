use super::*;
use runmat_analysis_core::{
    AssessmentEvidenceRequirement, AssessmentQuantity, AssessmentResult, AssessmentRule,
    AssessmentStatus, AuthoredAssessment, DisplayRange, DisplayUnitSource,
};
use runmat_types::PhysicalDimension;
use runmat_plot::legend_text::LegendPrecision;

fn kelvin(value: f64) -> AssessmentQuantity {
    AssessmentQuantity::try_from_descriptor(PhysicalDimension::TEMPERATURE.into(), value).unwrap()
}

fn assessment(rule: AssessmentRule, measured: f64) -> PublishedStudyAssessment {
    PublishedStudyAssessment {
        definition: AuthoredAssessment {
            assessment_id: "stays_below_85_celsius".into(),
            output_id: "peak_temperature".into(),
            rule,
            evidence: AssessmentEvidenceRequirement::AuthoredSamples,
        },
        result: AssessmentResult {
            assessment_id: "stays_below_85_celsius".into(),
            output_id: "peak_temperature".into(),
            status: AssessmentStatus::Passed,
            measured: Some(kelvin(measured)),
            issue: None,
        },
        study_definition: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            .parse()
            .unwrap(),
        run_id: None,
        evidence: None,
        function_execution_identity: None,
        function_execution_diagnostic: None,
        summary_line: None,
    }
}

fn display(unit: Option<&str>) -> AuthoredOutputDisplay {
    AuthoredOutputDisplay {
        output_id: "temperature".into(),
        unit: unit.map(str::to_string),
        unit_source: DisplayUnitSource::Study,
        range: None,
        precision: None,
        hide: Vec::new(),
        context: Vec::new(),
        deformation: None,
    }
}

fn busbar_legend() -> GeometrySceneScalarLegend {
    GeometrySceneScalarLegend::of_data(
        "Temperature".into(),
        "K".into(),
        329.9642170922981,
        336.5473805835758,
    )
}

#[test]
fn the_busbar_legend_reads_in_celsius_and_leads_with_its_peak() {
    let mut legend = busbar_legend();
    let terms = FieldLegendTerms {
        display: Some(display(Some("degC"))),
        assessments: vec![assessment(
            AssessmentRule::Maximum {
                bound: kelvin(358.15),
            },
            336.5473805835758,
        )],
    };
    assert_eq!(
        present(&mut legend, "K", &terms, None, &|_| None)
            .unwrap()
            .as_deref(),
        Some("degC")
    );
    let precision = LegendPrecision::for_range(legend.minimum, legend.maximum);
    assert_eq!(precision.format(legend.minimum), "56.8");
    assert_eq!(precision.format(legend.maximum), "63.4");
    assert_eq!(legend.unit, "degC");
    assert!((legend.limit.unwrap() - 85.0).abs() < 1e-9);
    assert_eq!(
        legend.note.as_deref(),
        Some("Peak 63.4 °C · limit 85 °C · margin 21.6 K")
    );
}

#[test]
fn without_a_display_or_assessment_the_legend_keeps_the_field_unit() {
    let mut legend = busbar_legend();
    assert_eq!(
        present(
            &mut legend,
            "K",
            &FieldLegendTerms::default(),
            None,
            &|_| None
        )
        .unwrap(),
        None
    );
    assert_eq!(legend.unit, "K");
    assert_eq!(legend.minimum, 329.9642170922981);
    assert!(legend.note.is_none() && legend.limit.is_none());
}

#[test]
fn a_broken_limit_still_states_the_result_first() {
    let mut legend = busbar_legend();
    let terms = FieldLegendTerms {
        display: None,
        assessments: vec![assessment(
            AssessmentRule::Maximum {
                bound: kelvin(333.15),
            },
            336.5473805835758,
        )],
    };
    assert_eq!(
        present(&mut legend, "K", &terms, None, &|_| None).unwrap(),
        None
    );
    assert_eq!(
        legend.note.as_deref(),
        Some("Peak 337 K · limit 333.15 K · 3.40 K over")
    );
}

#[test]
fn a_stress_limit_states_its_factor_and_overflows_convert_to_the_display_unit() {
    let mut stress = assessment(
        AssessmentRule::Maximum {
            bound: pascal(184e6),
        },
        0.0,
    );
    stress.result.measured = Some(pascal(17.7e6));
    let mut legend =
        GeometrySceneScalarLegend::of_data("Von Mises stress".into(), "Pa".into(), 0.0, 12e6);
    legend.above = Some(14.2e6);
    let terms = FieldLegendTerms {
        display: Some(AuthoredOutputDisplay {
            output_id: "von_mises".into(),
            unit: Some("MPa".into()),
            unit_source: DisplayUnitSource::Display,
            range: Some(DisplayRange {
                minimum: 0.0,
                maximum: 12e6,
            }),
            precision: Some(3),
            ..AuthoredOutputDisplay::unit_only("von_mises", DisplayUnitSource::Display)
        }),
        assessments: vec![stress],
    };
    assert_eq!(terms.fixed_range(), Some((0.0, 12e6)));
    assert_eq!(
        present(&mut legend, "Pa", &terms, None, &|_| None)
            .unwrap()
            .as_deref(),
        Some("MPa")
    );
    assert_eq!((legend.minimum, legend.maximum), (0.0, 12.0));
    assert!((legend.above.unwrap() - 14.2).abs() < 1e-9);
    assert_eq!(legend.precision, Some(3));
    assert_eq!(legend.overflow_lines(), ["14.2 MPa above the range"]);
    assert_eq!(
        legend.note.as_deref(),
        Some("Peak 17.7 MPa · limit 184 MPa · factor 10.4")
    );
}

fn pascal(value: f64) -> AssessmentQuantity {
    AssessmentQuantity::try_from_descriptor(PhysicalDimension::PRESSURE.into(), value).unwrap()
}

/// The yaw arm's assessment: the element von Mises peak over the arm (76.32 MPa) against
/// 184 MPa, read from the run's element von Mises field.
fn yaw_arm_stress() -> PublishedStudyAssessment {
    let mut stress = assessment(
        AssessmentRule::Maximum {
            bound: pascal(184e6),
        },
        0.0,
    );
    stress.result.measured = Some(pascal(76.32e6));
    stress.evidence = Some(
        runmat_analysis_fea::results::assessment::ScalarObservationEvidence {
            observation_id: runmat_analysis_core::ObservationId("peak_stress".into()),
            value: 76.32e6,
            source_identity: runmat_execution::Digest::sha256("peak stress"),
            sample: None,
            entity: None,
            source_field_id: Some("structural.von_mises".into()),
        },
    );
    stress
}

/// The yaw arm's display of its von Mises output, in MPa.
fn yaw_arm_display() -> AuthoredOutputDisplay {
    AuthoredOutputDisplay {
        unit: Some("MPa".into()),
        ..AuthoredOutputDisplay::unit_only("von_mises", DisplayUnitSource::Display)
    }
}

fn yaw_arm_locate(id: &str) -> Option<AnalysisFieldLocation> {
    match id {
        "structural.von_mises" => Some(AnalysisFieldLocation::Element),
        "structural.nodal_von_mises" => Some(AnalysisFieldLocation::Node),
        _ => None,
    }
}

/// The yaw arm's legend over the run field `field_id` at `location`.
fn yaw_arm_legend(field_id: &str, location: AnalysisFieldLocation) -> GeometrySceneScalarLegend {
    let terms = FieldLegendTerms {
        display: Some(yaw_arm_display()),
        assessments: vec![yaw_arm_stress()],
    };
    let mut legend =
        GeometrySceneScalarLegend::of_data("Von Mises stress".into(), "Pa".into(), 0.0, 76.32e6);
    present(
        &mut legend,
        "Pa",
        &terms,
        Some(ShownField { field_id, location }),
        &yaw_arm_locate,
    )
    .unwrap();
    legend
}

/// The yaw arm: the study assesses the element von Mises peak over the arm (76.3 MPa). A view
/// of the element field says so after its result; a view of the nodal average says so on its
/// label, so neither reads as a second peak.
#[test]
fn the_assessment_line_names_its_location_and_another_location_labels_the_legend() {
    let element = yaw_arm_legend("structural.von_mises", AnalysisFieldLocation::Element);
    assert_eq!(element.label, "Von Mises stress");
    assert_eq!(
        element.note.as_deref(),
        Some("Peak 76.3 MPa (element) · limit 184 MPa · factor 2.41")
    );
    // The bar's top and the line's result are the same number.
    assert!((element.maximum - 76.32).abs() < 1e-9);
    let nodal = yaw_arm_legend("structural.nodal_von_mises", AnalysisFieldLocation::Node);
    assert_eq!(nodal.label, "Von Mises stress, nodal average");
    assert_eq!(nodal.note, element.note);
}

/// Publication writes the yaw arm's assessment line in the unit of the display of the field
/// output the peak reads, naming the element location; the legend of that output draws the
/// same text.
#[test]
fn the_published_assessment_line_is_the_legend_line() {
    use runmat_analysis_core::{
        AuthoredExtremaTiePolicy, AuthoredOutput, AuthoredOutputExpression,
        AuthoredSpatialReduction, FieldLocation, ObservationQuantity,
    };
    let model = crate::analysis::tests::sample_solid_model();
    let step_id = model.sequence.steps[0].step_id.0.clone();
    let field = |output_id: &str, location| AuthoredOutput {
        output_id: output_id.into(),
        expression: AuthoredOutputExpression::AnalysisField {
            step_id: step_id.clone(),
            quantity: ObservationQuantity::VonMisesStress,
            location,
        },
        policy: Default::default(),
    };
    let peak = AuthoredOutput {
        output_id: "peak_stress".into(),
        expression: AuthoredOutputExpression::SpatialReduction {
            reduction: AuthoredSpatialReduction::Maximum,
            source: Box::new(AuthoredOutputExpression::OutputReference {
                output_id: "von_mises".into(),
            }),
            target: runmat_analysis_core::AuthoredRegionSource::Geometry(
                runmat_geometry_core::GeometryRegionRef {
                    module_path: "study.fea".into(),
                    geometry_id: model.geometry_id.clone(),
                    region_id: runmat_meshing_core::fixtures::fixture_region("arm"),
                    role: runmat_geometry_core::GeometryRegionRole::Solid,
                },
            ),
            ties: AuthoredExtremaTiePolicy::FirstCanonical,
        },
        policy: Default::default(),
    };
    let outputs = vec![field("von_mises", FieldLocation::Element), peak];
    let observations =
        runmat_analysis_core::lower_authored_observations(&outputs, &model.sequence.steps).unwrap();
    let producer = observations
        .iter()
        .find(|output| output.request.observation_id().0 == "peak_stress")
        .unwrap();
    let mut stress = yaw_arm_stress();
    stress.definition.output_id = "peak_stress".into();
    // A nodal display of another unit does not change the element peak's line.
    let nodal = AuthoredOutputDisplay {
        unit: Some("kPa".into()),
        ..AuthoredOutputDisplay::unit_only("nodal_von_mises", DisplayUnitSource::Display)
    };
    let published = crate::analysis::assessment::summary_line::published_line(
        &stress,
        producer,
        &observations,
        &[nodal, yaw_arm_display()],
    );
    let legend = yaw_arm_legend("structural.von_mises", AnalysisFieldLocation::Element);
    assert_eq!(
        published.as_deref(),
        Some("Peak 76.3 MPa (element) · limit 184 MPa · factor 2.41")
    );
    assert_eq!(published, legend.note);
    // Without a display the line reads in SI, as a legend of the field in Pa does.
    let si = crate::analysis::assessment::summary_line::published_line(
        &stress,
        producer,
        &observations,
        &[],
    );
    assert_eq!(
        si.as_deref(),
        Some("Peak 7.63e7 Pa (element) · limit 1.84e8 Pa · factor 2.41")
    );
}
