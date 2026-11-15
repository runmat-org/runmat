use super::*;
use runmat_analysis_core::{
    AssessmentEvidenceRequirement, AssessmentQuantity, AssessmentResult, AuthoredAssessment,
    DisplayRange, DisplayUnitSource,
};
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
    let pascal = |value: f64| {
        AssessmentQuantity::try_from_descriptor(PhysicalDimension::PRESSURE.into(), value).unwrap()
    };
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

/// The yaw arm: the study assesses the element von Mises peak over the arm (76.3 MPa). A view
/// of the element field says so after its result; a view of the nodal average says so on its
/// label, so neither reads as a second peak.
#[test]
fn the_assessment_line_names_its_location_and_another_location_labels_the_legend() {
    let pascal = |value: f64| {
        AssessmentQuantity::try_from_descriptor(PhysicalDimension::PRESSURE.into(), value).unwrap()
    };
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
    let terms = FieldLegendTerms {
        display: Some(AuthoredOutputDisplay {
            unit: Some("MPa".into()),
            ..AuthoredOutputDisplay::unit_only("von_mises", DisplayUnitSource::Display)
        }),
        assessments: vec![stress],
    };
    let locate = |id: &str| match id {
        "structural.von_mises" => Some(AnalysisFieldLocation::Element),
        "structural.nodal_von_mises" => Some(AnalysisFieldLocation::Node),
        _ => None,
    };
    let drawn = |field_id: &str, location| {
        let mut legend = GeometrySceneScalarLegend::of_data(
            "Von Mises stress".into(),
            "Pa".into(),
            0.0,
            76.32e6,
        );
        present(
            &mut legend,
            "Pa",
            &terms,
            Some(ShownField { field_id, location }),
            &locate,
        )
        .unwrap();
        legend
    };
    let element = drawn("structural.von_mises", AnalysisFieldLocation::Element);
    assert_eq!(element.label, "Von Mises stress");
    assert_eq!(
        element.note.as_deref(),
        Some("Peak 76.3 MPa (element) · limit 184 MPa · factor 2.41")
    );
    // The bar's top and the line's result are the same number.
    assert!((element.maximum - 76.32).abs() < 1e-9);
    let nodal = drawn("structural.nodal_von_mises", AnalysisFieldLocation::Node);
    assert_eq!(nodal.label, "Von Mises stress, nodal average");
    assert_eq!(nodal.note, element.note);
}
