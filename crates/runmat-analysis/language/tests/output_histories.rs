mod support;

use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(source: &str) -> CheckedFeaProgram {
    let host = TestHost::default()
        .with_module("project/unused.fea", "", 2)
        .with_geometry("project/unused.step", 3)
        .with_function("project/unused.m", "unused", Vec::new(), Vec::new());
    FeaCompiler::new(&host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", source, 1),
        },
    })
}

const TRANSIENT_HISTORY_STUDY: &str = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
    region body : solid {
        select result
        expect exactly_one
    }
}
mesh part on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {}
study response {
    geometry = geometry.part
    mesh = mesh.part
    model = model.structure
    analysis transient {
        structural transient {
            start_time = 0 s
            end_time = 1 s
            time_step = 0.1 s
        }
    }
    output displacement = analysis.transient.field.node.displacement
    output body_displacement = sample(output.displacement, over = geometry.part.region.body)
    output tip = interpolate(output.displacement, at = [1, 0.5, 0.5] m)
    output path = interpolate(
        output.displacement,
        at = [[0, 0.5, 0.5] m, [1, 0.5, 0.5] m],
    )
    output displacement_x = component(output.displacement, component = x)
    output tip_x = interpolate(output.displacement_x, at = [1, 0.5, 0.5] m)
    output tip_history {
        value = sample(
            source = output.tip_x,
            over = time,
            at = [0 s, 0.5 s, 1 s],
        )
        precision = real64
        compression = lossless
        retention = explicit_samples
        maximum_samples = 3
        partial_access = manifest_and_chunks
        purpose = publication
    }
    output peak = max(output.tip_history, over = time)
    output absolute_peak = absmax(output.tip_history, over = time)
    output average = mean(
        output.tip_history,
        over = time,
        weighting = trapezoidal_independent_domain,
    )
    output rms_value = rms(
        output.tip_history,
        over = time,
        weighting = uniform_samples,
    )
    output variance_value = variance(
        output.tip_history,
        over = time,
        weighting = uniform_samples,
    )
    output range = envelope(output.tip_history, over = time)
}
"#;

// witness: FEA-AUTHORING-OUTPUT-002/sampling_and_interpolation @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Bounded sampling, interpolation and series reductions are typed end to end.
#[test]
fn bounded_sampling_interpolation_and_series_reductions_are_typed_end_to_end() {
    let program = check(TRANSIENT_HISTORY_STUDY);
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let declaration = program
        .declarations
        .values()
        .find(|declaration| declaration.key.name == "response")
        .unwrap();
    assert_eq!(
        declaration.outputs["tip_history"],
        FeaType::History(FeaHistoryType {
            sample: Box::new(FeaType::Quantity(
                runmat_types::physical::DynamicDimension::try_new(
                    PhysicalDimension::LENGTH,
                    Some(runmat_types::physical::QuantityKind::Displacement),
                )
                .unwrap(),
            )),
            domain: Some(FeaSeriesDomain::Time),
        })
    );
    assert_eq!(
        declaration.outputs["variance_value"],
        FeaType::Quantity((PhysicalDimension::LENGTH.power(2).unwrap()).into())
    );
    assert_eq!(
        declaration.outputs["range"],
        FeaType::Envelope(Box::new(FeaType::Quantity(
            runmat_types::physical::DynamicDimension::try_new(
                PhysicalDimension::LENGTH,
                Some(runmat_types::physical::QuantityKind::Displacement),
            )
            .unwrap()
        )))
    );

    let study = program.studies.values().next().unwrap();
    let observations = &study.sequence.steps[0].observations;
    assert_eq!(observations.len(), 13);
    assert!(matches!(
        observations[1],
        runmat_analysis_core::ObservationRequest::Field(_)
    ));
    assert!(matches!(
        observations[2],
        runmat_analysis_core::ObservationRequest::Probe(_)
    ));
    assert!(matches!(
        observations[6],
        runmat_analysis_core::ObservationRequest::History(_)
    ));
    assert_eq!(
        observations[6].spec().unwrap().sampling,
        runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![0.0, 0.5, 1.0])
    );
    assert_eq!(
        observations[6].spec().unwrap().storage.retention,
        runmat_analysis_core::RetentionPolicy::ExplicitSamples { maximum_count: 3 }
    );
    let variance = observations[11].spec().unwrap();
    assert_eq!(
        variance.result_dimension(),
        Some(PhysicalDimension::LENGTH.power(2).unwrap())
    );
    assert_eq!(
        variance.result_quantity(),
        runmat_analysis_core::ObservationResultQuantity::Derived(
            PhysicalDimension::LENGTH.power(2).unwrap().into()
        )
    );
    assert!(matches!(
        variance.reduction.as_ref().unwrap().series,
        runmat_analysis_core::SeriesReduction::Variance {
            weighting: runmat_analysis_core::SeriesWeighting::UniformSamples
        }
    ));
}

#[test]
fn sampling_and_series_domains_must_match() {
    let wrong_units =
        TRANSIENT_HISTORY_STUDY.replace("at = [0 s, 0.5 s, 1 s]", "at = [0 Hz, 0.5 Hz, 1 Hz]");
    let program = check(&wrong_units);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("sample coordinate dimension does not match")));

    let wrong_domain = TRANSIENT_HISTORY_STUDY.replace(
        "output peak = max(output.tip_history, over = time)",
        "output peak = max(output.tip_history, over = frequency)",
    );
    let program = check(&wrong_domain);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("series reduction domain does not match")));
}

#[test]
fn series_weighting_has_one_canonical_spelling() {
    let source = TRANSIENT_HISTORY_STUDY.replace(
        "weighting = trapezoidal_independent_domain",
        "weighting = trapezoidal",
    );
    let program = check(&source);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("trapezoidal_independent_domain")));
}

#[test]
fn output_policy_is_closed_and_bounded() {
    let missing_bound = TRANSIENT_HISTORY_STUDY.replace("maximum_samples = 3", "");
    let program = check(&missing_bound);
    assert!(!program.is_valid());
    assert!(program.diagnostics.iter().any(|diagnostic| diagnostic
        .message
        .contains("maximum_relative_error")
        || diagnostic.message.contains("maximum_samples")));

    let unknown = TRANSIENT_HISTORY_STUDY.replace(
        "purpose = publication",
        "purpose = publication\n        mystery = enabled",
    );
    let program = check(&unknown);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("unknown output policy member")));

    let wrong_precision =
        TRANSIENT_HISTORY_STUDY.replace("precision = real64", "precision = complex128");
    let program = check(&wrong_precision);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("precision")));

    let lossy_solver_control = TRANSIENT_HISTORY_STUDY
        .replace(
            "compression = lossless",
            "compression = error_bounded\n        maximum_relative_error = 0.001",
        )
        .replace("purpose = publication", "purpose = solver_control");
    let program = check(&lossy_solver_control);
    assert!(!program.is_valid());
    assert!(program
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("solver control")));
}
