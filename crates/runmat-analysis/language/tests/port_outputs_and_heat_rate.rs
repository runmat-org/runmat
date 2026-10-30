pub mod support;
use runmat_analysis_core::{
    AuthoredExcitationKind, ExcitationDistribution, FieldLocation, ObservationQuantity,
    ObservationRequest, ObservationTarget, PhysicalSignal, SpatialReduction,
};
use runmat_analysis_language::*;
use support::{source_module, TestHost};

fn check(materials: &str, model: &str, analysis: &str, outputs: &str) -> CheckedFeaProgram {
    let source = format!(
        r#"
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region body : solid {{ select result; expect exactly_one }}
    region inlet : face {{ select result; expect one_or_more }}
    region outlet : face {{ select result; expect one_or_more }}
}}
mesh part on geometry.part {{
    tetrahedra {{ order = linear
        target_size = 0.5 m
        maximum_deviation = 0.01 m }}
}}
{materials}
model physics {{ {model} }}
study response {{
    geometry = geometry.part
    mesh = mesh.part
    model = model.physics
    analysis run {{ {analysis} }}
    {outputs}
}}
"#
    );
    FeaCompiler::new(&TestHost::default()).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new("project").unwrap(),
            root: source_module("project/main.fea", &source, 1),
        },
    })
}

fn reduction(program: &CheckedFeaProgram) -> runmat_analysis_core::ObservationSpec {
    let step = &program.studies.values().next().unwrap().sequence.steps[0];
    step.observations
        .iter()
        .find_map(|request| match request {
            ObservationRequest::Reduction(spec) => Some(spec.clone()),
            _ => None,
        })
        .expect("reduction observation")
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: pressure_drop checks on a fluid analysis and lowers to the mean difference of the
//   finite-volume face pressure over the two named face regions, in from/to order.
#[test]
fn pressure_drop_lowers_to_a_paired_face_pressure_mean_difference() {
    let program = check(
        "",
        "",
        "fluid steady {}",
        "output dp = pressure_drop(analysis.run, from = geometry.part.region.inlet, to = geometry.part.region.outlet)",
    );
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let spec = reduction(&program);
    assert_eq!(spec.quantity, ObservationQuantity::FluidPressure);
    assert_eq!(spec.source_location, FieldLocation::Face);
    assert_eq!(spec.output_location, FieldLocation::Global);
    assert_eq!(spec.spatial_reduction(), SpatialReduction::MeanDifference);
    let ObservationTarget::RegionPair { first, second } = &spec.target else {
        panic!("pressure_drop reads two regions: {:?}", spec.target);
    };
    assert!(first.region_id.contains("inlet") && second.region_id.contains("outlet"));
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: transmission_loss checks on an acoustic harmonic analysis and lowers to the paired
//   dB ratio of the acoustic port intensity, incident region first.
#[test]
fn transmission_loss_lowers_to_the_paired_port_intensity_ratio() {
    let program = check(
        "",
        "",
        "acoustic harmonic { frequencies = [100, 200] Hz }",
        "output tl = transmission_loss(analysis.run, incident = geometry.part.region.inlet, transmitted = geometry.part.region.outlet)",
    );
    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    let spec = reduction(&program);
    assert_eq!(spec.quantity, ObservationQuantity::AcousticPortIntensity);
    assert_eq!(spec.spatial_reduction(), SpatialReduction::TransmissionLoss);
    assert_eq!(
        spec.result_quantity(),
        runmat_analysis_core::ObservationResultQuantity::Physical(
            ObservationQuantity::TransmissionLoss
        )
    );
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: port outputs reject solid regions and analyses that do not produce their field.
#[test]
fn port_outputs_reject_solid_regions_and_other_physics() {
    assert!(!check(
        "",
        "",
        "fluid steady {}",
        "output dp = pressure_drop(analysis.run, from = geometry.part.region.body, to = geometry.part.region.outlet)",
    )
    .is_valid());
    assert!(!check(
        "",
        "",
        "acoustic harmonic { frequencies = [100] Hz }",
        "output dp = pressure_drop(analysis.run, from = geometry.part.region.inlet, to = geometry.part.region.outlet)",
    )
    .is_valid());
    assert!(!check(
        "",
        "",
        "fluid steady {}",
        "output tl = transmission_loss(analysis.run, incident = geometry.part.region.inlet, transmitted = geometry.part.region.outlet)",
    )
    .is_valid());
}

// witness: FEA-PUBLIC-LANGUAGE-001/output_expression_forms @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: a heat generation authored in W lowers to a heat-rate signal with resultant-
//   preserving distribution; one in W/m^3 stays a density.
#[test]
fn heat_generation_in_watts_is_a_total_heat_rate() {
    let material = r#"material copper {
    mass_density = 8960 kg/m^3
    thermal {
        reference_temperature = 293.15 K
        modulus_temperature_coefficient = 0 K^-1
        conductivity = 400 W/m/K
        specific_heat = 385 J/kg/K
        expansion_coefficient = 0.0000165 K^-1
    }
}"#;
    for (value, distribution, watts) in [
        ("2 W", ExcitationDistribution::PreserveResultant, true),
        ("2e6 W/m^3", ExcitationDistribution::IntensityPerVolume, false),
    ] {
        let model = format!(
            "assign material.copper to geometry.part.region.body\n    load heater {{ heat_generation on geometry.part.region.body = {value} }}\n    load_case running {{ include load.heater }}"
        );
        let program = check(material, &model, "thermal steady {}", "");
        assert!(program.is_valid(), "{value}: {:#?}", program.diagnostics);
        let model = program
            .models
            .values()
            .find(|model| {
                model
                    .excitations
                    .iter()
                    .any(|excitation| excitation.kind == AuthoredExcitationKind::HeatGeneration)
            })
            .expect("model with a heat generation");
        let excitation = model
            .excitations
            .iter()
            .find(|excitation| excitation.kind == AuthoredExcitationKind::HeatGeneration)
            .expect("heat generation");
        assert_eq!(excitation.distribution, distribution);
        let signal = model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &excitation.signal_id)
            .expect("signal");
        assert_eq!(matches!(signal, PhysicalSignal::HeatRate(_)), watts, "{value}");
    }
}
