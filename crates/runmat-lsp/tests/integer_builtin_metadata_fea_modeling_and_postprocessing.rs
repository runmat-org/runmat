use lsp_types::{Position, SignatureHelp};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerClass,
};
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

const FEA_MODELING_AND_POSTPROCESSING_BUILTINS: [&str; 20] = [
    "fea.ConjugateHeatTransferInterface.conductance",
    "fea.ConjugateHeatTransferInterface.perfect",
    "fea.ConjugateHeatTransferInterface.resistance",
    "fea.ContactInterface.coulomb",
    "fea.ContactInterface.frictionless",
    "fea.FluidStructureInterface.noSlip",
    "fea.ThermoMechanicalDomain.atReferenceTemperature",
    "fea.ElectroThermalDomain.atReferenceTemperature",
    "fea.ElectromagneticDomain.atFrequency",
    "fea.FluidDomain.newtonian",
    "fea.Frequency.hertz",
    "fea.MassDensity.kilogramsPerCubicMeter",
    "fea.DynamicViscosity.pascalSeconds",
    "fea.TurbulenceIntensity.ratio",
    "fea.compare",
    "fea.field",
    "fea.material",
    "fea.materialAssignment",
    "fea.model",
    "fea.plan",
];

const FEA_INTEGER_CONSTRUCTORS: [&str; 5] = [
    "fea.Frequency.hertz",
    "fea.MassDensity.kilogramsPerCubicMeter",
    "fea.DynamicViscosity.pascalSeconds",
    "fea.TurbulenceIntensity.ratio",
    "fea.material",
];

const FEA_SIGNAL_CONSTRUCTORS: [&str; 22] = [
    "fea.DisplacementSignal.constantMeters",
    "fea.RotationSignal.constantRadians",
    "fea.VelocitySignal.constantMetersPerSecond",
    "fea.AccelerationSignal.constantMetersPerSecondSquared",
    "fea.AngularVelocitySignal.constantRadiansPerSecond",
    "fea.AngularAccelerationSignal.constantRadiansPerSecondSquared",
    "fea.ForceSignal.constantNewtons",
    "fea.MomentSignal.constantNewtonMeters",
    "fea.TractionSignal.constantPascals",
    "fea.PressureSignal.constantPascals",
    "fea.BodyForceSignal.constantNewtonsPerCubicMeter",
    "fea.TemperatureSignal.constantKelvin",
    "fea.TemperatureDifferenceSignal.constantKelvin",
    "fea.HeatFluxSignal.constantWattsPerSquareMeter",
    "fea.HeatGenerationSignal.constantWattsPerCubicMeter",
    "fea.ElectricPotentialSignal.constantVolts",
    "fea.ElectricCurrentSignal.constantAmperes",
    "fea.ElectricChargeSignal.constantCoulombs",
    "fea.CurrentDensitySignal.constantAmperesPerSquareMeter",
    "fea.ChargeDensitySignal.constantCoulombsPerCubicMeter",
    "fea.MagneticVectorPotentialSignal.constantWebersPerMeter",
    "fea.MassFlowSignal.constantKilogramsPerSecond",
];

const FEA_EXCITATION_CONSTRUCTORS: [&str; 25] = [
    "fea.StructuralExcitation.force",
    "fea.StructuralExcitation.moment",
    "fea.StructuralExcitation.traction",
    "fea.StructuralExcitation.pressure",
    "fea.StructuralExcitation.bodyForce",
    "fea.StructuralExcitation.baseDisplacement",
    "fea.StructuralExcitation.baseVelocity",
    "fea.StructuralExcitation.baseAcceleration",
    "fea.ThermalExcitation.heatGeneration",
    "fea.ThermalExcitation.heatFlux",
    "fea.ThermalExcitation.temperatureField",
    "fea.ThermalExcitation.ambientTemperature",
    "fea.ElectricalExcitation.terminalVoltage",
    "fea.ElectricalExcitation.current",
    "fea.ElectricalExcitation.charge",
    "fea.ElectricalExcitation.currentDensity",
    "fea.MagneticExcitation.vectorPotential",
    "fea.AcousticExcitation.pressure",
    "fea.AcousticExcitation.normalVelocity",
    "fea.FluidExcitation.inletVelocity",
    "fea.FluidExcitation.massFlow",
    "fea.FluidExcitation.wallMotion",
    "fea.ExcitationCase.always",
    "fea.ExcitationCase.duringSteps",
    "fea.ExcitationCase.mutuallyExclusive",
];

const FEA_INTEGER_INAPPLICABLE: [&str; 15] = [
    "fea.ConjugateHeatTransferInterface.conductance",
    "fea.ConjugateHeatTransferInterface.perfect",
    "fea.ConjugateHeatTransferInterface.resistance",
    "fea.ContactInterface.coulomb",
    "fea.ContactInterface.frictionless",
    "fea.FluidStructureInterface.noSlip",
    "fea.ThermoMechanicalDomain.atReferenceTemperature",
    "fea.ElectroThermalDomain.atReferenceTemperature",
    "fea.ElectromagneticDomain.atFrequency",
    "fea.FluidDomain.newtonian",
    "fea.compare",
    "fea.field",
    "fea.materialAssignment",
    "fea.model",
    "fea.plan",
];

const INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

fn signature_help(source: &str, lookup: &str) -> SignatureHelp {
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    assert!(analysis.syntax_error.is_none(), "{source}");
    assert!(
        analysis.lowering_error.is_none(),
        "{source}: {:?}",
        analysis.lowering_error
    );
    assert!(
        analysis.compile_error.is_none(),
        "{source}: {:?}",
        analysis.compile_error
    );
    let column = source.find(lookup).expect("builtin call") as u32;
    signature_help_at(source, &analysis, &Position::new(0, column))
        .unwrap_or_else(|| panic!("descriptor-backed signature help for {lookup}"))
}

#[test]
fn fea_modeling_and_postprocessing_descriptors_are_visible_to_signature_help() {
    for (lookup, source) in [
        (
            "ConjugateHeatTransferInterface.perfect",
            "out = fea.ConjugateHeatTransferInterface.perfect(\"cht\", 0, 0);",
        ),
        (
            "ConjugateHeatTransferInterface.conductance",
            "out = fea.ConjugateHeatTransferInterface.conductance(\"cht\", 0, 0, 0);",
        ),
        (
            "ConjugateHeatTransferInterface.resistance",
            "out = fea.ConjugateHeatTransferInterface.resistance(\"cht\", 0, 0, 0);",
        ),
        (
            "ContactInterface.frictionless",
            "out = fea.ContactInterface.frictionless(\"contact\", 0, 0);",
        ),
        (
            "ContactInterface.coulomb",
            "out = fea.ContactInterface.coulomb(\"contact\", 0, 0, 0);",
        ),
        (
            "FluidStructureInterface.noSlip",
            "out = fea.FluidStructureInterface.noSlip(\"fsi\", 0, 0);",
        ),
        ("compare", "out = fea.compare(\"base\", \"candidate\");"),
        (
            "ThermoMechanicalDomain.atReferenceTemperature",
            "out = fea.ThermoMechanicalDomain.atReferenceTemperature(fea.Temperature.kelvin(293));",
        ),
        (
            "ElectroThermalDomain.atReferenceTemperature",
            "out = fea.ElectroThermalDomain.atReferenceTemperature(fea.Temperature.kelvin(293));",
        ),
        (
            "ElectromagneticDomain.atFrequency",
            "out = fea.ElectromagneticDomain.atFrequency(fea.Frequency.hertz(60));",
        ),
        (
            "FluidDomain.newtonian",
            "out = fea.FluidDomain.newtonian(fea.MassDensity.kilogramsPerCubicMeter(1), fea.DynamicViscosity.pascalSeconds(1), fea.TurbulenceIntensity.ratio(0));",
        ),
        ("Frequency.hertz", "out = fea.Frequency.hertz(60);"),
        (
            "MassDensity.kilogramsPerCubicMeter",
            "out = fea.MassDensity.kilogramsPerCubicMeter(1);",
        ),
        (
            "DynamicViscosity.pascalSeconds",
            "out = fea.DynamicViscosity.pascalSeconds(1);",
        ),
        (
            "TurbulenceIntensity.ratio",
            "out = fea.TurbulenceIntensity.ratio(0);",
        ),
        ("field", "out = fea.field(\"run\", \"stress\");"),
        (
            "material",
            "out = fea.material(\"steel\", \"YoungsModulusPa\", 200, \"PoissonRatio\", 0);",
        ),
        (
            "materialAssignment",
            "out = fea.materialAssignment(\"region\", \"steel\");",
        ),
        ("model", "out = fea.model(\"model\", \"geometry\");"),
        ("plan", "out = fea.plan(\"study.fea\");"),
    ] {
        let help = signature_help(source, lookup);
        assert!(!help.signatures.is_empty(), "{lookup} signatures");
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label.contains(&format!("fea.{lookup}"))),
            "{lookup} must expose its namespaced signature"
        );
    }
}

#[test]
fn fea_packet_is_public_and_has_settled_integer_metadata() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in FEA_MODELING_AND_POSTPROCESSING_BUILTINS
        .into_iter()
        .chain(FEA_SIGNAL_CONSTRUCTORS)
        .chain(FEA_EXCITATION_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        let descriptor = builtin
            .descriptor
            .unwrap_or_else(|| panic!("{name} descriptor"));
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        assert!(
            !descriptor.signatures.is_empty(),
            "{name} reference-visible signatures"
        );
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "public completion for {name}"
        );
    }
}

#[test]
fn fea_numeric_constructor_forms_cover_every_integer_class_on_the_host() {
    for name in FEA_INTEGER_CONSTRUCTORS
        .into_iter()
        .chain(FEA_SIGNAL_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_audit.is_none(), "{name}");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} integer forms"
        );
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::HostOnly,
                "{name} {}",
                capability.form
            );
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{name} {} {}",
                    capability.form, input.name
                );
            }
        }
    }
}

#[test]
fn fea_non_numeric_roles_have_explicit_not_applicable_audits() {
    for name in FEA_INTEGER_INAPPLICABLE
        .into_iter()
        .chain(FEA_EXCITATION_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(builtin.integer_capabilities.is_empty(), "{name}");
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(!audit.notes.is_empty(), "{name} audit rationale");
    }
}
