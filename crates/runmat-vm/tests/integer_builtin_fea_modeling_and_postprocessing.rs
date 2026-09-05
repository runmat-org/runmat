#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{BuiltinIntegerAuditKind, BuiltinIntegerBackendRule, BuiltinIntegerClass};
use runmat_value::Value;
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
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

#[test]
fn fea_numeric_constructors_accept_every_host_integer_class() {
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            frequency = fea.Frequency.hertz({constructor}(2));
            domain = fea.ElectromagneticDomain.atFrequency(frequency);
            material = fea.material("steel", ...
                "YoungsModulusPa", {constructor}(200), ...
                "PoissonRatio", {constructor}(0));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled FEA constructors: {error}"));
        for class_name in ["fea.ElectromagneticDomain", "fea.Material"] {
            assert!(
                values.iter().any(|value| {
                    matches!(value, Value::Object(object) if object.class_name == class_name)
                }),
                "{constructor} must construct {class_name}"
            );
        }
    }
}

#[test]
fn fea_object_and_text_roles_reject_every_integer_class_without_coercion() {
    for constructor in INTEGER_CONSTRUCTORS {
        for source in [
            format!("value = fea.compare({constructor}(1), \"candidate\");"),
            format!("value = fea.field({constructor}(1), \"stress\");"),
            format!("value = fea.ContactInterface.frictionless({constructor}(1), 2, 3);"),
            format!("value = fea.ContactInterface.coulomb({constructor}(1), 2, 3, 4);"),
            format!("value = fea.FluidStructureInterface.noSlip({constructor}(1), 2, 3);"),
            format!("value = fea.ConjugateHeatTransferInterface.perfect({constructor}(1), 2, 3);"),
            format!("value = fea.ConjugateHeatTransferInterface.conductance({constructor}(1), 2, 3, 4);"),
            format!("value = fea.ConjugateHeatTransferInterface.resistance({constructor}(1), 2, 3, 4);"),
            format!("value = fea.materialAssignment({constructor}(1), \"steel\");"),
            format!("value = fea.model({constructor}(1), {constructor}(2));"),
            format!("value = fea.plan({constructor}(1));"),
        ] {
            let error = execute_source(&source).expect_err("integer object/text role must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:InvalidInput"),
                "{source}"
            );
        }
    }
}

#[test]
fn fea_integer_metadata_declares_host_only_native_contracts() {
    for name in FEA_INTEGER_CONSTRUCTORS
        .into_iter()
        .chain(FEA_SIGNAL_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            !builtin.integer_capabilities.is_empty(),
            "{name} must declare its integer forms"
        );
        assert!(
            builtin.integer_audit.is_none(),
            "{name} capability contract"
        );
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
        for capability in builtin.integer_capabilities {
            assert_eq!(
                capability.backend,
                BuiltinIntegerBackendRule::HostOnly,
                "{name} {} backend",
                capability.form
            );
            assert!(
                !capability.inputs.is_empty(),
                "{name} {} numeric inputs",
                capability.form
            );
            for input in capability.inputs {
                assert_eq!(
                    input.classes, &INTEGER_CLASSES,
                    "{name} {} {} integer classes",
                    capability.form, input.name
                );
            }
        }
    }

    for name in FEA_INTEGER_INAPPLICABLE
        .into_iter()
        .chain(FEA_EXCITATION_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert!(
            builtin.integer_capabilities.is_empty(),
            "{name} has no numeric role"
        );
        let audit = builtin
            .integer_audit
            .unwrap_or_else(|| panic!("{name} must carry an integer audit"));
        assert_eq!(audit.kind, BuiltinIntegerAuditKind::NotApplicable, "{name}");
        assert_eq!(audit.canonical_builtin, None, "{name}");
        assert!(builtin.extensions.is_empty(), "{name} is RunMat-native");
    }
}

#[test]
fn typed_signal_excitation_and_case_compose_through_compiled_runmat_code() {
    let path = std::env::temp_dir().join(format!(
        "runmat-fea-typed-excitation-{}.stl",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "solid part\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid part\n",
    )
    .unwrap();
    let source = format!(
        r#"
        geom = geometry.load("{}", "Units", "meter");
        loadPad = geometry.region(geom, "name:Default Region");
        forceSignal = fea.ForceSignal.constantNewtons("tip.signal", [0 -1250 0]);
        tipLoad = fea.StructuralExcitation.force("tip", "service", loadPad, forceSignal);
        serviceLoads = fea.ExcitationCase.always("service", {{tipLoad}});
        model = fea.model("bracket", geom, "Template", "linear_static_structural", "Signals", {{forceSignal}}, "Excitations", {{tipLoad}}, "ExcitationCases", {{serviceLoads}});
        "#,
        path.to_string_lossy()
    );
    let result = execute_source(&source);
    let _ = std::fs::remove_file(&path);
    let values = result.unwrap();
    for class_name in [
        "fea.ForceSignal",
        "fea.StructuralExcitation",
        "fea.ExcitationCase",
        "fea.Model",
    ] {
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Object(object) if object.class_name == class_name)
            ),
            "compiled source must construct {class_name}"
        );
    }
}
