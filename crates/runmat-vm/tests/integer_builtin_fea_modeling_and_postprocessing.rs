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

const FEA_SAMPLED_SIGNAL_CONSTRUCTORS: [&str; 22] = [
    "fea.DisplacementSignal.fromTimeSamples",
    "fea.RotationSignal.fromTimeSamples",
    "fea.VelocitySignal.fromTimeSamples",
    "fea.AccelerationSignal.fromTimeSamples",
    "fea.AngularVelocitySignal.fromTimeSamples",
    "fea.AngularAccelerationSignal.fromTimeSamples",
    "fea.ForceSignal.fromTimeSamples",
    "fea.MomentSignal.fromTimeSamples",
    "fea.TractionSignal.fromTimeSamples",
    "fea.PressureSignal.fromTimeSamples",
    "fea.BodyForceSignal.fromTimeSamples",
    "fea.TemperatureSignal.fromTimeSamples",
    "fea.TemperatureDifferenceSignal.fromTimeSamples",
    "fea.HeatFluxSignal.fromTimeSamples",
    "fea.HeatGenerationSignal.fromTimeSamples",
    "fea.ElectricPotentialSignal.fromTimeSamples",
    "fea.ElectricCurrentSignal.fromTimeSamples",
    "fea.ElectricChargeSignal.fromTimeSamples",
    "fea.CurrentDensitySignal.fromTimeSamples",
    "fea.ChargeDensitySignal.fromTimeSamples",
    "fea.MagneticVectorPotentialSignal.fromTimeSamples",
    "fea.MassFlowSignal.fromTimeSamples",
];

const FEA_KNOT_SIGNAL_CONSTRUCTORS: [&str; 24] = [
    "fea.SignalKnot.continuous",
    "fea.SignalKnot.jump",
    "fea.DisplacementSignal.fromTimeKnots",
    "fea.RotationSignal.fromTimeKnots",
    "fea.VelocitySignal.fromTimeKnots",
    "fea.AccelerationSignal.fromTimeKnots",
    "fea.AngularVelocitySignal.fromTimeKnots",
    "fea.AngularAccelerationSignal.fromTimeKnots",
    "fea.ForceSignal.fromTimeKnots",
    "fea.MomentSignal.fromTimeKnots",
    "fea.TractionSignal.fromTimeKnots",
    "fea.PressureSignal.fromTimeKnots",
    "fea.BodyForceSignal.fromTimeKnots",
    "fea.TemperatureSignal.fromTimeKnots",
    "fea.TemperatureDifferenceSignal.fromTimeKnots",
    "fea.HeatFluxSignal.fromTimeKnots",
    "fea.HeatGenerationSignal.fromTimeKnots",
    "fea.ElectricPotentialSignal.fromTimeKnots",
    "fea.ElectricCurrentSignal.fromTimeKnots",
    "fea.ElectricChargeSignal.fromTimeKnots",
    "fea.CurrentDensitySignal.fromTimeKnots",
    "fea.ChargeDensitySignal.fromTimeKnots",
    "fea.MagneticVectorPotentialSignal.fromTimeKnots",
    "fea.MassFlowSignal.fromTimeKnots",
];

const FEA_INITIAL_CONSTRUCTORS: [&str; 10] = [
    "fea.InitialConditionTarget.wholeModel",
    "fea.StructuralInitialCondition.displacement",
    "fea.StructuralInitialCondition.velocity",
    "fea.StructuralInitialCondition.angularVelocity",
    "fea.ThermalInitialCondition.temperature",
    "fea.ElectricalInitialCondition.potential",
    "fea.MagneticInitialCondition.vectorPotential",
    "fea.AcousticInitialCondition.pressure",
    "fea.FluidInitialCondition.pressure",
    "fea.FluidInitialCondition.velocity",
];

const FEA_SPECTRUM_NUMERIC_CONSTRUCTORS: [&str; 9] = [
    "fea.SpectrumAxis.frequencyHertz",
    "fea.SpectrumAxis.periodSeconds",
    "fea.SpectrumDamping.ratio",
    "fea.SpectrumDamping.qualityFactor",
    "fea.SpectrumDirection.new",
    "fea.DirectionalCombination.percentage",
    "fea.ModalTruncation.new",
    "fea.SpectrumEvidence.bounded",
    "fea.ResponseSpectrum.new",
];

const FEA_TYPED_POLICY_CONSTRUCTORS: [&str; 35] = [
    "fea.CoordinateFrame.globalFrame",
    "fea.SignalInterpolation.leftEndpoint",
    "fea.SignalInterpolation.rightEndpoint",
    "fea.SignalInterpolation.linear",
    "fea.SignalExtrapolation.reject",
    "fea.SignalExtrapolation.holdEndpoint",
    "fea.SignalExtrapolation.zero",
    "fea.SignalSampleApproach.at",
    "fea.SignalSampleApproach.leftLimit",
    "fea.SignalSampleApproach.rightLimit",
    "fea.SignalPointContinuity.left",
    "fea.SignalPointContinuity.right",
    "fea.SpectrumResponse.relativeDisplacement",
    "fea.SpectrumResponse.relativeVelocity",
    "fea.SpectrumResponse.relativeAcceleration",
    "fea.SpectrumResponse.absoluteAcceleration",
    "fea.SpectrumResponse.pseudoVelocity",
    "fea.SpectrumResponse.pseudoAcceleration",
    "fea.SpectrumEnvelope.positive",
    "fea.SpectrumEnvelope.negative",
    "fea.SpectrumEnvelope.maximax",
    "fea.SpectrumInterpolation.linearFrequency",
    "fea.SpectrumInterpolation.linearPeriod",
    "fea.SpectrumInterpolation.logLog",
    "fea.SpectrumExtrapolation.reject",
    "fea.SpectrumExtrapolation.holdEndpoint",
    "fea.ModalCombination.srss",
    "fea.ModalCombination.completeQuadratic",
    "fea.ModalCombination.tenPercent",
    "fea.ModalCombination.absoluteSum",
    "fea.DirectionalCombination.srss",
    "fea.DirectionalCombination.absoluteSum",
    "fea.SpectrumConstruction.authoredDesign",
    "fea.SpectrumConstruction.fromAccelerationHistoryExact",
    "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
];

const FEA_EXCITATION_CONSTRUCTORS: [&str; 24] = [
    "fea.StructuralExcitation.force",
    "fea.StructuralExcitation.moment",
    "fea.StructuralExcitation.traction",
    "fea.StructuralExcitation.pressure",
    "fea.StructuralExcitation.bodyForce",
    "fea.StructuralMotion.acceleration",
    "fea.StructuralConstraint.prescribedMotion",
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
            let class_identity =
                runmat_types::ClassIdentity::new(class_name).expect("test class identity");
            assert!(
                values.iter().any(|value| {
                    matches!(value, Value::Object(object) if object.class_name() == &class_identity)
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
        .chain(FEA_SAMPLED_SIGNAL_CONSTRUCTORS)
        .chain(FEA_KNOT_SIGNAL_CONSTRUCTORS.into_iter().take(2))
        .chain(FEA_INITIAL_CONSTRUCTORS.into_iter().skip(1))
        .chain(FEA_SPECTRUM_NUMERIC_CONSTRUCTORS)
    {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("{name} must be a canonical catalog builtin"));
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
        .chain(FEA_KNOT_SIGNAL_CONSTRUCTORS.into_iter().skip(2))
        .chain(FEA_EXCITATION_CONSTRUCTORS)
        .chain(FEA_TYPED_POLICY_CONSTRUCTORS)
        .chain(FEA_INITIAL_CONSTRUCTORS.into_iter().take(1))
    {
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("{name} must be a canonical catalog builtin"));
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
        tipLoad = fea.StructuralExcitation.force("tip", loadPad, forceSignal);
        serviceLoads = fea.ExcitationCase.always("service", {{tipLoad}});
        model = fea.model("bracket", geom, "Steps", {{fea.StructuralStep.static("static")}}, "Signals", {{forceSignal}}, "Excitations", {{tipLoad}}, "ExcitationCases", {{serviceLoads}});
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
        let class_identity =
            runmat_types::ClassIdentity::new(class_name).expect("test class identity");
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Object(object) if object.class_name() == &class_identity)
            ),
            "compiled source must construct {class_name}"
        );
    }
}

#[test]
fn sampled_signal_initial_state_and_response_spectrum_compose_through_compiled_code() {
    let path = std::env::temp_dir().join(format!(
        "runmat-fea-typed-spectrum-{}.stl",
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
        interpolation = fea.SignalInterpolation.linear();
        extrapolation = fea.SignalExtrapolation.zero();
        acceleration = fea.AccelerationSignal.fromTimeSamples("launch.acceleration", [0 0.01 0.02], [0 0 0; 2 0 0; 0 0 0], interpolation, extrapolation);
        displacement = fea.DisplacementSignal.constantMeters("initial.displacement", [0 0 0]);
        target = fea.InitialConditionTarget.wholeModel();
        sampleApproach = fea.SignalSampleApproach.at();
        initial = fea.StructuralInitialCondition.displacement("initial", target, displacement, 0, sampleApproach, "static");
        axis = fea.SpectrumAxis.frequencyHertz([5 10 20]);
        damping = fea.SpectrumDamping.ratio(0.05);
        response = fea.SpectrumResponse.absoluteAcceleration();
        envelope = fea.SpectrumEnvelope.maximax();
        spectrumInterpolation = fea.SpectrumInterpolation.linearFrequency();
        spectrumExtrapolation = fea.SpectrumExtrapolation.reject();
        direction = fea.SpectrumDirection.new(fea.CoordinateFrame.globalFrame(), [1 0 0]);
        modalCombination = fea.ModalCombination.completeQuadratic();
        directionalCombination = fea.DirectionalCombination.srss();
        correction = fea.ModalTruncationCorrection.missingMass(4.9);
        truncation = fea.ModalTruncation.new(0.9, correction);
        rigid = fea.RigidResponse.gupta(8, 33);
        construction = fea.SpectrumConstruction.authoredDesign();
        evidence = fea.SpectrumEvidence.bounded(0.95, 0.01);
        spectrum = fea.ResponseSpectrum.new("launch", axis, [12 18 9], damping, response, envelope, spectrumInterpolation, spectrumExtrapolation, direction, modalCombination, directionalCombination, truncation, rigid, construction, evidence, "sha256:launch-spectrum");
        model = fea.model("launch-model", geom, "Steps", {{fea.StructuralStep.static("static")}}, "Signals", {{acceleration, displacement}}, "InitialConditions", {{initial}}, "ResponseSpectra", {{spectrum}});
        "#,
        path.to_string_lossy()
    );
    let result = execute_source(&source);
    let _ = std::fs::remove_file(&path);
    let values = result.unwrap();
    for class_name in [
        "fea.AccelerationSignal",
        "fea.StructuralInitialCondition",
        "fea.ResponseSpectrum",
        "fea.Model",
    ] {
        let class_identity =
            runmat_types::ClassIdentity::new(class_name).expect("test class identity");
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Object(object) if object.class_name() == &class_identity)
            ),
            "compiled source must construct {class_name}"
        );
    }
}
