use super::*;
use crate::builtins::fea::damping::mechanical_damping_rayleigh_builtin;
use crate::builtins::fea::values::{
    dynamic_viscosity_builtin, frequency_builtin, mass_density_builtin,
    rayleigh_mass_coefficient_builtin, rayleigh_stiffness_coefficient_builtin,
    temperature_kelvin_builtin, turbulence_intensity_builtin,
};
use futures::executor::block_on;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_analysis_fea::ComputeBackend;
use runmat_meshing_core::CanonicalMeshingContract;
use runmat_types::MemberAccess;
use runmat_value::CellArray;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";
const SIMPLE_STEP: &str = "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n";

fn test_run_identity(
    procedure: runmat_analysis_core::AnalysisProcedure,
) -> runmat_analysis_fea::AnalysisRunIdentity {
    let assembly = runmat_analysis_core::StructuralFormulationRealization::new(
        runmat_analysis_core::StructuralOperatorImplementation::Tet4LinearElasticity,
        Some(runmat_analysis_core::StructuralRecoveryImplementation::Tet4ConstantStrain),
    );
    let realization = runmat_analysis_fea::admission::structural_exact_realization(
        runmat_analysis_fea::admission::StructuralRealizationInput {
            assemblies: vec![assembly],
            procedure,
            kinematics: runmat_analysis_core::StructuralKinematics::SmallStrain,
            coupling: runmat_analysis_core::CouplingImplementation::Uncoupled,
            sources: vec![runmat_analysis_core::SourceImplementation::NotApplicable],
            boundary_transfers: vec![
                runmat_analysis_core::BoundaryTransferImplementation::NotApplicable,
            ],
        },
    )
    .expect("test structural formulation must be valid");
    runmat_analysis_fea::admission::admit_formulation(
        runmat_analysis_fea::admission::FormulationAdmissionRequest {
            family: runmat_analysis_core::PhysicsFamily::Structural,
            procedure,
            solution_space: runmat_analysis_core::SolutionSpace::Direct,
            realization,
        },
    )
    .expect("test structural formulation must be admitted")
    .into_identity()
}

fn synthetic_connector_result(
) -> Box<runmat_analysis_fea::connectors::CanonicalStructuralConnectorResult> {
    let topology = runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity::Solid {
        mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
    };
    let plan =
        runmat_analysis_fea::connectors::CanonicalStructuralConnectorPlan::empty(topology, 0)
            .expect("empty synthetic connector plan");
    Box::new(
        runmat_analysis_fea::connectors::CanonicalStructuralConnectorResult::recover_static(
            plan,
            &runmat_analysis_fea::assembly::dofs::StructuralDofLayout::default(),
            &[],
        )
        .expect("zero synthetic connector state"),
    )
}

fn synthetic_static_balance() -> Box<runmat_analysis_fea::solve::StructuralStaticBalanceEvidence> {
    Box::new(
        runmat_analysis_fea::solve::StructuralStaticBalanceEvidence {
            internal_work_joules: 0.0,
            applied_load_work_joules: 0.0,
            constraint_work_joules: 0.0,
            work_balance_residual_joules: 0.0,
            work_balance_relative_error: 0.0,
            equilibrium_residual_norm_newtons: 0.0,
            equilibrium_scale_newtons: 1.0,
            equilibrium_relative_error: 0.0,
            relative_tolerance: 1.0e-8,
            accepted: true,
        },
    )
}

fn synthetic_constraint_result(
) -> Box<runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult> {
    Box::new(
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult {
            schema_version: 1,
            constraint_plan_digest: runmat_execution::Digest::sha256(b"synthetic-constraint-plan"),
            topology:
                runmat_analysis_fea::constraints::CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: runmat_meshing_core::StableDigest::from_bytes([7; 32]),
                },
            reference_points: Vec::new(),
            groups: Vec::new(),
            resultant_force_newtons: [0.0; 3],
            resultant_moment_about_global_origin_newton_meters: [0.0; 3],
            constraint_work_joules: 0.0,
            maximum_displacement_error_meters: 0.0,
            maximum_rotation_error_radians: 0.0,
        },
    )
}

fn cell(values: Vec<Value>) -> Value {
    let shape = if values.is_empty() {
        (0, 0)
    } else {
        (1, values.len())
    };
    Value::Cell(CellArray::new(values, shape.0, shape.1).expect("cell should build"))
}

fn object_payload<T: DeserializeOwned>(value: &Value) -> T {
    let json = value_to_json("fea.test", value).expect("typed FEA value should encode");
    serde_json::from_value(json).expect("typed FEA value should decode")
}

#[test]
fn fea_usize_parsers_preserve_typed_bounds_and_reject_invalid_values() {
    use runmat_value::{IntValue, IntegerStorage};

    assert_eq!(
        usize_from_value("fea.test", &Value::Int(IntValue::U16(7))).unwrap(),
        7
    );
    assert!(usize_from_value("fea.test", &Value::Int(IntValue::I8(-1))).is_err());
    assert!(usize_from_value("fea.test", &Value::Num(1.5)).is_err());
    assert!(usize_vec_from_value(
        "fea.test",
        &Value::Tensor(Tensor::new_2d(vec![1.0, -1.0], 1, 2).unwrap())
    )
    .is_err());
    let typed_indices = Tensor::new_integer(IntegerStorage::U16(vec![2, 4]), vec![1, 2]).unwrap();
    assert_eq!(
        usize_vec_from_value("fea.test", &Value::Tensor(typed_indices)).unwrap(),
        vec![2, 4]
    );

    let bound = runmat_types::wire_integer::MAX_WIRE_INTEGER;
    let maximum = usize_from_value("fea.test", &Value::Int(IntValue::U64(bound)));
    if usize::BITS == 64 {
        assert_eq!(maximum.unwrap(), bound as usize);
    } else {
        assert!(maximum.is_err());
    }
    assert!(usize_from_value("fea.test", &Value::Int(IntValue::U64(bound + 1))).is_err());
    assert!(usize_from_value("fea.test", &Value::Int(IntValue::U64(u64::MAX))).is_err());
}

#[test]
fn fea_public_object_mirror_preserves_recursive_integer_kinds() {
    #[derive(Serialize)]
    struct IntegerMirror {
        signed: i64,
        unsigned: u64,
        matrix: Vec<Vec<u64>>,
        floating: f64,
        floating_vector: Vec<f64>,
    }

    let object = serializable_to_object_value(
        RESULTS_NAME,
        &ERROR_INTERNAL,
        FEA_RESULTS_CLASS,
        &IntegerMirror {
            signed: -9,
            unsigned: u64::MAX,
            matrix: vec![vec![1, 2], vec![3, 4]],
            floating: 1.0,
            floating_vector: vec![2.0, 3.0],
        },
    )
    .expect("integer-preserving public mirror");
    assert!(matches!(
        object.properties().get("signed"),
        Some(Value::Int(IntValue::I64(-9)))
    ));
    assert!(matches!(
        object.properties().get("unsigned"),
        Some(Value::Int(IntValue::U64(u64::MAX)))
    ));
    let Some(Value::Tensor(matrix)) = object.properties().get("matrix") else {
        panic!("exact integer matrix");
    };
    assert_eq!(matrix.shape(), vec![2, 2]);
    assert_eq!(
        matrix
            .integer_storage()
            .expect("integer storage")
            .exact_values(),
        vec![
            IntValue::U64(1),
            IntValue::U64(3),
            IntValue::U64(2),
            IntValue::U64(4),
        ]
    );
    assert!(matches!(
        object.properties().get("floating"),
        Some(Value::Num(1.0))
    ));
    assert!(matches!(
        object.properties().get("floating_vector"),
        Some(Value::Tensor(values)) if values.integer_storage().is_none()
    ));

    #[derive(Serialize)]
    struct EmptyIntegerMirror {
        available_mode_indices: Vec<usize>,
        iteration_counts: Vec<usize>,
    }
    let empty = serializable_to_object_preserving_integers(
        RESULTS_NAME,
        &ERROR_INTERNAL,
        FEA_RESULTS_CLASS,
        &EmptyIntegerMirror {
            available_mode_indices: Vec::new(),
            iteration_counts: Vec::new(),
        },
        &[],
        &["available_mode_indices", "iteration_counts"],
    )
    .expect("schema-aware empty integer vectors");
    let Value::Object(empty) = empty else {
        panic!("results object");
    };
    for name in ["available_mode_indices", "iteration_counts"] {
        let Some(Value::Tensor(values)) = empty.properties().get(name) else {
            panic!("empty exact vector {name}");
        };
        assert!(values.integer_storage().is_some());
        assert!(values.is_empty());
    }
}

#[test]
fn fea_execution_controls_enforce_exact_structural_boundaries() {
    let empty_step = block_on(structural_modal_builtin(Value::String(
        String::new().into(),
    )))
    .expect_err("step identity must not be empty");
    assert_eq!(empty_step.identifier(), Some("RunMat:fea:InvalidInput"));

    let zero_window = block_on(fea_trends_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("WindowSize".into()),
            Value::Int(IntValue::U8(0)),
        ]),
    ))
    .expect_err("trend window must be positive");
    assert_eq!(zero_window.identifier(), Some("RunMat:fea:InvalidInput"));

    assert!(exact_bool_from_value(
        RESULTS_NAME,
        &Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("double flag"))
    )
    .unwrap());
}

#[test]
fn fea_result_selectors_are_exact_one_based_vectors_and_flags_are_zero_one() {
    let wide = runmat_types::wire_integer::MAX_WIRE_INTEGER;
    let selectors = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
        .expect("wide selector vector");
    let decoded = one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(selectors));
    if usize::BITS == 64 {
        assert_eq!(decoded.unwrap(), vec![(wide - 1) as usize]);
    } else {
        assert_eq!(
            decoded.unwrap_err().identifier(),
            Some("RunMat:fea:InvalidInput")
        );
    }
    let beyond = Tensor::new_integer(IntegerStorage::U64(vec![wide + 2]), vec![1, 1])
        .expect("selector beyond the wire bound");
    assert_eq!(
        one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(beyond))
            .unwrap_err()
            .identifier(),
        Some("RunMat:fea:InvalidInput")
    );

    let selectors = Tensor::new_integer(IntegerStorage::U32(vec![3, 1, 3]), vec![3, 1])
        .expect("selector vector");
    assert_eq!(
        one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(selectors)).unwrap(),
        vec![2, 0, 2]
    );
    let matrix = Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2])
        .expect("selector matrix");
    assert!(one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(matrix)).is_err());
    assert!(!exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::I8(0))).unwrap());
    assert!(exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::U64(1))).unwrap());
    assert!(exact_bool_from_value(RESULTS_NAME, &Value::Int(IntValue::I16(2))).is_err());

    let single_selectors =
        Tensor::new_with_dtype(vec![1.0, 2.0], vec![1, 2], runmat_value::NumericDType::F32)
            .expect("single selectors");
    assert!(
        one_based_usize_vec_from_value(RESULTS_NAME, &Value::Tensor(single_selectors)).is_err()
    );
}

#[test]
fn fea_sweep_failures_cross_to_one_based_public_indices() {
    let mut entries = vec![AnalysisStudySweepFailureEntry {
        study_id: "bad-study".into(),
        study_index: 0,
        kind: crate::analysis::AnalysisStudySweepFailureKind::Failed,
        error_code: "RM.TEST".into(),
        message: "failed".into(),
    }];
    one_base_failure_entries(RUN_NAME, &mut entries).expect("public index translation");
    assert_eq!(entries[0].study_index, 1);

    let mut context = std::collections::BTreeMap::new();
    context.insert("study_index".into(), "0".into());
    let error = public_sweep_error(OperationErrorEnvelope {
        error_code: "RM.TEST".into(),
        error_type: crate::operations::OperationErrorType::Validation,
        message: "study sweep failed at index 0 for study_id bad-study".into(),
        operation: "test".into(),
        op_version: "1".into(),
        retryable: false,
        severity: crate::operations::OperationErrorSeverity::Error,
        context,
        trace_id: None,
        request_id: None,
        timestamp: "test".into(),
    });
    assert_eq!(
        error.context.get("study_index").map(String::as_str),
        Some("1")
    );
    assert!(error.message.contains("at index 1 "));
}

#[test]
fn sweep_operation_classes_are_exact_and_do_not_inherit_single_run_methods() {
    ensure_fea_classes_registered();

    let validation = serializable_to_object(
        VALIDATE_NAME,
        &ERROR_INTERNAL,
        FEA_SWEEP_VALIDATION_CLASS,
        &AnalysisStudySweepValidateData {
            sweep_id: "sweep".into(),
            valid: true,
            issue_codes: Vec::new(),
            study_entries: Vec::new(),
            evidence_artifact_path: "validation.json".into(),
        },
    )
    .expect("sweep validation object");
    let plan = serializable_to_object_preserving_integers(
        PLAN_NAME,
        &ERROR_INTERNAL,
        FEA_SWEEP_PLAN_CLASS,
        &AnalysisStudySweepPlanData {
            sweep_id: "sweep".into(),
            study_count: 0,
            planned_count: 0,
            failed_count: 0,
            failure_entries: Vec::new(),
            plan_entries: Vec::new(),
            evidence_artifact_path: "plan.json".into(),
        },
        &[],
        &[
            "study_count",
            "planned_count",
            "failed_count",
            "study_index",
        ],
    )
    .expect("sweep plan object");
    let run = serializable_to_object_preserving_integers(
        RUN_NAME,
        &ERROR_INTERNAL,
        FEA_SWEEP_RUN_RESULT_CLASS,
        &AnalysisStudySweepData {
            sweep_id: "sweep".into(),
            identity: runmat_analysis_fea::execution::SweepIdentity::new(
                runmat_execution::Digest::sha256(b"sweep"),
                vec![
                    runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
                        runmat_execution::Digest::sha256(b"study"),
                        None,
                    )
                    .expect("study identity"),
                ],
                runmat_analysis_fea::execution::SweepCompositionPolicy::CollectAll,
            )
            .expect("sweep identity"),
            study_count: 0,
            success_count: 0,
            failed_count: 0,
            failure_entries: Vec::new(),
            run_entries: Vec::new(),
            evidence_artifact_path: "run.json".into(),
        },
        &[],
        &[
            "study_count",
            "success_count",
            "failed_count",
            "study_index",
        ],
    )
    .expect("sweep run object");

    for (value, expected_class) in [
        (validation, FEA_SWEEP_VALIDATION_CLASS),
        (plan, FEA_SWEEP_PLAN_CLASS),
        (run, FEA_SWEEP_RUN_RESULT_CLASS),
    ] {
        assert_object_class(&value, expected_class);
        let payload = crate::execution::value_codec::encode_inline_value(&value)
            .expect("sweep value should be portable");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&payload)
                .expect("sweep value should decode"),
            value
        );
    }

    for class_name in [FEA_VALIDATION_CLASS, FEA_PLAN_CLASS, FEA_RUN_RESULT_CLASS] {
        assert!(crate::class_registry::lookup_property(
            &runmat_types::ClassIdentity::from(class_name),
            &runmat_types::MemberName::from("sweep_id")
        )
        .is_none());
    }
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
        &runmat_types::MethodName::from("results")
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
        &runmat_types::MethodName::from("step"),
    )
    .is_some());
    assert!(crate::class_registry::lookup_property(
        &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
        &runmat_types::MemberName::from("steps")
    )
    .is_some());
    assert!(crate::class_registry::lookup_property(
        &runmat_types::ClassIdentity::from(FEA_RUN_RESULT_CLASS),
        &runmat_types::MemberName::from("run_id")
    )
    .is_none());
    assert!(crate::class_registry::lookup_property(
        &runmat_types::ClassIdentity::from(FEA_STEP_RUN_RESULT_CLASS),
        &runmat_types::MemberName::from("run_id")
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_STEP_RUN_RESULT_CLASS),
        &runmat_types::MethodName::from("results")
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_SWEEP_RUN_RESULT_CLASS),
        &runmat_types::MethodName::from("results")
    )
    .is_none());
}

#[test]
fn fea_json_preserves_native_integer_scalars_and_tensors() {
    let maximum = runmat_value::IntValue::U64(u64::MAX);
    assert_eq!(
        value_to_json("fea.test", &Value::Int(maximum.clone()))
            .expect("scalar json")
            .to_string(),
        maximum.decimal_string()
    );

    let scalar = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX]),
        vec![1, 1],
    )
    .expect("scalar tensor");
    assert_eq!(
        value_to_json("fea.test", &Value::Tensor(scalar))
            .expect("scalar tensor json")
            .to_string(),
        u64::MAX.to_string()
    );

    let tensor = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![42, u64::MAX]),
        vec![1, 2],
    )
    .expect("tensor");
    assert_eq!(
        value_to_json("fea.test", &Value::Tensor(tensor))
            .expect("tensor json")
            .to_string(),
        "[42,18446744073709551615]"
    );

    let column = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![42, u64::MAX]),
        vec![2, 1],
    )
    .expect("column tensor");
    assert_eq!(
        value_to_json("fea.test", &Value::Tensor(column))
            .expect("column tensor json")
            .to_string(),
        "[[42],[18446744073709551615]]"
    );
}

#[test]
fn fea_numeric_constructors_cross_all_integer_classes_once_into_binary64() {
    for integer in [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(u64::MAX),
    ] {
        let expected = integer.to_f64();
        let frequency = block_on(frequency_builtin(Value::Int(integer.clone())))
            .expect("frequency integer field");
        let domain = block_on(domains::electromagnetic_domain_builtin(frequency))
            .expect("typed electromagnetic domain");
        let domain: runmat_analysis_core::ElectromagneticDomain = object_payload(&domain);
        assert_eq!(
            domain,
            runmat_analysis_core::ElectromagneticDomain::Harmonic {
                reference_frequency: runmat_analysis_core::Frequency::new(expected),
            }
        );

        let material = block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("material".into()),
                Value::String("YoungsModulusPa".into()),
                Value::Int(integer),
                Value::String("PoissonRatio".into()),
                Value::Int(IntValue::U8(0)),
            ]),
        ))
        .expect("material integer field");
        let material: MaterialModel = object_payload(&material);
        assert_eq!(
            material
                .require_mechanical()
                .expect("authored mechanical capability")
                .youngs_modulus_pa
                .canonical_value(),
            expected
        );
    }
}

#[test]
fn fea_material_preserves_typed_rayleigh_damping() {
    let mass =
        block_on(rayleigh_mass_coefficient_builtin(Value::Num(0.2))).expect("mass coefficient");
    let stiffness = block_on(rayleigh_stiffness_coefficient_builtin(Value::Num(0.003)))
        .expect("stiffness coefficient");
    let damping =
        block_on(mechanical_damping_rayleigh_builtin(mass, stiffness)).expect("Rayleigh damping");
    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(210.0e9),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
            Value::String("Damping".into()),
            damping,
        ]),
    ))
    .expect("material");
    let material: MaterialModel = object_payload(&material);
    assert_eq!(
        material.damping,
        Some(MechanicalDampingModel::Rayleigh {
            mass_coefficient: runmat_analysis_core::RayleighMassCoefficient::new(0.2),
            stiffness_coefficient: runmat_analysis_core::RayleighStiffnessCoefficient::new(0.003),
        })
    );

    let error = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(210.0e9),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
            Value::String("Damping".into()),
            Value::String("rayleigh".into()),
        ]),
    ))
    .expect_err("untyped damping must be rejected");
    assert!(error.message().contains("typed fea.MechanicalDamping"));
}

#[test]
fn fea_material_requires_explicit_capabilities_and_canonical_density_name() {
    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("air".into()),
            Value::String("MassDensityKgPerM3".into()),
            Value::Num(1.225),
        ]),
    ))
    .expect("a density-only material is valid for compatible formulations");
    let material: MaterialModel = object_payload(&material);
    assert_eq!(
        material
            .mass_density_kg_per_m3
            .map(|quantity| quantity.canonical_value()),
        Some(1.225)
    );
    assert!(material.elasticity.is_none());

    let empty = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "empty".into(),
        )]),
    ))
    .expect_err("an empty material must fail at construction");
    assert!(empty.message().contains("constitutive capability"));

    let retired = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("air".into()),
            Value::String("DensityKgPerM3".into()),
            Value::Num(1.225),
        ]),
    ))
    .expect_err("the ambiguous retired density spelling must not be accepted");
    assert!(retired.message().contains("unsupported option field"));
    assert!(retired.message().contains("density_kg_per_m3"));
}

#[test]
fn fea_domain_constructors_preserve_typed_physics_boundaries() {
    let temperature =
        block_on(temperature_kelvin_builtin(Value::Num(293.15))).expect("typed temperature");
    let frequency = block_on(frequency_builtin(Value::Num(60.0))).expect("typed frequency");
    let density = block_on(mass_density_builtin(Value::Num(1.225))).expect("typed density");
    let viscosity =
        block_on(dynamic_viscosity_builtin(Value::Num(1.81e-5))).expect("typed viscosity");
    let turbulence = block_on(turbulence_intensity_builtin(Value::Num(0.05)))
        .expect("typed turbulence intensity");

    let thermal =
        block_on(domains::thermal_domain_builtin(temperature.clone())).expect("thermal domain");
    let thermo = block_on(domains::thermomechanical_domain_builtin(
        temperature.clone(),
    ))
    .expect("thermo-mechanical domain");
    let electro = block_on(domains::electrothermal_domain_builtin(temperature))
        .expect("electro-thermal domain");
    let electromagnetic = block_on(domains::electromagnetic_domain_builtin(frequency.clone()))
        .expect("electromagnetic domain");
    let fluid = block_on(domains::fluid_domain_builtin(
        density, viscosity, turbulence,
    ))
    .expect("fluid domain");

    for (value, class_name) in [
        (thermal, FEA_THERMAL_DOMAIN_CLASS),
        (thermo, FEA_THERMOMECHANICAL_DOMAIN_CLASS),
        (electro, FEA_ELECTROTHERMAL_DOMAIN_CLASS),
        (electromagnetic, FEA_ELECTROMAGNETIC_DOMAIN_CLASS),
        (fluid, FEA_FLUID_DOMAIN_CLASS),
    ] {
        assert_object_class(&value, class_name);
    }

    let error = block_on(domains::thermomechanical_domain_builtin(frequency))
        .expect_err("frequency must not cross the temperature boundary");
    assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
}

#[test]
fn fea_domain_classes_expose_only_their_own_constructors() {
    ensure_fea_classes_registered();
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_THERMAL_DOMAIN_CLASS),
        &runmat_types::MethodName::from("atReferenceTemperature")
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_THERMOMECHANICAL_DOMAIN_CLASS),
        &runmat_types::MethodName::from("atReferenceTemperature")
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_THERMOMECHANICAL_DOMAIN_CLASS),
        &runmat_types::MethodName::from("atFrequency")
    )
    .is_none());
    assert!(crate::class_registry::lookup_method(
        &runmat_types::ClassIdentity::from(FEA_FLUID_DOMAIN_CLASS),
        &runmat_types::MethodName::from("newtonian")
    )
    .is_some());
    assert!(runmat_builtins::builtin_function_by_name("fea.domain").is_none());
}

#[test]
fn fea_model_rejects_duplicate_domain_families() {
    let temperature =
        block_on(temperature_kelvin_builtin(Value::Num(293.15))).expect("typed temperature");
    let first = block_on(domains::thermomechanical_domain_builtin(
        temperature.clone(),
    ))
    .expect("first domain");
    let second =
        block_on(domains::thermomechanical_domain_builtin(temperature)).expect("second domain");
    let error = modeling::domain_vec_from_value("fea.model", &cell(vec![first, second]))
        .expect_err("duplicate family must reject");
    assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(error
        .message()
        .contains("duplicate thermo-mechanical FEA domain"));
}

#[test]
fn vector_field_values_have_one_row_per_node_and_one_column_per_component() {
    // Three nodes with displacement u_i = (i, 10 i, 100 i), stored node-major as the
    // analysis producers publish them.
    let field = AnalysisField {
        field_id: "structural.displacement".into(),
        layout: runmat_analysis_core::AnalysisFieldLayout::vector3(
            runmat_analysis_core::ObservationQuantity::Displacement,
            runmat_analysis_core::FieldLocation::Node,
        ),
        shape: vec![3, 3],
        values: AnalysisFieldValues::HostF64(vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0,
        ]),
    };
    let descriptor = AnalysisFieldDescriptor::from_field(&field);
    let object = field_to_object(&field, &descriptor).expect("field object");
    let Some(Value::Tensor(values)) = object.properties().get("values") else {
        panic!("expected values tensor");
    };
    assert_eq!(values.shape(), vec![3, 3]);
    let column_major = values.materialize_f64();
    let column = |k: usize| column_major[3 * k..3 * (k + 1)].to_vec();
    assert_eq!(column(0), vec![1.0, 2.0, 3.0], "u(:,1) is the x component");
    assert_eq!(
        column(1),
        vec![10.0, 20.0, 30.0],
        "u(:,2) is the y component"
    );
    assert_eq!(
        column(2),
        vec![100.0, 200.0, 300.0],
        "u(:,3) is the z component"
    );
    let row = |i: usize| [0, 1, 2].map(|k| column_major[i + 3 * k]);
    assert_eq!(row(1), [2.0, 20.0, 200.0], "u(2,:) is node 2's vector");

    // A rank-3 tensor field keeps the same rule on every axis.
    let tensor = AnalysisField {
        field_id: "structural.stress_tensor".into(),
        layout: runmat_analysis_core::AnalysisFieldLayout::components(
            runmat_analysis_core::ObservationQuantity::Stress,
            runmat_analysis_core::FieldLocation::Element,
            4,
        ),
        shape: vec![2, 2, 2],
        values: AnalysisFieldValues::HostF64((0..8).map(f64::from).collect()),
    };
    let Value::Tensor(values) = field_values_value(&tensor).expect("tensor values") else {
        panic!("expected values tensor");
    };
    // Row-major entry (i, j, k) = 4 i + 2 j + k sits at column-major i + 2 j + 4 k.
    assert_eq!(
        values.materialize_f64(),
        vec![0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]
    );
}

#[test]
fn fea_field_metadata_remains_exact_in_public_objects() {
    let field = AnalysisField {
        field_id: "stress".into(),
        layout: runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::Stress,
            runmat_analysis_core::FieldLocation::Element,
        ),
        shape: vec![u32::MAX as usize, 0],
        values: AnalysisFieldValues::HostF64(Vec::new()),
    };
    let descriptor = AnalysisFieldDescriptor::from_field(&field);
    let object = field_to_object(&field, &descriptor).expect("field object");
    let Value::Tensor(shape) = object.properties().get("shape").expect("shape") else {
        panic!("expected integer shape tensor");
    };
    assert_eq!(shape.numeric_dtype(), runmat_value::NumericDType::U64);
    assert_eq!(
        shape.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::from(u32::MAX), 0]))
    );
    assert_eq!(
        object.properties().get("element_count"),
        Some(&Value::Int(IntValue::U64(0)))
    );
    let reference = field_object(&descriptor, None).expect("artifact field reference");
    assert_eq!(
        reference.properties().get("materialized"),
        Some(&Value::Bool(false))
    );
    assert!(!reference.properties().contains_key("values"));

    let device_field = AnalysisField {
        field_id: "device".into(),
        layout: runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::Residual,
            runmat_analysis_core::FieldLocation::Global,
        ),
        shape: vec![u32::MAX as usize],
        values: AnalysisFieldValues::DeviceRef(runmat_analysis_core::DeviceFieldRef {
            backend: "wgpu".into(),
            token: "buffer".into(),
            element_count: u32::MAX as usize,
        }),
    };
    let Value::Struct(device) = field_values_value(&device_field).expect("device metadata") else {
        panic!("expected device field metadata");
    };
    assert_eq!(
        device.fields.get("element_count"),
        Some(&Value::Int(IntValue::U64(u64::from(u32::MAX))))
    );
}

#[test]
fn structural_field_selectors_are_typed_and_portable() {
    ensure_value_classes_registered();
    let selector = block_on(field_selectors::stress()).expect("typed structural field");
    let Value::Object(object) = &selector else {
        panic!("expected structural field selector object");
    };
    assert_eq!(
        object.class_name().display_name(),
        field_selectors::STRUCTURAL_FIELD_CLASS
    );
    assert_eq!(
        field_selectors::from_value("test", &selector).unwrap(),
        field_selectors::FieldSelector::Structural(
            runmat_analysis_fea::contracts::StructuralFieldSelector::Stress
        )
    );
    let encoded = crate::execution::value_codec::encode_inline_value(&selector)
        .expect("structural field selector should be portable");
    let decoded = crate::execution::value_codec::decode_inline_value(&encoded)
        .expect("portable structural field selector should decode");
    assert_eq!(selector, decoded);
    assert!(field_selectors::from_value("test", &Value::String("stress".into())).is_err());

    let integration_point = block_on(field_selectors::integration_point_stress())
        .expect("typed integration-point field");
    assert_eq!(
        field_selectors::from_value("test", &integration_point).unwrap(),
        field_selectors::FieldSelector::Structural(
            runmat_analysis_fea::contracts::StructuralFieldSelector::IntegrationPointStress
        )
    );
    assert_eq!(
        field_selectors::from_value("test", &integration_point)
            .unwrap()
            .field_id(),
        runmat_analysis_fea::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
    );
}

#[test]
fn statistical_field_selectors_are_typed_composable_and_portable() {
    ensure_value_classes_registered();
    let stress = block_on(field_selectors::stress()).expect("typed structural field");
    let envelope = block_on(statistical_field_selectors::response_spectrum_envelope(
        stress.clone(),
    ))
    .expect("typed spectrum envelope");
    assert_eq!(
        field_selectors::from_value("test", &envelope)
            .unwrap()
            .field_id(),
        "structural.response_spectrum.envelope.stress"
    );
    let rms = block_on(statistical_field_selectors::random_response_rms(
        stress.clone(),
    ))
    .expect("typed random RMS");
    assert_eq!(
        field_selectors::from_value("test", &rms)
            .unwrap()
            .field_id(),
        "structural.random_response.rms.stress"
    );
    let variance = block_on(statistical_field_selectors::random_response_variance(
        stress,
    ))
    .expect("typed random variance");
    assert_eq!(
        field_selectors::from_value("test", &variance)
            .unwrap()
            .field_id(),
        "structural.random_response.variance.stress"
    );
    let nodal = block_on(statistical_field_selectors::response_spectrum_absolute_acceleration())
        .expect("typed nodal envelope");
    assert_eq!(
        field_selectors::from_value("test", &nodal)
            .unwrap()
            .field_id(),
        "structural.response_spectrum.nodal.absolute_acceleration"
    );
    for selector in [envelope, rms, variance, nodal] {
        let encoded = crate::execution::value_codec::encode_inline_value(&selector)
            .expect("statistical selector should be portable");
        assert_eq!(
            crate::execution::value_codec::decode_inline_value(&encoded)
                .expect("portable selector should decode"),
            selector
        );
    }
    let von_mises = block_on(field_selectors::von_mises()).expect("typed derived field");
    assert!(block_on(statistical_field_selectors::random_response_rms(von_mises)).is_err());
}

#[test]
fn fea_numeric_constructors_reject_resident_fields_without_provider_access() {
    let resident = || {
        Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("owned resident fixture")
                .handle,
        )
    };
    let cases = [
        block_on(frequency_builtin(resident())),
        block_on(fea_material_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("material".into()),
                Value::String("YoungsModulusPa".into()),
                resident(),
                Value::String("PoissonRatio".into()),
                Value::Num(0.3),
            ]),
        )),
    ];
    for result in cases {
        let error = result.expect_err("resident FEA constructor field must reject");
        assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(
            error.message().contains("real numeric scalar")
                || error.message().contains("cannot convert value")
        );
    }
}

#[test]
fn fea_structural_serializer_preserves_plan_counts_and_compare_deltas() {
    #[derive(Serialize)]
    struct FailureEntry {
        study_index: usize,
    }
    #[derive(Serialize)]
    struct StructuralPayload {
        study_count: usize,
        failure_entries: Vec<FailureEntry>,
        quality_reason_count_delta: i64,
        optional_delta: Option<i64>,
    }
    let value = serializable_to_object_preserving_integers(
        PLAN_NAME,
        &ERROR_INTERNAL,
        FEA_PLAN_CLASS,
        &StructuralPayload {
            study_count: 3,
            failure_entries: vec![FailureEntry { study_index: 2 }],
            quality_reason_count_delta: -4,
            optional_delta: None,
        },
        &["quality_reason_count_delta", "optional_delta"],
        &["study_count", "study_index"],
    )
    .expect("structural serializer");
    let Value::Object(object) = value else {
        panic!("expected structural object");
    };
    assert_eq!(
        object.properties().get("study_count"),
        Some(&Value::Int(IntValue::U64(3)))
    );
    assert_eq!(
        object.properties().get("quality_reason_count_delta"),
        Some(&Value::Int(IntValue::I64(-4)))
    );
    assert!(!object.properties().contains_key("optional_delta"));
    let Some(Value::Cell(entries)) = object.properties().get("failure_entries") else {
        panic!("expected failure entry cell");
    };
    let Some(Value::Struct(entry)) = entries.data().first() else {
        panic!("expected failure entry struct");
    };
    assert_eq!(
        entry.fields.get("study_index"),
        Some(&Value::Int(IntValue::U64(2)))
    );
}

#[test]
fn fea_constructor_aliases_arity_and_error_attribution_are_stable() {
    for (alias, expected) in [
        ("ConductivityWPerMk", "conductivity_w_per_mk"),
        ("SpecificHeatJPerKgK", "specific_heat_j_per_kgk"),
        ("ConductivitySPerM", "conductivity_s_per_m"),
        ("SpeedOfSoundMPerS", "speed_of_sound_m_per_s"),
        ("VolumetricWPerM3", "volumetric_w_per_m3"),
        ("InletVelocityMPerS", "inlet_velocity_m_per_s"),
    ] {
        assert_eq!(canonical_field_name(alias), expected, "{alias}");
    }

    let field_error = futures::executor::block_on(create_field_object_from_args(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("surplus fea.field argument");
    assert_eq!(field_error.context.builtin.as_deref(), Some(FIELD_NAME));

    let compare_error = futures::executor::block_on(create_compare_object_from_args(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("base".into()),
            Value::String("candidate".into()),
            Value::Num(2.0),
        ]),
    ))
    .expect_err("surplus fea.compare argument");
    assert_eq!(compare_error.context.builtin.as_deref(), Some(COMPARE_NAME));

    let model_error = block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("model".into()),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("invalid model geometry");
    assert_eq!(model_error.context.builtin.as_deref(), Some(MODEL_NAME));

    let duplicate_error = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("material".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(1.0),
            Value::String("youngs_modulus_pa".into()),
            Value::Num(2.0),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
        ]),
    ))
    .expect_err("duplicate normalized field");
    assert_eq!(
        duplicate_error.context.builtin.as_deref(),
        Some(MATERIAL_NAME)
    );
    assert!(duplicate_error.message().contains("duplicate"));
}

#[test]
fn fea_study_requires_geometry_asset() {
    let err = block_on(fea_study_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("demo".to_string().into()),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("invalid geometry should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
}

#[test]
fn fea_study_options_accept_a_jointly_validated_meshing_object() {
    let directory = tempfile::tempdir().expect("temporary artifact directory");
    let artifact = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
    let mesh_path = directory.path().join("fixture.solver-mesh.cbor");
    let evidence_path = directory.path().join("fixture.meshing-evidence.cbor");
    std::fs::write(
        &mesh_path,
        artifact.canonical_encode().expect("canonical mesh bytes"),
    )
    .expect("mesh fixture");
    std::fs::write(
        &evidence_path,
        evidence
            .canonical_encode()
            .expect("canonical evidence bytes"),
    )
    .expect("evidence fixture");
    let mesh = block_on(crate::builtins::meshing::load_builtin(
        mesh_path.to_string_lossy().to_string(),
        evidence_path.to_string_lossy().to_string(),
    ))
    .expect("validated meshing object");

    let options = StudyConstructorOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::String("Mesh".to_string().into()),
        mesh,
    ]))
    .expect("study mesh option");
    let supplied = options.mesh.expect("typed mesh output");

    assert_eq!(supplied.artifact, artifact);
    assert_eq!(supplied.evidence, evidence);
}

#[test]
fn fea_study_requires_explicit_model() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(geometry_path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("geometry should load");

    let err = block_on(fea_study_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("missing_model".to_string().into()),
            geometry,
        ]),
    ))
    .expect_err("missing model should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err
        .message()
        .contains("fea.study requires id, geometry, and model arguments"));
}

#[test]
fn fea_model_requires_explicit_procedure_sequence() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(geometry_path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("geometry should load");

    let err = block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("missing_sequence_model".to_string().into()),
            geometry,
        ]),
    ))
    .expect_err("missing sequence should fail");
    assert_eq!(err.identifier(), Some("RunMat:fea:InvalidInput"));
    assert!(err
        .message()
        .contains("fea.model requires exactly one of Sequence or Steps"));
}

#[test]
fn material_and_plan_objects_match_their_portable_schemas() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    ensure_value_classes_registered();
    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(200e9),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
            Value::String("MassDensityKgPerM3".into()),
            Value::Num(7850.0),
        ]),
    ))
    .expect("material");
    let encoded = crate::execution::value_codec::encode_inline_value(&material)
        .expect("a material must satisfy its portable schema");
    assert_eq!(
        crate::execution::value_codec::decode_inline_value(&encoded).expect("decode"),
        material
    );
    for property in ["mass_density_kg_per_m3", "elasticity", "provenance"] {
        assert!(
            crate::class_registry::lookup_property(
                &runmat_types::ClassIdentity::from(FEA_MATERIAL_CLASS),
                &runmat_types::MemberName::from(property),
            )
            .is_some(),
            "fea.Material must expose {property}"
        );
    }

    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(geometry_path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("geometry should load");
    let region_id = geometry_asset_from_value(MODEL_NAME, &geometry)
        .expect("geometry payload")
        .regions
        .first()
        .expect("fixture region")
        .region_id
        .clone();
    let region = || {
        block_on(crate::builtins::geometry::region::geometry_region_builtin(
            geometry.clone(),
            region_id.clone(),
        ))
        .expect("region")
    };
    let target = block_on(fea_property_region_target_solid_builtin(region())).expect("target");
    let assignment = block_on(fea_material_assignment_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![target, material.clone()]),
    ))
    .expect("assignment");
    let fixed =
        block_on(structural_fixed_builtin("support".to_string(), region())).expect("constraint");
    let step = block_on(structural_static_builtin(Value::String("static".into()))).expect("step");
    let model = block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("schema_model".into()),
            geometry.clone(),
            Value::String("Materials".into()),
            cell(vec![material]),
            Value::String("MaterialAssignments".into()),
            cell(vec![assignment]),
            Value::String("Constraints".into()),
            cell(vec![fixed]),
            Value::String("Steps".into()),
            cell(vec![step]),
        ]),
    ))
    .expect("model");
    let study = block_on(fea_study_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("schema_study".into()),
            geometry,
            model,
        ]),
    ))
    .expect("study");
    let plan = block_on(fea_plan_builtin(study)).expect("plan");
    let encoded = crate::execution::value_codec::encode_inline_value(&plan)
        .expect("a plan must satisfy its portable schema");
    assert_eq!(
        crate::execution::value_codec::decode_inline_value(&encoded).expect("decode"),
        plan
    );
    for property in ["sequence_id", "steps", "operation_sequence"] {
        assert!(
            crate::class_registry::lookup_property(
                &runmat_types::ClassIdentity::from(FEA_PLAN_CLASS),
                &runmat_types::MemberName::from(property),
            )
            .is_some(),
            "fea.Plan must expose {property}"
        );
    }
}

#[test]
fn scalar_enum_errors_are_attributed_to_the_calling_builtin() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry = test_support::load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
    let step =
        block_on(structural_static_builtin(Value::String("static".into()))).expect("static step");
    let frame_error = block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("framed".into()),
            geometry.clone(),
            Value::String("Steps".into()),
            step,
            Value::String("Frame".into()),
            Value::String("not_a_frame".into()),
        ]),
    ))
    .expect_err("an unknown frame must be rejected");
    assert_eq!(frame_error.context.builtin.as_deref(), Some(MODEL_NAME));
    assert_eq!(frame_error.identifier(), Some("RunMat:fea:InvalidInput"));

    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(200e9),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
        ]),
    ))
    .expect("material");
    let target = block_on(fea_property_region_target_all_solids_builtin(geometry))
        .expect("all-solids target");
    let confidence_error = block_on(fea_material_assignment_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            target,
            material,
            Value::String("Confidence".into()),
            Value::String("certain".into()),
        ]),
    ))
    .expect_err("an unknown confidence must be rejected");
    assert_eq!(
        confidence_error.context.builtin.as_deref(),
        Some(MATERIAL_ASSIGNMENT_NAME)
    );
}

#[test]
fn material_assignment_revision_is_uint32_and_round_trips_into_a_model() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry = test_support::load_stl_fixture(&tmp.path().join("part.stl"), 1.0);
    let revision = geometry_asset_from_value(MODEL_NAME, &geometry)
        .expect("geometry payload")
        .revision;
    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Num(200e9),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
        ]),
    ))
    .expect("material");
    let target = block_on(fea_property_region_target_all_solids_builtin(
        geometry.clone(),
    ))
    .expect("all-solids target");
    let assignment = block_on(fea_material_assignment_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![target, material.clone()]),
    ))
    .expect("assignment");
    let Value::Object(object) = &assignment else {
        panic!("expected fea.MaterialAssignment");
    };
    assert_eq!(
        object.properties().get("geometry_revision"),
        Some(&Value::Int(IntValue::U32(revision)))
    );
    let step =
        block_on(structural_static_builtin(Value::String("static".into()))).expect("static step");
    block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("uint32_revision".into()),
            geometry,
            Value::String("Materials".into()),
            cell(vec![material]),
            Value::String("MaterialAssignments".into()),
            cell(vec![assignment]),
            Value::String("Steps".into()),
            step,
        ]),
    ))
    .expect("a uint32 revision must decode when the assignment enters a model");
}

#[test]
fn fea_model_rejects_ambiguous_component_option_aliases() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");
    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(geometry_path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Units".to_string().into()),
            Value::String("meter".to_string().into()),
        ]),
    ))
    .expect("geometry should load");

    for alias in ["Assignments", "Cases"] {
        let error = block_on(fea_model_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(format!("rejected_{alias}").into()),
                geometry.clone(),
                Value::String("Steps".to_string().into()),
                modeling::static_test_steps_value(),
                Value::String(alias.to_string().into()),
                cell(Vec::new()),
            ]),
        ))
        .expect_err("ambiguous component option alias must be rejected");
        assert!(error.message().contains("unsupported fea.model option"));
    }
}

#[test]
fn typed_constructors_build_full_study_and_sweep_objects() {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    let geometry_path = tmp.path().join("part.step");
    std::fs::write(&geometry_path, SIMPLE_STEP).expect("geometry fixture should write");

    let geometry = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(geometry_path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("geometry should load");
    let asset =
        geometry_asset_from_value(MODEL_NAME, &geometry).expect("geometry payload should decode");
    let region_id = asset
        .regions
        .first()
        .expect("fixture should import a region")
        .region_id
        .clone();

    let material = block_on(fea_material_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("steel".to_string().into()),
            Value::String("YoungsModulusPa".to_string().into()),
            Value::Num(200e9),
            Value::String("PoissonRatio".to_string().into()),
            Value::Num(0.30),
        ]),
    ))
    .expect("material should build");
    assert_object_class(&material, FEA_MATERIAL_CLASS);

    let material_region = block_on(crate::builtins::geometry::region::geometry_region_builtin(
        geometry.clone(),
        region_id.clone(),
    ))
    .expect("material region should resolve");
    let beam_target = block_on(fea_property_region_target_beam_builtin(
        material_region.clone(),
    ))
    .expect("beam material target should build");
    let shell_target = block_on(fea_property_region_target_shell_builtin(
        material_region.clone(),
    ))
    .expect("shell material target should build");
    let material_target = block_on(fea_property_region_target_solid_builtin(
        material_region.clone(),
    ))
    .expect("solid material target should build");
    let cell_target = block_on(fea_property_region_target_cell_builtin(material_region))
        .expect("cell material target should build");
    for (target, expected) in [
        (beam_target, "beam"),
        (shell_target, "shell"),
        (material_target.clone(), "solid"),
        (cell_target, "cell"),
    ] {
        let decoded: modeling::GeometryBoundPropertyRegionTarget = object_payload(&target);
        let actual = match decoded.target() {
            runmat_analysis_core::PropertyRegionTarget::Beam(_) => "beam",
            runmat_analysis_core::PropertyRegionTarget::Shell(_) => "shell",
            runmat_analysis_core::PropertyRegionTarget::Solid(_) => "solid",
            runmat_analysis_core::PropertyRegionTarget::Cell(_) => "cell",
        };
        assert_eq!(actual, expected);
        assert!(!decoded.target().selects_all());
    }
    let all_solids_target = block_on(fea_property_region_target_all_solids_builtin(
        geometry.clone(),
    ))
    .expect("all-solids material target should build");
    let all_targets = [
        block_on(fea_property_region_target_all_beams_builtin(
            geometry.clone(),
        ))
        .expect("all-beams material target should build"),
        block_on(fea_property_region_target_all_shells_builtin(
            geometry.clone(),
        ))
        .expect("all-shells material target should build"),
        all_solids_target,
        block_on(fea_property_region_target_all_cells_builtin(
            geometry.clone(),
        ))
        .expect("all-cells material target should build"),
    ];
    for target in all_targets {
        assert_object_class(&target, FEA_PROPERTY_REGION_TARGET_CLASS);
        let decoded: modeling::GeometryBoundPropertyRegionTarget = object_payload(&target);
        assert!(decoded.target().selects_all());
    }
    let assignment = block_on(fea_material_assignment_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            material_target,
            material.clone(),
        ]),
    ))
    .expect("material assignment should build");
    assert_object_class(&assignment, FEA_MATERIAL_ASSIGNMENT_CLASS);
    assert!(block_on(fea_material_assignment_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String(region_id.clone().into()),
            Value::String("steel".into()),
        ])
    ))
    .is_err());

    let mut foreign_assignment = assignment.clone();
    let Value::Object(foreign_assignment_object) = &mut foreign_assignment else {
        unreachable!("material assignment is an object")
    };
    crate::builtins::fea::object_api::replace(
        "fea.fixture",
        foreign_assignment_object,
        "geometry_id",
        Value::String("different_geometry".into()),
    )
    .expect("ordinary fixture property edit");
    assert!(block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("cross_geometry_model".into()),
            geometry.clone(),
            Value::String("Materials".into()),
            cell(vec![material.clone()]),
            Value::String("MaterialAssignments".into()),
            cell(vec![foreign_assignment]),
        ])
    ))
    .is_err());

    let fixed_region = block_on(crate::builtins::geometry::region::geometry_region_builtin(
        geometry.clone(),
        region_id.clone(),
    ))
    .expect("fixed region should resolve");
    let fixed = block_on(structural_fixed_builtin(
        "fixed_base".to_string(),
        fixed_region,
    ))
    .expect("structural constraint should build");
    assert_object_class(&fixed, FEA_STRUCTURAL_CONSTRAINT_CLASS);

    let step = block_on(structural_static_builtin(Value::String(
        "static_step".to_string().into(),
    )))
    .expect("analysis step should build");
    assert_object_class(&step, FEA_STRUCTURAL_STEP_CLASS);

    let reference_temperature =
        block_on(temperature_kelvin_builtin(Value::Num(293.15))).expect("reference temperature");
    let domain = block_on(domains::thermomechanical_domain_builtin(
        reference_temperature,
    ))
    .expect("thermo-mechanical domain should build");

    let signal_domain = block_on(generator_policies::time_domain()).unwrap();
    let signal_sampling = block_on(generator_policies::uniform_sampling(
        Value::Num(0.0),
        Value::Num(1.0),
        Value::Int(IntValue::U64(3)),
    ))
    .unwrap();
    let (force_callable, _catalog) =
        test_support::portable_closure_fixture("loads.force", 41, vec![Value::Num(2.0)]);
    let generated_force = block_on(signals::callable::force(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("generated_force".into()),
            force_callable,
            signal_domain,
            signal_sampling,
        ]),
    ))
    .expect("callable force signal should build");

    let model = block_on(fea_model_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bracket_static_model".to_string().into()),
            geometry.clone(),
            Value::String("Materials".to_string().into()),
            cell(vec![material]),
            Value::String("MaterialAssignments".to_string().into()),
            cell(vec![assignment]),
            Value::String("Constraints".to_string().into()),
            cell(vec![fixed]),
            Value::String("Steps".to_string().into()),
            cell(vec![step.clone()]),
            Value::String("Domains".to_string().into()),
            cell(vec![domain]),
            Value::String("Signals".to_string().into()),
            cell(vec![generated_force]),
        ]),
    ))
    .expect("model should build");
    assert_object_class(&model, FEA_MODEL_CLASS);
    let encoded_model = crate::execution::value_codec::encode_inline_value(&model)
        .expect("model with a frozen generator should be portable");
    let model = crate::execution::value_codec::decode_inline_value(&encoded_model)
        .expect("portable model should decode");
    let composed_model = modeling::model_from_value("test", &model).unwrap();
    assert_eq!(composed_model.frozen_generators.bindings().len(), 1);
    let thermo = composed_model
        .model
        .thermo_mechanical
        .expect("thermo-mechanical domain");
    assert_eq!(thermo.reference_temperature.canonical_value(), 293.15);

    let mut substituted_model = model.clone();
    let Value::Object(substituted) = &mut substituted_model else {
        unreachable!()
    };
    let Some(Value::Cell(mut bindings)) = substituted
        .property(signals::bindings::FROZEN_GENERATORS_PROPERTY)
        .cloned()
    else {
        unreachable!()
    };
    bindings
        .make_data_unique_unbounded()
        .expect("detach the unclaimed tampered fixture from the original model");
    let Value::Struct(binding) = &mut bindings
        .try_data_mut_unique()
        .expect("tampered fixture owns its bindings")[0]
    else {
        unreachable!()
    };
    *binding = runmat_value::StructValue::from_fields_unbounded(
        binding
            .clone()
            .into_fields_unbounded()
            .map(|(name, value)| {
                if name == "signal_id" {
                    (name, Value::String("substituted".into()))
                } else {
                    (name, value)
                }
            }),
    );
    substituted.replace_property_unbounded(
        signals::bindings::FROZEN_GENERATORS_PROPERTY.to_string(),
        Value::Cell(bindings),
    );
    assert!(modeling::model_from_value("test", &substituted_model).is_err());

    let sequence = block_on(sequences::sequence(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bracket_sequence".to_string().into()),
            cell(vec![step]),
            cell(Vec::new()),
            cell(Vec::new()),
            cell(Vec::new()),
        ]),
    ))
    .expect("sequence should build");
    let execution = block_on(execution::execution_plan_defaults(
        sequence,
        block_on(execution::backend_cpu()).unwrap(),
    ))
    .expect("execution should build");

    let study = block_on(fea_study_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bracket_static".to_string().into()),
            geometry,
            model,
            Value::String("Execution".to_string().into()),
            execution,
        ]),
    ))
    .expect("study should build");
    assert_object_class(&study, FEA_STUDY_CLASS);
    let (binding_property, owner) = crate::class_registry::lookup_property(
        &runmat_types::ClassIdentity::from(FEA_STUDY_CLASS),
        &runmat_types::MemberName::from(signals::bindings::FROZEN_GENERATORS_PROPERTY),
    )
    .expect("study frozen-generator property should be registered");
    assert_eq!(owner.display_name(), FEA_STUDY_CLASS);
    assert_eq!(binding_property.get_access, MemberAccess::Private);
    let encoded_study = crate::execution::value_codec::encode_inline_value(&study)
        .expect("study with a frozen generator should be portable");
    let study = crate::execution::value_codec::decode_inline_value(&encoded_study)
        .expect("portable study should decode");
    let composed_study = modeling::study_from_value_with_generators("test", &study).unwrap();
    assert_eq!(composed_study.frozen_generators.bindings().len(), 1);
    let error = block_on(fea_run_builtin(study.clone()))
        .expect_err("unmaterialized callable must not enter a solver");
    assert!(error.message().contains("active RunMat runtime context"));

    let sweep = block_on(fea_sweep_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bracket_sweep".to_string().into()),
            cell(vec![study.clone(), study]),
            Value::String("FailFast".to_string().into()),
            Value::Bool(false),
        ]),
    ))
    .expect_err("duplicate study identities must be rejected");
    assert!(sweep.message().contains("declared more than once"));

    let study =
        study_to_object_with_generators(composed_study.study, composed_study.frozen_generators)
            .unwrap();
    let sweep = block_on(fea_sweep_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("bracket_sweep".to_string().into()),
            cell(vec![study]),
            Value::String("FailFast".to_string().into()),
            Value::Bool(false),
        ]),
    ))
    .expect("sweep should build");
    assert_object_class(&sweep, FEA_SWEEP_CLASS);
    let encoded_sweep = crate::execution::value_codec::encode_inline_value(&sweep)
        .expect("sweep with frozen generators should be portable");
    let sweep = crate::execution::value_codec::decode_inline_value(&encoded_sweep)
        .expect("portable sweep should decode");
    let sweep = modeling::sweep_from_value_with_generators("test", &sweep).unwrap();
    assert_eq!(
        sweep
            .frozen_generators
            .for_study("bracket_static")
            .bindings()
            .len(),
        1
    );
}

#[test]
fn fea_results_field_exposes_values_metadata_and_plot_context() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    let (run_value, _study, run_id) = synthetic_plot_run_value();

    let results = block_on(fea_results_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
    ))
    .expect("results should load");
    let Value::Object(results_object) = results.clone() else {
        panic!("expected results object");
    };
    assert_eq!(
        results_object.class_name().display_name(),
        FEA_RESULTS_CLASS
    );
    assert_eq!(
        results_object.class_name().display_name(),
        FEA_RESULTS_CLASS
    );
    assert_eq!(
        results_object.properties().get("run_id"),
        Some(&Value::String(run_id.clone().into()))
    );
    assert!(results_object
        .properties()
        .contains_key(FEA_STUDY_CONTEXT_PROPERTY));

    // Inline values remain usable even when the configured store cannot
    // provide synchronous access outside a captured transaction.
    let original_store = crate::analysis::storage::replace_artifact_store(std::sync::Arc::new(
        crate::analysis::storage::InMemoryAnalysisArtifactStore::new(),
    ))
    .unwrap();
    let filesystem = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    crate::analysis::storage::configure_artifact_store(
        crate::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
            root: "/unavailable-results".into(),
        },
    )
    .unwrap();
    assert!(crate::analysis::storage::resolve_run_attempt(&run_id).is_err());
    assert_eq!(
        block_on(fea_results_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![results.clone()]),
        ))
        .unwrap(),
        results,
    );
    let field = block_on(fea_field_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            results,
            block_on(field_selectors::von_mises()).expect("typed field selector"),
        ]),
    ))
    .expect("field should resolve by unique suffix");
    crate::analysis::storage::replace_artifact_store(original_store).unwrap();
    drop(filesystem);
    let Value::Object(field_object) = &field else {
        panic!("expected field object");
    };
    assert_eq!(field_object.class_name().display_name(), FEA_FIELD_CLASS);
    assert_eq!(
        field_object.properties().get("field_id"),
        Some(&Value::String("structural.von_mises".to_string().into()))
    );
    assert_eq!(
        field_object.properties().get("unit"),
        Some(&Value::String("Pa".to_string().into()))
    );
    assert_eq!(
        field_object.properties().get("location"),
        Some(&Value::String("element".to_string().into()))
    );
    assert_eq!(
        field_object.properties().get("topology_id"),
        Some(&Value::String("analysis_mesh".to_string().into()))
    );
    assert_eq!(
        field_object.properties().get("element_kind"),
        Some(&Value::String("tetrahedron4".to_string().into()))
    );
    assert_eq!(
        field_object.properties().get("entity_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    assert_eq!(
        field_object.properties().get("value_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    assert_eq!(
        field_object.properties().get("element_count"),
        Some(&Value::Int(runmat_value::IntValue::U64(1)))
    );
    let Some(Value::Tensor(values)) = field_object.properties().get("values") else {
        panic!("expected values tensor");
    };
    assert_eq!(
        field_object.properties().get("materialized"),
        Some(&Value::Bool(true))
    );
    assert_eq!(values.shape(), vec![1]);
    assert_eq!(values.materialize_f64(), vec![42.0]);
    assert!(field_object
        .properties()
        .contains_key(FEA_STUDY_CONTEXT_PROPERTY));
    assert_eq!(
        field_object.properties().get("run_id"),
        Some(&Value::String(run_id.clone().into()))
    );

    let page = block_on(fea_field_page_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            field,
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(1)),
        ]),
    ))
    .expect("persisted field page should load");
    let Value::Object(page) = page else {
        panic!("expected field page object")
    };
    assert_eq!(page.class_name().display_name(), FEA_FIELD_PAGE_CLASS);
    assert_eq!(
        crate::class_registry::lookup_method(
            &runmat_types::ClassIdentity::from(FEA_FIELD_CLASS),
            &runmat_types::MethodName::from("page"),
        )
        .expect("fea.Field.page method")
        .0
        .function_name,
        FIELD_PAGE_NAME
    );
    assert_eq!(
        page.properties().get("start"),
        Some(&Value::Int(IntValue::U64(1)))
    );
    assert_eq!(
        page.properties().get("count"),
        Some(&Value::Int(IntValue::U64(1)))
    );
    assert_eq!(page.properties().get("has_more"), Some(&Value::Bool(false)));
    assert!(matches!(
        page.properties().get("next_start"),
        Some(Value::Tensor(tensor)) if tensor.is_empty()
    ));
    let Some(Value::Tensor(values)) = page.properties().get("values") else {
        panic!("expected real field page values")
    };
    assert_eq!(values.shape(), vec![1, 1]);
    assert_eq!(values.materialize_f64(), vec![42.0]);
    let page = Value::Object(page);
    let encoded = crate::execution::value_codec::encode_inline_value(&page)
        .expect("field page should be portable");
    assert_eq!(
        crate::execution::value_codec::decode_inline_value(&encoded)
            .expect("portable field page should decode"),
        page
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn plot_and_mesh_overloads_return_figure_handles_for_fea_objects() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    let (run_value, _study, _run_id) = synthetic_plot_run_value();

    let mesh_handle = block_on(crate::call_builtin_async(
        "mesh",
        &[
            run_value.clone(),
            Value::String("Source".to_string().into()),
            Value::String("solver".to_string().into()),
            Value::String("Edges".to_string().into()),
            Value::Bool(true),
        ],
    ))
    .expect("run mesh should create a figure");
    assert!(matches!(mesh_handle.value(), Value::Num(handle) if *handle >= 1.0));

    let results = block_on(fea_results_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
    ))
    .expect("results should load");
    let field = block_on(fea_field_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            results,
            block_on(field_selectors::von_mises()).expect("typed field selector"),
        ]),
    ))
    .expect("field should resolve");
    let field_handle = block_on(crate::call_builtin_async("plot", &[field]))
        .expect("field plot should create a figure");
    assert!(matches!(field_handle.value(), Value::Num(handle) if *handle >= 1.0));
}

#[cfg(feature = "plot-core")]
#[test]
fn mesh_request_defaults_to_solver_edges_and_accepts_cad_reference() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    let (run_value, _study, _run_id) = synthetic_plot_run_value();
    let default = mesh_request_from_args(std::slice::from_ref(&run_value))
        .expect("default mesh request should parse");
    assert_eq!(
        default.source,
        crate::analysis::AnalysisFigureMeshSource::Solver
    );
    assert!(default.show_edges);

    let configured = mesh_request_from_args(&[
        run_value,
        Value::String("Source".to_string().into()),
        Value::String("cad_reference".to_string().into()),
        Value::String("Edges".to_string().into()),
        Value::Bool(false),
    ])
    .expect("configured mesh request should parse");
    assert_eq!(
        configured.source,
        crate::analysis::AnalysisFigureMeshSource::CadReference
    );
    assert!(!configured.show_edges);
}

#[cfg(feature = "plot-core")]
#[test]
fn fea_plot_request_accepts_solver_mesh_edge_option() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    let (run_value, _study, _run_id) = synthetic_plot_run_value();
    let results = block_on(fea_results_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![run_value]),
    ))
    .expect("results should load");
    let field = block_on(fea_field_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            results,
            block_on(field_selectors::von_mises()).expect("typed field selector"),
        ]),
    ))
    .expect("field should resolve");

    let request = plot_request_from_args(&[
        field,
        Value::String("Edges".to_string().into()),
        Value::Bool(true),
        Value::String("deformed".to_string().into()),
        Value::Bool(false),
        Value::String("overlay".to_string().into()),
        Value::String("cad".to_string().into()),
    ])
    .expect("plot request should parse mesh, deformation, and overlay options");

    assert_eq!(request.field_id, "structural.von_mises");
    assert_eq!(request.sample, None);
    assert!(request.options.show_solver_mesh_edges);
    assert!(!request.options.apply_deformation_overlay);
    assert_eq!(
        request.options.mesh_source,
        crate::analysis::AnalysisFigureMeshSource::CadReference
    );
}

#[test]
fn fea_usize_parser_reads_typed_integer_storage_exactly_and_rejects_float_boundary() {
    let wide = if usize::BITS == 64 {
        runmat_types::wire_integer::MAX_WIRE_INTEGER
    } else {
        u32::MAX as u64
    };
    let typed = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![wide]), vec![1, 1])
        .expect("typed integer");

    assert_eq!(
        usize_from_value(STUDY_NAME, &Value::Tensor(typed)).expect("typed integer"),
        wide as usize
    );
    let beyond = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![runmat_types::wire_integer::MAX_WIRE_INTEGER + 2]),
        vec![1, 1],
    )
    .expect("typed integer beyond the wire bound");
    assert!(usize_from_value(STUDY_NAME, &Value::Tensor(beyond)).is_err());

    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(usize_from_value(STUDY_NAME, &Value::Num(boundary)).is_err());
}

fn synthetic_plot_run_value() -> (Value, Value, String) {
    let tmp = tempfile::tempdir().expect("tempdir should be created");
    std::fs::write(tmp.path().join("part.stl"), TRIANGLE_STL)
        .expect("geometry fixture should write");
    let geometry_value = block_on(crate::builtins::geometry::geometry_load_builtin(
        Value::String(
            tmp.path()
                .join("part.stl")
                .to_string_lossy()
                .to_string()
                .into(),
        ),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("geometry should load");
    let geometry = geometry_asset_from_value(STUDY_NAME, &geometry_value)
        .expect("geometry payload should decode");
    let sequence = AnalysisSequence::from_steps(
        "synthetic_plot.sequence",
        vec![ProcedureStep::structural_static("static")],
    );
    let execution = AnalysisExecutionPlan::for_sequence(&sequence, ComputeBackend::Cpu)
        .expect("execution plan should match the sequence");
    let model = runmat_analysis_core::AnalysisModel {
        model_id: runmat_analysis_core::AnalysisModelId("synthetic_plot.model".to_string()),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: runmat_types::physical::CoordinateUnits::canonical(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: Vec::new(),
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        sequence,
    };
    let study = study_to_object(AnalysisStudySpec {
        study_id: "synthetic_plot".to_string(),
        geometry,
        model_source_evidence: None,
        model,
        execution,
        meshing_settings: None,
        immutable_values: Vec::new(),
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    })
    .expect("study context should encode");
    let identity = test_run_identity(runmat_analysis_core::AnalysisProcedure::Static);
    let run_id = crate::analysis::storage::next_run_id();
    let run = crate::analysis::AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: run_id.clone(),
        run: runmat_analysis_fea::FeaRunResult {
            accelerator_evidence: Vec::new(),
            identity,
            requested_backend: ComputeBackend::Cpu,
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend:
                runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_method: runmat_analysis_fea::SolverMethod::SparseDirectCholesky,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            solver_host_sync_count: 0,
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_convergence: Some(
                runmat_analysis_fea::contracts::StructuralStaticConvergence {
                    converged: true,
                    iterations: 0,
                    residual_norm: 0.0,
                    relative_residual: Some(0.0),
                    operator_proof:
                        runmat_analysis_fea::contracts::StaticOperatorProof::SparseCholesky,
                    spectral_estimate: None,
                },
            ),
            structural_static_balance: Some(synthetic_static_balance()),
            structural_constraint_result: Some(synthetic_constraint_result()),
            structural_connector_result: Some(synthetic_connector_result()),
            observations: Vec::new(),
            observation_histories: Vec::new(),
            series_reductions: Vec::new(),
            fields: vec![AnalysisField::host_f64(
                "structural.von_mises",
                AnalysisFieldLayout::scalar(
                    ObservationQuantity::VonMisesStress,
                    FieldLocation::Element,
                ),
                vec![1],
                vec![42.0],
            )],
        },
        render_topology: Some(crate::analysis::AnalysisRenderTopology {
            schema_version: "analysis_render_topology/v1".to_string(),
            source: crate::analysis::AnalysisRenderTopologySource::AnalysisMesh,
            volume_element_count: 1,
            meshes: vec![crate::analysis::AnalysisRenderMesh {
                mesh_id: "synthetic_plot_boundary".to_string(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]],
                regions: Vec::new(),
                vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
                triangle_volume_element_indices: vec![Some(0)],
            }],
        }),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: crate::analysis::QualityGate::Pass,
        solver_convergence: crate::analysis::QualityGate::Pass,
        result_quality: crate::analysis::QualityGate::Pass,
        run_status: crate::analysis::RunStatus::Completed,
        numerically_acceptable: true,
        quality_reasons: Vec::new(),
        provenance: crate::analysis::RunProvenance {
            requested_backend: ComputeBackend::Cpu,
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend:
                runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: "fp64".to_string(),
            deterministic_mode: true,
            solver_method: runmat_analysis_fea::SolverMethod::SparseDirectCholesky,
            preconditioner: runmat_analysis_fea::SolverPreconditioner::None,
            quality_policy: "balanced".to_string(),
            fallback_events: Vec::new(),
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
    };
    crate::analysis::storage::persist_run_result(&run).expect("run should persist");

    let mut object = ObjectInstance::new(FEA_STEP_RUN_RESULT_CLASS.to_string());
    crate::builtins::fea::object_api::replace(
        "fea.fixture",
        &mut object,
        "run_id",
        Value::String(run_id.clone().into()),
    )
    .expect("ordinary fixture property edit");
    crate::builtins::fea::object_api::replace(
        "fea.fixture",
        &mut object,
        FEA_STUDY_CONTEXT_PROPERTY,
        study.clone(),
    )
    .expect("ordinary fixture property edit");
    (Value::Object(object), study, run_id)
}

fn persist_synthetic_indexed_results() -> String {
    let indexed_fields = |prefix: &str| {
        vec![
            AnalysisField::host_f64(
                format!("{prefix}.0"),
                AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
                vec![1],
                vec![10.0],
            ),
            AnalysisField::host_f64(
                format!("{prefix}.1"),
                AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
                vec![1],
                vec![20.0],
            ),
        ]
    };
    let run_id = crate::analysis::storage::next_run_id();
    let identity = test_run_identity(runmat_analysis_core::AnalysisProcedure::TransientResponse);
    let run = crate::analysis::AnalysisRunResult {
            observation_scalars: Vec::new(),
            run_id: run_id.clone(),
            run: runmat_analysis_fea::FeaRunResult {
                accelerator_evidence: Vec::new(),
                identity,
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_method: runmat_analysis_fea::SolverMethod::NewmarkAverageAccelerationPcg,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::Jacobi,
                solver_host_sync_count: 0,
                diagnostics: Vec::new(),
                spatial_projection_evidence: Vec::new(),
                structural_static_convergence: None,
                structural_static_balance: None,
                structural_constraint_result: None,
                structural_connector_result: Some(synthetic_connector_result()),
                observations: Vec::new(),
                observation_histories: Vec::new(),
                series_reductions: Vec::new(),
                fields: Vec::new(),
            },
            render_topology: None,
            modal_results: None,
            buckling_results: None,
            thermal_results: None,
            transient_results: Some(crate::analysis::TransientResultsData {
                transient_payload_version: "transient_results/v1".to_string(),
                initial_state: runmat_analysis_fea::StructuralInitialStateEvidence {
                    procedure_step_id: "synthetic_transient".to_string(),
                    basis: runmat_analysis_fea::StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
                        dof_count: 1,
                        operator_identity: runmat_execution::Digest::sha256(
                            b"runtime-synthetic-transient-fixture",
                        ),
                    },
                    total_dof_count: 1,
                    authored_displacement_dof_count: 0,
                    constraint_derived_displacement_dof_count: 0,
                    zero_default_displacement_dof_count: 1,
                    authored_velocity_dof_count: 0,
                    constraint_derived_velocity_dof_count: 0,
                    zero_default_velocity_dof_count: 1,
                    direct_constrained_dof_count: 0,
                    constraint_equation_count: 0,
                    maximum_displacement_constraint_residual_bits: 0.0_f64.to_bits(),
                    maximum_velocity_constraint_residual_bits: 0.0_f64.to_bits(),
                    applied_conditions: Vec::new(),
                },
                time_points_s: vec![0.0, 1.0],
                displacement_snapshots: indexed_fields("displacement"),
                rotation_snapshots: Vec::new(),
                velocity_snapshots: indexed_fields("velocity"),
                angular_velocity_snapshots: Vec::new(),
                acceleration_snapshots: indexed_fields("acceleration"),
                angular_acceleration_snapshots: Vec::new(),
                strain_snapshots: indexed_fields("strain"),
                stress_snapshots: indexed_fields("stress"),
                strain_energy_density_snapshots: indexed_fields("strain_energy_density"),
                von_mises_snapshots: indexed_fields("von_mises"),
                kinetic_energy_snapshots: indexed_fields("kinetic_energy"),
                strain_energy_snapshots: indexed_fields("strain_energy"),
                residual_norm_snapshots: indexed_fields("residual_norm"),
                reaction_force_snapshots: indexed_fields("reaction_force"),
                reaction_moment_snapshots: Vec::new(),
                external_work_snapshots: indexed_fields("external_work"),
                support_work_snapshots: indexed_fields("support_work"),
                damping_dissipation_snapshots: indexed_fields("damping_dissipation"),
                energy_balance_residual_snapshots: indexed_fields("energy_balance_residual"),
                relative_energy_balance_error_snapshots: indexed_fields(
                    "relative_energy_balance_error",
                ),
                dynamic_balance_snapshots: (0..2)
                    .map(|_| runmat_analysis_fea::StructuralDynamicBalanceSnapshot {
                        applied_load: vec![0.0],
                        inertia_force: vec![0.0],
                        damping_force: vec![0.0],
                        elastic_force: vec![0.0],
                        equilibrium_residual: vec![0.0],
                    })
                    .collect(),
                electro_thermal_temperature_snapshots: Vec::new(),
                electro_thermal_thermal_residual_snapshots: Vec::new(),
                residual_norms: vec![0.5, 0.25],
                integration_method:
                    crate::analysis::TransientIntegrationMethod::NewmarkAverageAcceleration,
            }),
            nonlinear_results: None,
            model_validity: crate::analysis::QualityGate::Pass,
            solver_convergence: crate::analysis::QualityGate::Pass,
            result_quality: crate::analysis::QualityGate::Pass,
            run_status: crate::analysis::RunStatus::Completed,
            numerically_acceptable: true,
            quality_reasons: Vec::new(),
            provenance: crate::analysis::RunProvenance {
                requested_backend: ComputeBackend::Cpu,
                execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
                solver_backend: runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
                solver_device_apply_k_ratio: 0.0,
                solver_host_sync_count: 0,
                precision_mode: "fp64".to_string(),
                deterministic_mode: true,
                solver_method: runmat_analysis_fea::SolverMethod::NewmarkAverageAccelerationPcg,
                preconditioner: runmat_analysis_fea::SolverPreconditioner::Jacobi,
                quality_policy: "balanced".to_string(),
                fallback_events: Vec::new(),
                effective_memory_ceiling_bytes: None,
                static_solve_path: None,
            },
        };
    crate::analysis::storage::persist_run_result(&run).expect("indexed run should persist");
    run_id
}

#[test]
fn fea_results_translates_successful_snapshot_selectors_once() {
    let _guard = crate::analysis::storage::test_support::isolated_artifact_store();
    let run_id = persist_synthetic_indexed_results();
    let selected = block_on(fea_results_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String(run_id.into()),
            Value::String("TransientSnapshotIndices".to_string().into()),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).expect("double selector")),
        ]),
    ))
    .expect("one-based selectors should resolve the second stored entries");
    let Value::Object(selected) = selected else {
        panic!("results object");
    };
    let Some(Value::Struct(transient)) = selected.properties().get("transient_results") else {
        panic!("transient results");
    };
    let Some(Value::Tensor(time_points)) = transient.fields.get("time_points_s") else {
        panic!("transient time points");
    };
    assert_eq!(time_points.materialize_f64(), vec![1.0]);
    assert!(transient.fields.contains_key("strain_snapshots"));
    assert!(transient.fields.contains_key("stress_snapshots"));
    assert!(transient
        .fields
        .contains_key("strain_energy_density_snapshots"));
    assert!(transient.fields.contains_key("strain_energy_snapshots"));
    assert!(transient.fields.contains_key("reaction_force_snapshots"));
    assert!(transient.fields.contains_key("external_work_snapshots"));
    assert!(transient.fields.contains_key("support_work_snapshots"));
    assert!(transient
        .fields
        .contains_key("damping_dissipation_snapshots"));
    assert!(transient
        .fields
        .contains_key("energy_balance_residual_snapshots"));
    assert!(transient
        .fields
        .contains_key("relative_energy_balance_error_snapshots"));
    assert!(transient.fields.contains_key("dynamic_balance_snapshots"));
    let Some(Value::Struct(initial_state)) = transient.fields.get("initial_state") else {
        panic!("structural initial-state evidence");
    };
    assert_eq!(
        initial_state.fields.get("procedure_step_id"),
        Some(&Value::String("synthetic_transient".to_string().into()))
    );

    let Some(Value::Struct(summary)) = selected.properties().get("summary") else {
        panic!("results summary");
    };
    let Some(Value::Tensor(indices)) = summary.fields.get("available_mode_indices") else {
        panic!("available mode indices");
    };
    assert!(indices
        .integer_storage()
        .expect("exact public indices")
        .exact_values()
        .is_empty());
}

fn assert_object_class(value: &Value, expected: &str) {
    let Value::Object(object) = value else {
        panic!("expected object value");
    };
    assert_eq!(object.class_name().display_name(), expected);
    assert!(!object.properties().is_empty());
    assert!(object
        .properties()
        .keys()
        .all(|name| !name.starts_with("__runmat_")
            || name == crate::builtins::common::json::typed_object_projection::PROPERTY));
    object_projection::validate(object).expect("only validated shape provenance is permitted");
}
