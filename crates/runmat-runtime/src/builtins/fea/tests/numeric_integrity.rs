use super::*;

#[test]
fn fea_usize_parsers_preserve_typed_bounds_and_reject_invalid_values() {
    use runmat_value::{IntValue, IntegerStorage};

    assert_eq!(
        usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::U16(7))).unwrap(),
        7
    );
    assert!(usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::I8(-1))).is_err());
    assert!(usize_from_value(INTERFACE_NAME, &Value::Num(1.5)).is_err());
    assert!(usize_vec_from_value(
        INTERFACE_NAME,
        &Value::Tensor(Tensor::new_2d(vec![1.0, -1.0], 1, 2).unwrap())
    )
    .is_err());
    let typed_indices = Tensor::new_integer(IntegerStorage::U16(vec![2, 4]), vec![1, 2]).unwrap();
    assert_eq!(
        usize_vec_from_value(INTERFACE_NAME, &Value::Tensor(typed_indices)).unwrap(),
        vec![2, 4]
    );

    let maximum = usize_from_value(INTERFACE_NAME, &Value::Int(IntValue::U64(u64::MAX)));
    if usize::BITS == 64 {
        assert_eq!(maximum.unwrap(), usize::MAX);
    } else {
        assert!(maximum.is_err());
    }
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
        RUN_OPTIONS_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_OPTIONS_CLASS,
        &IntegerMirror {
            signed: -9,
            unsigned: u64::MAX,
            matrix: vec![vec![1, 2], vec![3, 4]],
            floating: 1.0,
            floating_vector: vec![2.0, 3.0],
        },
        None,
    )
    .expect("integer-preserving public mirror");
    assert!(matches!(
        object.properties.get("signed"),
        Some(Value::Int(IntValue::I64(-9)))
    ));
    assert!(matches!(
        object.properties.get("unsigned"),
        Some(Value::Int(IntValue::U64(u64::MAX)))
    ));
    let Some(Value::Tensor(matrix)) = object.properties.get("matrix") else {
        panic!("exact integer matrix");
    };
    assert_eq!(matrix.shape, vec![2, 2]);
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
        object.properties.get("floating"),
        Some(Value::Num(1.0))
    ));
    assert!(matches!(
        object.properties.get("floating_vector"),
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
        None,
        &[],
        &["available_mode_indices", "iteration_counts"],
    )
    .expect("schema-aware empty integer vectors");
    let Value::Object(empty) = empty else {
        panic!("results object");
    };
    for name in ["available_mode_indices", "iteration_counts"] {
        let Some(Value::Tensor(values)) = empty.properties.get(name) else {
            panic!("empty exact vector {name}");
        };
        assert!(values.integer_storage().is_some());
        assert!(values.is_empty());
    }
}

#[test]
fn fea_execution_controls_enforce_exact_structural_boundaries() {
    let options = block_on(fea_run_options_builtin(vec![
        Value::String("modal".into()),
        Value::String("ModeCount".into()),
        Value::Num(3.0),
        Value::String("ResidualWarnThreshold".into()),
        Value::Int(IntValue::U64(1)),
    ]))
    .expect("ordinary integral double count and typed floating control");
    let Value::Object(options) = options else {
        panic!("run options object");
    };
    let Some(Value::Struct(payload)) = options.properties.get("options") else {
        panic!("run options payload");
    };
    assert!(matches!(
        payload.fields.get("mode_count"),
        Some(Value::Int(IntValue::U64(3)))
    ));
    assert!(matches!(
        payload.fields.get("residual_warn_threshold"),
        Some(Value::Num(1.0))
    ));

    for (solver, exact_field, floating_field) in [
        ("modal", "ModeCount", "ResidualWarnThreshold"),
        ("acoustic", "ModeCount", "ResidualWarnThreshold"),
        ("thermal", "StepCount", "TimeStepS"),
        ("transient", "MaxStepRetries", "Tolerance"),
        ("cfd", "MaxLinearIters", "ResidualWarnThreshold"),
        ("cht", "StepCount", "ResidualWarnThreshold"),
        ("fsi", "MaxLinearIters", "Tolerance"),
        ("nonlinear", "TangentRefreshInterval", "Tolerance"),
        (
            "electromagnetic",
            "HarmonicMaxIterations",
            "HarmonicTolerance",
        ),
    ] {
        let value = block_on(fea_run_options_builtin(vec![
            Value::String(solver.into()),
            Value::String(exact_field.into()),
            Value::Int(IntValue::U32(3)),
            Value::String(floating_field.into()),
            Value::Int(IntValue::I16(1)),
        ]))
        .unwrap_or_else(|error| panic!("{solver} typed run options: {error}"));
        let Value::Object(object) = value else {
            panic!("{solver} run options object");
        };
        let Some(Value::Struct(payload)) = object.properties.get("options") else {
            panic!("{solver} run options payload");
        };
        let exact_field = canonical_field_name(exact_field);
        let floating_field = canonical_field_name(floating_field);
        assert!(matches!(
            payload.fields.get(&exact_field),
            Some(Value::Int(IntValue::U64(3)))
        ));
        assert!(matches!(
            payload.fields.get(&floating_field),
            Some(Value::Num(1.0))
        ));
    }

    let prep_context = block_on(fea_run_options_builtin(vec![
        Value::String("modal".into()),
        Value::String("PrepContext".into()),
        Value::String("internal".into()),
    ]))
    .expect_err("internal prep context must not be public");
    assert_eq!(prep_context.identifier(), Some("RunMat:fea:InvalidInput"));

    let extra_step = block_on(fea_step_builtin(vec![
        Value::String("step".into()),
        Value::String("modal".into()),
        Value::Num(1.0),
    ]))
    .expect_err("step has exact arity");
    assert_eq!(extra_step.identifier(), Some("RunMat:fea:InvalidInput"));

    let zero_window = block_on(fea_trends_builtin(vec![
        Value::String("WindowSize".into()),
        Value::Int(IntValue::U8(0)),
    ]))
    .expect_err("trend window must be positive");
    assert_eq!(zero_window.identifier(), Some("RunMat:fea:InvalidInput"));

    let scalar_double = Tensor::new(vec![4.0], vec![1, 1]).expect("double scalar tensor");
    assert_eq!(
        usize_from_value(RUN_OPTIONS_NAME, &Value::Tensor(scalar_double.clone())).unwrap(),
        4
    );
    assert!(exact_bool_from_value(
        RESULTS_NAME,
        &Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("double flag"))
    )
    .unwrap());

    let scalar_single =
        Tensor::new_with_dtype(vec![4.0], vec![1, 1], runmat_value::NumericDType::F32)
            .expect("single scalar tensor");
    assert!(usize_from_value(RUN_OPTIONS_NAME, &Value::Tensor(scalar_single)).is_err());
}

#[test]
fn fea_result_selectors_are_exact_one_based_vectors_and_flags_are_zero_one() {
    let wide = 9_007_199_254_740_993_u64;
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
fn fea_json_preserves_native_integer_scalars_and_tensors() {
    let maximum = runmat_value::IntValue::U64(u64::MAX);
    assert_eq!(
        value_to_json(INTERFACE_NAME, &Value::Int(maximum.clone()))
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
        value_to_json(INTERFACE_NAME, &Value::Tensor(scalar))
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
        value_to_json(INTERFACE_NAME, &Value::Tensor(tensor))
            .expect("tensor json")
            .to_string(),
        "[42,18446744073709551615]"
    );
}

#[test]
fn fea_struct_array_integer_promotion_aligns_rectangular_json_column_major() {
    let element = || {
        let mut structure = StructValue::new();
        structure.insert("id", Value::Num(0.0));
        structure
    };
    let array =
        StructArray::new(vec![element(), element(), element(), element()], vec![2, 2]).unwrap();
    let mut value = Value::StructArray(array);
    let json = serde_json::json!([[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}]]);
    promote_named_integer_fields("fea.test", &ERROR_INPUT, &mut value, &json, &[], &["id"])
        .unwrap();
    let Value::StructArray(array) = value else {
        panic!("expected structure array");
    };
    assert_eq!(
        array.field_values("id").unwrap(),
        [
            Value::Int(IntValue::U64(1)),
            Value::Int(IntValue::U64(3)),
            Value::Int(IntValue::U64(2)),
            Value::Int(IntValue::U64(4)),
        ]
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
        let expected = boundary_integer_to_f64(&integer);
        let domain = block_on(fea_domain_builtin(vec![
            Value::String("electromagnetic".into()),
            Value::String("AppliedCurrentA".into()),
            Value::Int(integer.clone()),
        ]))
        .expect("domain integer field");
        let domain: DomainPayload = object_payload(&domain);
        let domain: runmat_analysis_core::ElectromagneticDomain =
            json_deserialize(DOMAIN_NAME, domain.data, "electromagnetic domain")
                .expect("typed domain storage boundary");
        assert_eq!(domain.applied_current_a, expected);

        let interface = block_on(fea_interface_builtin(vec![
            Value::String("contact".into()),
            Value::String("left".into()),
            Value::String("right".into()),
            Value::String("FrictionCoefficient".into()),
            Value::Int(integer.clone()),
        ]))
        .expect("interface integer field");
        let interface: AnalysisInterface = object_payload(&interface);
        let AnalysisInterfaceKind::Contact(contact) = interface.kind else {
            panic!("expected contact interface");
        };
        assert_eq!(contact.friction_coefficient, expected);

        let load = block_on(fea_load_case_builtin(vec![
            Value::String("pressure".into()),
            Value::String("face".into()),
            Value::String("pressure".into()),
            Value::String("MagnitudePa".into()),
            Value::Int(integer.clone()),
        ]))
        .expect("load integer field");
        let load: LoadCase = object_payload(&load);
        assert!(
            matches!(load.kind, LoadKind::Pressure { magnitude_pa } if magnitude_pa == expected)
        );

        let material = block_on(fea_material_builtin(vec![
            Value::String("material".into()),
            Value::String("YoungsModulusPa".into()),
            Value::Int(integer),
            Value::String("PoissonRatio".into()),
            Value::Int(IntValue::U8(0)),
        ]))
        .expect("material integer field");
        let material: MaterialModel = object_payload(&material);
        assert_eq!(material.mechanical.youngs_modulus_pa, expected);
    }
}

#[test]
fn fea_domain_revision_and_field_metadata_remain_exact_in_public_objects() {
    let mut field_source = StructValue::new();
    field_source.insert("source_id", Value::String("temperature".into()));
    field_source.insert("revision", Value::Int(IntValue::U32(u32::MAX)));
    let domain = block_on(fea_domain_builtin(vec![
        Value::String("thermoMechanical".into()),
        Value::String("FieldSource".into()),
        Value::Struct(field_source),
    ]))
    .expect("domain source revision");
    let Value::Object(domain) = domain else {
        panic!("expected domain object");
    };
    let Value::Struct(data) = domain.properties.get("data").expect("domain data") else {
        panic!("expected domain data struct");
    };
    let Value::Struct(source) = data.fields.get("field_source").expect("field source") else {
        panic!("expected field source struct");
    };
    assert_eq!(
        source.fields.get("revision"),
        Some(&Value::Int(IntValue::U64(u64::from(u32::MAX))))
    );

    for revision in [
        Value::Int(IntValue::I8(-1)),
        Value::Int(IntValue::U64(u64::MAX)),
        Value::Num(1.0),
    ] {
        let mut field_source = StructValue::new();
        field_source.insert("source_id", Value::String("temperature".into()));
        field_source.insert("revision", revision);
        let error = block_on(fea_domain_builtin(vec![
            Value::String("thermoMechanical".into()),
            Value::String("FieldSource".into()),
            Value::Struct(field_source),
        ]))
        .expect_err("invalid structural revision must reject");
        assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
        assert_eq!(error.context.builtin.as_deref(), Some(DOMAIN_NAME));
    }

    let field = AnalysisField {
        field_id: "stress".into(),
        shape: vec![u32::MAX as usize, 0],
        values: AnalysisFieldValues::HostF64(Vec::new()),
    };
    let descriptor = AnalysisFieldDescriptor::from_field(&field);
    let object = field_to_object(&field, &descriptor).expect("field object");
    let Value::Tensor(shape) = object.properties.get("shape").expect("shape") else {
        panic!("expected integer shape tensor");
    };
    assert_eq!(shape.numeric_dtype(), runmat_value::NumericDType::U64);
    assert_eq!(
        shape.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::from(u32::MAX), 0]))
    );
    assert_eq!(
        object.properties.get("element_count"),
        Some(&Value::Int(IntValue::U64(0)))
    );

    let device_field = AnalysisField {
        field_id: "device".into(),
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
fn fea_numeric_constructors_reject_resident_fields_without_provider_access() {
    let resident = || {
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 1,
            descriptor: Default::default(),
        })
    };
    let cases = [
        block_on(fea_domain_builtin(vec![
            Value::String("electromagnetic".into()),
            Value::String("AppliedCurrentA".into()),
            resident(),
        ])),
        block_on(fea_interface_builtin(vec![
            Value::String("contact".into()),
            Value::String("left".into()),
            Value::String("right".into()),
            Value::String("FrictionCoefficient".into()),
            resident(),
        ])),
        block_on(fea_load_case_builtin(vec![
            Value::String("pressure".into()),
            Value::String("face".into()),
            Value::String("pressure".into()),
            Value::String("MagnitudePa".into()),
            resident(),
        ])),
        block_on(fea_material_builtin(vec![
            Value::String("material".into()),
            Value::String("YoungsModulusPa".into()),
            resident(),
            Value::String("PoissonRatio".into()),
            Value::Num(0.3),
        ])),
    ];
    for result in cases {
        let error = result.expect_err("resident FEA constructor field must reject");
        assert_eq!(error.identifier(), Some("RunMat:fea:InvalidInput"));
        assert!(error.message().contains("cannot convert value"));
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
        None,
        &["quality_reason_count_delta", "optional_delta"],
        &["study_count", "study_index"],
    )
    .expect("structural serializer");
    let Value::Object(object) = value else {
        panic!("expected structural object");
    };
    assert_eq!(
        object.properties.get("study_count"),
        Some(&Value::Int(IntValue::U64(3)))
    );
    assert_eq!(
        object.properties.get("quality_reason_count_delta"),
        Some(&Value::Int(IntValue::I64(-4)))
    );
    assert!(matches!(
        object.properties.get("optional_delta"),
        Some(Value::Tensor(tensor)) if tensor.is_empty()
    ));
    let Some(Value::Struct(entry)) = object.properties.get("failure_entries") else {
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
        ("ThermalConductanceWPerM2K", "thermal_conductance_w_per_m2k"),
        ("ContactResistanceM2KPerW", "contact_resistance_m2k_per_w"),
    ] {
        assert_eq!(canonical_field_name(alias), expected, "{alias}");
    }

    let field_error = create_field_object_from_args(vec![
        Value::Num(1.0),
        Value::String("stress".into()),
        Value::Num(2.0),
    ])
    .expect_err("surplus fea.field argument");
    assert_eq!(field_error.context.builtin.as_deref(), Some(FIELD_NAME));

    let compare_error = create_compare_object_from_args(vec![
        Value::String("base".into()),
        Value::String("candidate".into()),
        Value::Num(2.0),
    ])
    .expect_err("surplus fea.compare argument");
    assert_eq!(compare_error.context.builtin.as_deref(), Some(COMPARE_NAME));

    let model_error = block_on(fea_model_builtin(vec![
        Value::String("model".into()),
        Value::Num(1.0),
    ]))
    .expect_err("invalid model geometry");
    assert_eq!(model_error.context.builtin.as_deref(), Some(MODEL_NAME));

    let duplicate_error = block_on(fea_material_builtin(vec![
        Value::String("material".into()),
        Value::String("YoungsModulusPa".into()),
        Value::Num(1.0),
        Value::String("youngs_modulus_pa".into()),
        Value::Num(2.0),
        Value::String("PoissonRatio".into()),
        Value::Num(0.3),
    ]))
    .expect_err("duplicate normalized field");
    assert_eq!(
        duplicate_error.context.builtin.as_deref(),
        Some(MATERIAL_NAME)
    );
    assert!(duplicate_error.message().contains("duplicate"));
}
