use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DomainPayload {
    pub(super) kind: String,
    pub(super) data: serde_json::Value,
}

#[derive(Debug, Clone)]
pub(super) enum BoundConstraintPayload {
    BoundStructural(BoundStructuralConstraint),
    BoundThermal(BoundThermalConstraint),
    BoundElectrical(BoundElectricalConstraint),
    BoundMagnetic(BoundMagneticConstraint),
    BoundAcoustic(BoundAcousticConstraint),
    BoundFluid(BoundFluidConstraint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelDefaultsMode {
    TemplateScaffold,
    None,
}

impl Default for ModelDefaultsMode {
    fn default() -> Self {
        Self::TemplateScaffold
    }
}

#[derive(Debug, Default)]
pub(super) struct StudyConstructorOptions {
    pub(super) template: Option<AnalysisModelTemplate>,
    pub(super) backend: Option<ComputeBackend>,
    pub(super) requirement: Option<FormulationRequirement>,
    pub(super) model_id: Option<String>,
    pub(super) model: Option<AnalysisModel>,
    pub(super) frame: Option<CoordinateFrame>,
    pub(super) model_defaults: ModelDefaultsMode,
    pub(super) materials: Vec<MaterialModel>,
    pub(super) material_assignments: Vec<MaterialAssignment>,
    pub(super) constraints: Vec<BoundConstraintPayload>,
    pub(super) boundaries: Vec<GeometryBoundBoundary>,
    pub(super) signals: Vec<PhysicalSignal>,
    pub(super) excitations: Vec<ExcitationAction>,
    pub(super) excitation_cases: Vec<ExcitationCase>,
    pub(super) steps: Vec<ProcedureStep>,
    pub(super) domains: Vec<DomainPayload>,
    pub(super) interfaces: Vec<AnalysisInterface>,
    pub(super) mesh: Option<crate::meshing::MeshingExecutionOutput>,
}

pub(super) fn create_study_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires id and geometry arguments",
        ));
    }
    let study_id = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(STUDY_NAME, &args[1])?;
    let options = StudyConstructorOptions::parse(&args[2..])?;
    let supplied_mesh = options.mesh.clone();
    let template = resolve_study_template(&options)?;
    let backend = options.backend.unwrap_or(ComputeBackend::Cpu);
    let requirement = options
        .requirement
        .unwrap_or(FormulationRequirement::Production);
    let model_id = options.model_id.clone().unwrap_or_else(|| {
        options
            .model
            .as_ref()
            .map(|model| model.model_id.0.clone())
            .unwrap_or_else(|| format!("{}_model", sanitize_id(&study_id)))
    });
    let mut model = match options.model {
        Some(model) => model,
        None if options.has_model_components() => build_model_from_parts(
            STUDY_NAME,
            &geometry,
            model_id.clone(),
            template,
            options.model_defaults,
            options.frame,
            options.materials,
            options.material_assignments,
            options.constraints,
            options.boundaries,
            options.signals,
            options.excitations,
            options.excitation_cases,
            options.steps,
            options.domains,
            options.interfaces,
        )?,
        None => {
            analysis_create_model_op(
                &geometry,
                AnalysisCreateModelIntentSpec {
                    model_id: model_id.clone(),
                    template,
                },
                OperationContext::new(None, None),
            )
            .map_err(|error| operation_error(STUDY_NAME, &ERROR_OPERATION, error))?
            .data
        }
    };
    if let Some(mesh) = supplied_mesh.as_ref() {
        crate::analysis::meshing::apply_boundary_region_ids(&mut model, &mesh.boundary_region_ids);
    }
    let [procedure_step] = model.sequence.steps.as_slice() else {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study currently requires a model with exactly one procedure step",
        ));
    };
    let solver_options = AnalysisSolverOptions::defaults_for(procedure_step)
        .map_err(|message| builtin_error(STUDY_NAME, &ERROR_INPUT, message))?;
    let execution = AnalysisExecutionPlan {
        steps: vec![AnalysisExecutionStep {
            step_id: procedure_step.step_id.clone(),
            backend,
            requirement,
            solver_options,
        }],
    };
    let persisted_mesh = supplied_mesh
        .as_ref()
        .map(crate::analysis::meshing::persist_mesh_output)
        .transpose()
        .map_err(|message| builtin_error(STUDY_NAME, &ERROR_OPERATION, message))?;
    let spec = AnalysisStudySpec {
        study_id,
        geometry,
        create_model_intent: AnalysisCreateModelIntentSpec { model_id, template },
        model: Some(model),
        execution,
        meshing_settings: None,
        solver_mesh_artifact_path: persisted_mesh
            .as_ref()
            .and_then(|mesh| mesh.artifact_path.clone()),
        meshing_evidence_artifact_path: persisted_mesh
            .as_ref()
            .and_then(|mesh| mesh.evidence_path.clone()),
    };
    study_to_object(spec)
}

impl StudyConstructorOptions {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                "fea.study options must be Name, Value pairs",
            ));
        }
        let mut options = Self::default();
        let mut seen = HashSet::new();
        for pair in args.chunks(2) {
            let key = option_key(&pair[0], STUDY_NAME)?;
            let canonical = match key.as_str() {
                "materialassignments" | "assignments" => "materialassignments",
                "excitationcases" | "cases" => "excitationcases",
                other => other,
            };
            if !seen.insert(canonical.to_string()) {
                return Err(builtin_error(
                    STUDY_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.study option `{canonical}`"),
                ));
            }
            match key.as_str() {
                "template" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.template = Some(parse_scalar_enum(&text, "Template")?);
                }
                "backend" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.backend = Some(parse_scalar_enum(&text, "Backend")?);
                }
                "requirement" | "formulationrequirement" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.requirement = Some(parse_scalar_enum(&text, "FormulationRequirement")?);
                }
                "modelid" => {
                    options.model_id = Some(scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?);
                }
                "model" => {
                    options.model = Some(model_from_value(STUDY_NAME, &pair[1])?);
                }
                "mesh" => {
                    options.mesh = Some(crate::builtins::meshing::execution_output_from_value(
                        STUDY_NAME, &pair[1],
                    )?);
                }
                "frame" => {
                    let text = scalar_string(&pair[1], STUDY_NAME, &ERROR_INPUT)?;
                    options.frame = Some(parse_scalar_enum(&text, "Frame")?);
                }
                "defaults" => {
                    options.model_defaults = parse_model_defaults_mode(&scalar_string(
                        &pair[1],
                        STUDY_NAME,
                        &ERROR_INPUT,
                    )?)?;
                }
                "materials" => options.materials = material_vec_from_value(STUDY_NAME, &pair[1])?,
                "materialassignments" | "assignments" => {
                    options.material_assignments =
                        material_assignment_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "constraints" => {
                    options.constraints = constraint_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "boundaries" => {
                    options.boundaries = boundary_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "signals" => {
                    options.signals = typed_vec_from_value(STUDY_NAME, &pair[1], "signals")?;
                }
                "excitations" => {
                    options.excitations =
                        typed_vec_from_value(STUDY_NAME, &pair[1], "excitations")?;
                }
                "excitationcases" | "cases" => {
                    options.excitation_cases =
                        typed_vec_from_value(STUDY_NAME, &pair[1], "excitation cases")?;
                }
                "steps" => options.steps = step_vec_from_value(STUDY_NAME, &pair[1])?,
                "domains" => options.domains = domain_vec_from_value(STUDY_NAME, &pair[1])?,
                "interfaces" => {
                    options.interfaces = interface_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                other => {
                    return Err(builtin_error(
                        STUDY_NAME,
                        &ERROR_INPUT,
                        format!("unsupported fea.study option `{other}`"),
                    ));
                }
            }
        }
        Ok(options)
    }

    fn has_model_components(&self) -> bool {
        self.frame.is_some()
            || !self.materials.is_empty()
            || !self.material_assignments.is_empty()
            || !self.constraints.is_empty()
            || !self.boundaries.is_empty()
            || !self.signals.is_empty()
            || !self.excitations.is_empty()
            || !self.excitation_cases.is_empty()
            || !self.steps.is_empty()
            || !self.domains.is_empty()
            || !self.interfaces.is_empty()
    }
}

#[derive(Debug, Default)]
struct ModelConstructorOptions {
    template: Option<AnalysisModelTemplate>,
    frame: Option<CoordinateFrame>,
    defaults: ModelDefaultsMode,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    constraints: Vec<BoundConstraintPayload>,
    boundaries: Vec<GeometryBoundBoundary>,
    signals: Vec<PhysicalSignal>,
    excitations: Vec<ExcitationAction>,
    excitation_cases: Vec<ExcitationCase>,
    steps: Vec<ProcedureStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
}

pub(super) fn create_sweep_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            SWEEP_NAME,
            &ERROR_INPUT,
            "fea.sweep requires id and studies arguments",
        ));
    }
    let sweep_id = scalar_string(&args[0], SWEEP_NAME, &ERROR_INPUT)?;
    let studies = study_vec_from_value(SWEEP_NAME, &args[1])?;
    let mut fail_fast = true;
    let mut fail_fast_seen = false;
    for pair in expect_name_value_tail(SWEEP_NAME, &args[2..])? {
        match pair.key.as_str() {
            "failfast" => {
                if fail_fast_seen {
                    return Err(builtin_error(
                        SWEEP_NAME,
                        &ERROR_INPUT,
                        "duplicate fea.sweep option `failfast`",
                    ));
                }
                fail_fast_seen = true;
                fail_fast = logical_from_value(SWEEP_NAME, pair.value)?;
            }
            other => {
                return Err(builtin_error(
                    SWEEP_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.sweep option `{other}`"),
                ));
            }
        }
    }
    sweep_to_object(AnalysisStudySweepSpec {
        sweep_id,
        studies,
        fail_fast,
    })
}

pub(super) fn create_model_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires id and geometry arguments",
        ));
    }
    let model_id = scalar_string(&args[0], MODEL_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(MODEL_NAME, &args[1])?;
    let options = parse_model_constructor_options(MODEL_NAME, &args[2..])?;
    let template = options.template.ok_or_else(|| {
        builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires Template; choose a model template from fea.capabilities().modelTemplates",
        )
    })?;
    let model = build_model_from_parts(
        MODEL_NAME,
        &geometry,
        model_id,
        template,
        options.defaults,
        options.frame,
        options.materials,
        options.material_assignments,
        options.constraints,
        options.boundaries,
        options.signals,
        options.excitations,
        options.excitation_cases,
        options.steps,
        options.domains,
        options.interfaces,
    )?;
    serializable_to_object_preserving_integers(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_MODEL_CLASS,
        &model,
        &[],
        &["geometry_revision", "revision"],
    )
}

fn parse_model_constructor_options(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<ModelConstructorOptions> {
    let mut options = ModelConstructorOptions::default();
    for pair in expect_name_value_tail(builtin, args)? {
        match pair.key.as_str() {
            "template" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.template = Some(parse_scalar_enum(&text, "Template")?);
            }
            "frame" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.frame = Some(parse_scalar_enum(&text, "Frame")?);
            }
            "defaults" => {
                options.defaults =
                    parse_model_defaults_mode(&scalar_string(pair.value, builtin, &ERROR_INPUT)?)?;
            }
            "materials" => options.materials = material_vec_from_value(builtin, pair.value)?,
            "materialassignments" | "assignments" => {
                options.material_assignments =
                    material_assignment_vec_from_value(builtin, pair.value)?;
            }
            "constraints" => {
                options.constraints = constraint_vec_from_value(builtin, pair.value)?;
            }
            "boundaries" => {
                options.boundaries = boundary_vec_from_value(builtin, pair.value)?;
            }
            "signals" => options.signals = typed_vec_from_value(builtin, pair.value, "signals")?,
            "excitations" => {
                options.excitations = typed_vec_from_value(builtin, pair.value, "excitations")?
            }
            "excitationcases" | "cases" => {
                options.excitation_cases =
                    typed_vec_from_value(builtin, pair.value, "excitation cases")?
            }
            "steps" => options.steps = step_vec_from_value(builtin, pair.value)?,
            "domains" => options.domains = domain_vec_from_value(builtin, pair.value)?,
            "interfaces" => options.interfaces = interface_vec_from_value(builtin, pair.value)?,
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported {builtin} option `{other}`"),
                ));
            }
        }
    }
    Ok(options)
}

pub(super) fn create_material_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            "fea.material requires a material id",
        ));
    }
    let material_id = scalar_string(&args[0], MATERIAL_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(MATERIAL_NAME, &args[1..])?;
    let name = fields
        .remove("name")
        .map(json_to_string)
        .transpose()?
        .unwrap_or_else(|| material_id.clone());
    let mechanical = if let Some(value) = fields.remove("mechanical") {
        json_deserialize(MATERIAL_NAME, value, "mechanical material model")?
    } else {
        let youngs = remove_required_f64(&mut fields, MATERIAL_NAME, "youngs_modulus_pa")?;
        let poisson = remove_required_f64(&mut fields, MATERIAL_NAME, "poisson_ratio")?;
        let density =
            remove_optional_f64(&mut fields, MATERIAL_NAME, "density_kg_per_m3")?.unwrap_or(7850.0);
        MaterialMechanicalModel {
            youngs_modulus_pa: youngs,
            poisson_ratio: poisson,
            density_kg_per_m3: density,
        }
    };
    let thermal = if let Some(value) = fields.remove("thermal") {
        json_deserialize(MATERIAL_NAME, value, "thermal material model")?
    } else {
        let mut thermal = serde_json::to_value(MaterialThermalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        move_known_fields(
            &mut fields,
            thermal.as_object_mut().expect("thermal model is object"),
            &[
                "reference_temperature_k",
                "modulus_temp_coeff_per_k",
                "conductivity_w_per_mk",
                "specific_heat_j_per_kgk",
                "expansion_coefficient_per_k",
            ],
        );
        json_deserialize(MATERIAL_NAME, thermal, "thermal material model")?
    };
    let electrical = if let Some(value) = fields.remove("electrical") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "electrical material model",
        )?)
    } else {
        let mut electrical = serde_json::to_value(MaterialElectricalModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            electrical
                .as_object_mut()
                .expect("electrical material model is object"),
            &[
                "reference_temperature_k",
                "conductivity_s_per_m",
                "resistive_heating_coefficient",
                "relative_permittivity",
                "relative_permeability",
                "conductivity_frequency_response",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                electrical,
                "electrical material model",
            )?)
        } else {
            None
        }
    };
    let acoustic = if let Some(value) = fields.remove("acoustic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "acoustic material model",
        )?)
    } else {
        let mut acoustic = serde_json::to_value(MaterialAcousticModel::default())
            .map_err(|err| builtin_error(MATERIAL_NAME, &ERROR_INTERNAL, err.to_string()))?;
        let moved = move_known_fields(
            &mut fields,
            acoustic
                .as_object_mut()
                .expect("acoustic material model is object"),
            &[
                "density_kg_per_m3",
                "speed_of_sound_m_per_s",
                "damping_ratio",
            ],
        );
        if moved {
            Some(json_deserialize(
                MATERIAL_NAME,
                acoustic,
                "acoustic material model",
            )?)
        } else {
            None
        }
    };
    let plastic = if let Some(value) = fields.remove("plastic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "plastic material model",
        )?)
    } else if fields.contains_key("yield_strain")
        || fields.contains_key("hardening_modulus_ratio")
        || fields.contains_key("saturation_exponent")
    {
        Some(MaterialPlasticModel {
            yield_strain: remove_required_f64(&mut fields, MATERIAL_NAME, "yield_strain")?,
            hardening_modulus_ratio: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "hardening_modulus_ratio",
            )?,
            saturation_exponent: remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "saturation_exponent",
            )?,
        })
    } else {
        None
    };
    reject_unknown_fields(MATERIAL_NAME, fields)?;
    material_to_object(MaterialModel {
        material_id,
        name,
        mechanical,
        thermal,
        acoustic,
        electrical,
        plastic,
    })
}

pub(super) fn create_material_assignment_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MATERIAL_ASSIGNMENT_NAME,
            &ERROR_INPUT,
            "fea.materialAssignment requires region and material arguments",
        ));
    }
    let region_id = scalar_string(&args[0], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let assigned_material_id = scalar_string(&args[1], MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
    let mut expected_material_id = assigned_material_id.clone();
    let mut confidence = EvidenceConfidence::Verified;
    for pair in expect_name_value_tail(MATERIAL_ASSIGNMENT_NAME, &args[2..])? {
        match pair.key.as_str() {
            "expectedmaterial" | "expectedmaterialid" => {
                expected_material_id =
                    scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
            }
            "confidence" => {
                let text = scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
                confidence = parse_scalar_enum(&text, "Confidence")?;
            }
            other => {
                return Err(builtin_error(
                    MATERIAL_ASSIGNMENT_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.materialAssignment option `{other}`"),
                ));
            }
        }
    }
    material_assignment_to_object(MaterialAssignment {
        region_id,
        expected_material_id,
        assigned_material_id,
        confidence,
    })
}

pub(super) fn create_domain_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            DOMAIN_NAME,
            &ERROR_INPUT,
            "fea.domain requires a domain kind",
        ));
    }
    let kind_text = scalar_string(&args[0], DOMAIN_NAME, &ERROR_INPUT)?;
    let kind = normalize_token(&kind_text);
    let fields = json_fields_from_name_values(DOMAIN_NAME, &args[1..])?;
    let payload = match kind.as_str() {
        "thermomechanical" => DomainPayload {
            kind: "thermo_mechanical".to_string(),
            data: typed_domain_data::<ThermoMechanicalDomain>(
                DOMAIN_NAME,
                "thermo_mechanical domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_temperature_delta_k": 0.0,
                        "field_artifact_id": null,
                        "field_source": null,
                        "region_temperature_deltas": [],
                        "temperature_signal_id": null
                    }),
                    fields,
                    "thermo_mechanical domain",
                )?,
            )?,
        },
        "electrothermal" => DomainPayload {
            kind: "electro_thermal".to_string(),
            data: typed_domain_data::<ElectroThermalDomain>(
                DOMAIN_NAME,
                "electro_thermal domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_temperature_k": 293.15,
                        "applied_voltage_v": 0.0,
                        "region_conductivity_scales": [],
                        "electric_potential_signal_id": null
                    }),
                    fields,
                    "electro_thermal domain",
                )?,
            )?,
        },
        "electromagnetic" => DomainPayload {
            kind: "electromagnetic".to_string(),
            data: typed_domain_data::<ElectromagneticDomain>(
                DOMAIN_NAME,
                "electromagnetic domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "reference_frequency_hz": 0.0,
                        "applied_current_a": 0.0
                    }),
                    fields,
                    "electromagnetic domain",
                )?,
            )?,
        },
        "cfd" => DomainPayload {
            kind: "cfd".to_string(),
            data: typed_domain_data::<CfdDomain>(
                DOMAIN_NAME,
                "cfd domain",
                json_with_overrides(
                    DOMAIN_NAME,
                    serde_json::json!({
                        "enabled": true,
                        "solve_family": "steady_state",
                        "reference_density_kg_per_m3": 1.225,
                        "dynamic_viscosity_pa_s": 1.8e-5,
                        "inlet_velocity_m_per_s": 0.0,
                        "turbulence_intensity": 0.0,
                        "inlet_velocity_signal_id": null
                    }),
                    fields,
                    "cfd domain",
                )?,
            )?,
        },
        other => {
            return Err(builtin_error(
                DOMAIN_NAME,
                &ERROR_INPUT,
                format!("unsupported FEA domain kind `{other}`"),
            ));
        }
    };
    domain_to_object(payload)
}

pub(super) fn create_interface_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            INTERFACE_NAME,
            &ERROR_INPUT,
            "fea.interface requires id, primary region, and secondary region arguments",
        ));
    }
    let interface_id = scalar_string(&args[0], INTERFACE_NAME, &ERROR_INPUT)?;
    let primary_region_id = scalar_string(&args[1], INTERFACE_NAME, &ERROR_INPUT)?;
    let secondary_region_id = scalar_string(&args[2], INTERFACE_NAME, &ERROR_INPUT)?;
    let mut kind = "contact".to_string();
    let mut kind_seen = false;
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(INTERFACE_NAME, &args[3..])? {
        if pair.key == "kind" {
            if kind_seen {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    "duplicate fea.interface option `kind`",
                ));
            }
            kind_seen = true;
            kind = scalar_string(pair.value, INTERFACE_NAME, &ERROR_INPUT)?;
        } else {
            let key =
                canonical_field_name(&scalar_string(pair.name, INTERFACE_NAME, &ERROR_INPUT)?);
            if fields
                .insert(key.clone(), value_to_json(INTERFACE_NAME, pair.value)?)
                .is_some()
            {
                return Err(builtin_error(
                    INTERFACE_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.interface option `{key}`"),
                ));
            }
        }
    }
    let kind = match normalize_token(&kind).as_str() {
        "contact" => AnalysisInterfaceKind::Contact(json_deserialize(
            INTERFACE_NAME,
            json_with_overrides(
                INTERFACE_NAME,
                serde_json::json!({
                    "penalty_stiffness_scale": 1.0,
                    "max_penetration_ratio": 0.0,
                    "friction_coefficient": 0.0
                }),
                fields,
                "contact interface",
            )?,
            "contact interface",
        )?),
        "fluid_structure" | "fluidstructure" | "fsi" => {
            AnalysisInterfaceKind::FluidStructure(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "normal_stiffness_pa_per_m": 1.0e9,
                        "damping_ratio": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "fluid-structure interface",
                )?,
                "fluid-structure interface",
            )?)
        }
        "conjugate_heat_transfer" | "conjugateheattransfer" | "cht" => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(json_deserialize(
                INTERFACE_NAME,
                json_with_overrides(
                    INTERFACE_NAME,
                    serde_json::json!({
                        "thermal_conductance_w_per_m2k": 500.0,
                        "contact_resistance_m2k_per_w": 0.0,
                        "relaxation_factor": 0.5
                    }),
                    fields,
                    "conjugate heat-transfer interface",
                )?,
                "conjugate heat-transfer interface",
            )?)
        }
        other => {
            return Err(builtin_error(
                INTERFACE_NAME,
                &ERROR_INPUT,
                format!("unsupported interface kind `{other}`"),
            ));
        }
    };
    interface_to_object(AnalysisInterface {
        interface_id,
        primary_region_id,
        secondary_region_id,
        kind,
    })
}
fn build_model_from_parts(
    builtin: &'static str,
    geometry: &GeometryAsset,
    model_id: String,
    template: AnalysisModelTemplate,
    defaults: ModelDefaultsMode,
    frame: Option<CoordinateFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    constraints: Vec<BoundConstraintPayload>,
    boundaries: Vec<GeometryBoundBoundary>,
    signals: Vec<PhysicalSignal>,
    excitations: Vec<ExcitationAction>,
    excitation_cases: Vec<ExcitationCase>,
    steps: Vec<ProcedureStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
) -> BuiltinResult<AnalysisModel> {
    let mut model = match defaults {
        ModelDefaultsMode::TemplateScaffold => analysis_create_model_op(
            geometry,
            AnalysisCreateModelIntentSpec {
                model_id: model_id.clone(),
                template,
            },
            OperationContext::new(None, None),
        )
        .map(|envelope| envelope.data)
        .map_err(|err| operation_error(builtin, &ERROR_OPERATION, err))?,
        ModelDefaultsMode::None => empty_model(model_id, geometry),
    };

    if let Some(frame) = frame {
        model.frame = frame;
    }
    if !materials.is_empty() {
        model.materials = materials;
    }
    if !material_assignments.is_empty() {
        model.material_assignments = material_assignments
            .into_iter()
            .map(|mut assignment| {
                assignment.region_id =
                    resolve_region_selector(builtin, &assignment.region_id, geometry)?;
                Ok(assignment)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !constraints.is_empty() {
        model.constraints.clear();
        for constraint in constraints {
            match constraint {
                BoundConstraintPayload::BoundStructural(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Structural(bound.constraint));
                }
                BoundConstraintPayload::BoundThermal(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Thermal(bound.constraint));
                }
                BoundConstraintPayload::BoundElectrical(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Electrical(bound.constraint));
                }
                BoundConstraintPayload::BoundMagnetic(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Magnetic(bound.constraint));
                }
                BoundConstraintPayload::BoundAcoustic(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Acoustic(bound.constraint));
                }
                BoundConstraintPayload::BoundFluid(bound) => {
                    bound.require_geometry(builtin, geometry)?;
                    model
                        .constraints
                        .push(AnalysisConstraint::Fluid(bound.constraint));
                }
            }
        }
    }
    if !boundaries.is_empty() {
        model.boundaries.clear();
        for boundary in boundaries {
            boundary.require_geometry(builtin, geometry)?;
            model.boundaries.push(boundary.boundary);
        }
    }
    if !signals.is_empty() {
        model.signals = signals;
    }
    if !excitations.is_empty() {
        model.excitations = excitations
            .into_iter()
            .map(|mut action| {
                if let ExcitationTarget::Region(target) = &mut action.excitation_mut().target {
                    target.region_id =
                        resolve_region_selector(builtin, &target.region_id, geometry)?;
                }
                Ok(action)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !excitation_cases.is_empty() {
        model.excitation_cases = excitation_cases;
    }
    if !steps.is_empty() {
        model.sequence =
            AnalysisSequence::from_steps(format!("{}.sequence", model.model_id.0), steps);
    }
    for domain in domains {
        match domain.kind.as_str() {
            "thermo_mechanical" => {
                let mut domain: ThermoMechanicalDomain =
                    json_deserialize(builtin, domain.data, "thermo_mechanical domain")?;
                for entry in &mut domain.region_temperature_deltas {
                    entry.region_id = resolve_region_selector(builtin, &entry.region_id, geometry)?;
                }
                if let Some(source) = &mut domain.field_source {
                    for region_id in &mut source.expected_region_ids {
                        *region_id = resolve_region_selector(builtin, region_id, geometry)?;
                    }
                }
                model.thermo_mechanical = Some(domain);
            }
            "electro_thermal" => {
                let mut domain: ElectroThermalDomain =
                    json_deserialize(builtin, domain.data, "electro_thermal domain")?;
                for entry in &mut domain.region_conductivity_scales {
                    entry.region_id = resolve_region_selector(builtin, &entry.region_id, geometry)?;
                }
                model.electro_thermal = Some(domain);
            }
            "electromagnetic" => {
                model.electromagnetic = Some(json_deserialize(
                    builtin,
                    domain.data,
                    "electromagnetic domain",
                )?);
            }
            "cfd" => {
                model.cfd = Some(json_deserialize(builtin, domain.data, "cfd domain")?);
            }
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported domain payload `{other}`"),
                ));
            }
        }
    }
    if !interfaces.is_empty() {
        model.interfaces = interfaces
            .into_iter()
            .map(|mut interface| {
                interface.primary_region_id =
                    resolve_region_selector(builtin, &interface.primary_region_id, geometry)?;
                interface.secondary_region_id =
                    resolve_region_selector(builtin, &interface.secondary_region_id, geometry)?;
                Ok(interface)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    Ok(model)
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id.clone()),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: CoordinateFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
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
        sequence: AnalysisSequence::from_steps(format!("{model_id}.sequence"), Vec::new()),
    }
}

fn resolve_region_selector(
    builtin: &'static str,
    selector: &str,
    geometry: &GeometryAsset,
) -> BuiltinResult<String> {
    if let Some(id) = selector
        .strip_prefix("id:")
        .or_else(|| selector.strip_prefix("region:"))
    {
        return require_region_id(builtin, id, geometry);
    }
    if let Some(tag) = selector.strip_prefix("tag:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.tag.as_deref() == Some(tag))
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("region tag `{tag}` was not found in geometry"),
                )
            });
    }
    if let Some(name) = selector.strip_prefix("name:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.name == name)
            .map(|region| region.region_id.clone())
            .ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("region name `{name}` was not found in geometry"),
                )
            });
    }
    require_region_id(builtin, selector, geometry)
}

fn require_region_id(
    builtin: &'static str,
    region_id: &str,
    geometry: &GeometryAsset,
) -> BuiltinResult<String> {
    geometry
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .map(|region| region.region_id.clone())
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("region id `{region_id}` was not found in geometry"),
            )
        })
}

fn material_to_object(material: MaterialModel) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_CLASS,
        &material,
    )
}

fn material_assignment_to_object(assignment: MaterialAssignment) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_ASSIGNMENT_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_ASSIGNMENT_CLASS,
        &assignment,
    )
}

fn domain_to_object(domain: DomainPayload) -> BuiltinResult<Value> {
    serializable_to_object_preserving_integers(
        DOMAIN_NAME,
        &ERROR_INTERNAL,
        FEA_DOMAIN_CLASS,
        &domain,
        &[],
        &["revision"],
    )
}

fn interface_to_object(interface: AnalysisInterface) -> BuiltinResult<Value> {
    serializable_to_object(
        INTERFACE_NAME,
        &ERROR_INTERNAL,
        FEA_INTERFACE_CLASS,
        &interface,
    )
}

fn model_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<AnalysisModel> {
    object_payload(builtin, value, FEA_MODEL_CLASS)
}

fn study_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisStudySpec>> {
    object_vec_from_value(builtin, value, FEA_STUDY_CLASS)
}

fn material_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialModel>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_CLASS)
}

fn material_assignment_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<MaterialAssignment>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_ASSIGNMENT_CLASS)
}

fn constraint_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<BoundConstraintPayload>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected a typed FEA constraint object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::Object(object) if object.class_name == FEA_STRUCTURAL_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_STRUCTURAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundStructural)
            }
            Value::Object(object) if object.class_name == FEA_THERMAL_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_THERMAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundThermal)
            }
            Value::Object(object) if object.class_name == FEA_ELECTRICAL_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_ELECTRICAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundElectrical)
            }
            Value::Object(object) if object.class_name == FEA_MAGNETIC_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_MAGNETIC_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundMagnetic)
            }
            Value::Object(object) if object.class_name == FEA_ACOUSTIC_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_ACOUSTIC_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundAcoustic)
            }
            Value::Object(object) if object.class_name == FEA_FLUID_CONSTRAINT_CLASS => {
                object_payload(builtin, value, FEA_FLUID_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundFluid)
            }
            Value::Object(object) => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported FEA constraint class `{}`", object.class_name),
            )),
            _ => unreachable!("constraint inputs were classified above"),
        })
        .collect()
}

fn boundary_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<GeometryBoundBoundary>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected a typed FEA boundary object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::Object(object) if object.class_name == FEA_STRUCTURAL_BOUNDARY_CLASS => {
                bound_boundary_payload(builtin, value, FEA_STRUCTURAL_BOUNDARY_CLASS)
            }
            Value::Object(object) if object.class_name == FEA_THERMAL_BOUNDARY_CLASS => {
                bound_boundary_payload(builtin, value, FEA_THERMAL_BOUNDARY_CLASS)
            }
            Value::Object(object) if object.class_name == FEA_MAGNETIC_BOUNDARY_CLASS => {
                bound_boundary_payload(builtin, value, FEA_MAGNETIC_BOUNDARY_CLASS)
            }
            Value::Object(object) if object.class_name == FEA_ACOUSTIC_BOUNDARY_CLASS => {
                bound_boundary_payload(builtin, value, FEA_ACOUSTIC_BOUNDARY_CLASS)
            }
            Value::Object(object) if object.class_name == FEA_FLUID_BOUNDARY_CLASS => {
                bound_boundary_payload(builtin, value, FEA_FLUID_BOUNDARY_CLASS)
            }
            Value::Object(object) => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported FEA boundary class `{}`", object.class_name),
            )),
            _ => unreachable!("boundary inputs were classified above"),
        })
        .collect()
}

fn bound_boundary_payload(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
) -> BuiltinResult<GeometryBoundBoundary> {
    let boundary: GeometryBoundBoundary = object_payload(builtin, value, class_name)?;
    boundary.require_class(builtin, class_name)?;
    Ok(boundary)
}

fn typed_vec_from_value<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| json_deserialize(builtin, value_to_json(builtin, item)?, label))
            .collect(),
        _ => Ok(vec![json_deserialize(
            builtin,
            value_to_json(builtin, value)?,
            label,
        )?]),
    }
}

fn step_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<ProcedureStep>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected a typed FEA step object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| {
            let Value::Object(object) = value else {
                unreachable!("step inputs were classified above")
            };
            let class_name = match object.class_name.as_str() {
                FEA_STRUCTURAL_STEP_CLASS => FEA_STRUCTURAL_STEP_CLASS,
                FEA_THERMAL_STEP_CLASS => FEA_THERMAL_STEP_CLASS,
                FEA_ELECTROMAGNETIC_STEP_CLASS => FEA_ELECTROMAGNETIC_STEP_CLASS,
                FEA_ACOUSTIC_STEP_CLASS => FEA_ACOUSTIC_STEP_CLASS,
                FEA_FLUID_STEP_CLASS => FEA_FLUID_STEP_CLASS,
                FEA_THERMOMECHANICAL_STEP_CLASS => FEA_THERMOMECHANICAL_STEP_CLASS,
                FEA_ELECTROTHERMAL_STEP_CLASS => FEA_ELECTROTHERMAL_STEP_CLASS,
                FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS => FEA_CONJUGATE_HEAT_TRANSFER_STEP_CLASS,
                FEA_FLUID_STRUCTURE_STEP_CLASS => FEA_FLUID_STRUCTURE_STEP_CLASS,
                other => {
                    return Err(builtin_error(
                        builtin,
                        &ERROR_INPUT,
                        format!("unsupported FEA step class `{other}`"),
                    ))
                }
            };
            let step: ProcedureStep = object_payload(builtin, value, class_name)?;
            require_step_class(builtin, class_name, &step)?;
            Ok(step)
        })
        .collect()
}

fn domain_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<DomainPayload>> {
    object_vec_from_value(builtin, value, FEA_DOMAIN_CLASS)
}

fn interface_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<AnalysisInterface>> {
    object_vec_from_value(builtin, value, FEA_INTERFACE_CLASS)
}

fn object_vec_from_value<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    expected_class: &'static str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| object_payload(builtin, item, expected_class))
            .collect(),
        Value::Object(_) => Ok(vec![object_payload(builtin, value, expected_class)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {expected_class} object or cell array; got {other:?}"),
        )),
    }
}
fn parse_model_defaults_mode(text: &str) -> BuiltinResult<ModelDefaultsMode> {
    match normalize_token(text).as_str() {
        "templatescaffold" | "scaffold" | "template" => Ok(ModelDefaultsMode::TemplateScaffold),
        "none" | "empty" => Ok(ModelDefaultsMode::None),
        other => Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            format!("unsupported model defaults mode `{other}`"),
        )),
    }
}

fn resolve_study_template(
    options: &StudyConstructorOptions,
) -> BuiltinResult<AnalysisModelTemplate> {
    options.template.ok_or_else(|| {
        builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires Template; choose a model template from fea.capabilities().modelTemplates",
        )
    })
}
