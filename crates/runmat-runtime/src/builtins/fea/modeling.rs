use super::*;

#[derive(Debug, Clone)]
pub(super) enum DomainPayload {
    ThermoMechanical(ThermoMechanicalDomain),
    ElectroThermal(ElectroThermalDomain),
    Electromagnetic(ElectromagneticDomain),
    Fluid(CfdDomain),
}

impl DomainPayload {
    fn family_name(&self) -> &'static str {
        match self {
            Self::ThermoMechanical(_) => "thermo-mechanical",
            Self::ElectroThermal(_) => "electro-thermal",
            Self::Electromagnetic(_) => "electromagnetic",
            Self::Fluid(_) => "fluid",
        }
    }
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
    pub(super) execution: Option<AnalysisExecutionPlan>,
    pub(super) model_id: Option<String>,
    pub(super) model: Option<AnalysisModel>,
    pub(super) frame: Option<CoordinateFrame>,
    pub(super) model_defaults: ModelDefaultsMode,
    pub(super) materials: Vec<MaterialModel>,
    pub(super) material_assignments: Vec<MaterialAssignment>,
    pub(super) constraints: Vec<BoundConstraintPayload>,
    pub(super) boundaries: Vec<GeometryBoundBoundary>,
    pub(super) signals: Vec<PhysicalSignal>,
    pub(super) excitations: Vec<super::excitations::GeometryBoundExcitation>,
    pub(super) excitation_cases: Vec<ExcitationCase>,
    pub(super) initial_conditions: Vec<super::initial_conditions::GeometryBoundInitialCondition>,
    pub(super) response_spectra: Vec<runmat_analysis_core::ResponseSpectrum>,
    pub(super) steps: Vec<ProcedureStep>,
    pub(super) sequence: Option<AnalysisSequence>,
    pub(super) domains: Vec<DomainPayload>,
    pub(super) interfaces: Vec<GeometryBoundInterface>,
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
    let supplied_execution = options.execution.clone();
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
            options.initial_conditions,
            options.response_spectra,
            options.steps,
            options.sequence,
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
    let execution = match supplied_execution {
        Some(execution) => {
            execution
                .validate_against(&model.sequence)
                .map_err(|message| builtin_error(STUDY_NAME, &ERROR_INPUT, message))?;
            execution
        }
        None => AnalysisExecutionPlan::for_sequence(
            &model.sequence,
            runmat_analysis_fea::ComputeBackend::Cpu,
        )
        .map_err(|message| builtin_error(STUDY_NAME, &ERROR_INPUT, message))?,
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
                "execution" => {
                    options.execution = Some(super::execution::execution_plan_from_value(
                        STUDY_NAME, &pair[1],
                    )?)
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
                    options.signals = signal_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "excitations" => {
                    options.excitations =
                        super::excitations::bound_excitation_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "excitationcases" | "cases" => {
                    options.excitation_cases =
                        super::excitations::excitation_case_vec_from_value(STUDY_NAME, &pair[1])?;
                }
                "initialconditions" => {
                    options.initial_conditions =
                        super::initial_conditions::initial_condition_vec_from_value(
                            STUDY_NAME, &pair[1],
                        )?;
                }
                "responsespectra" => {
                    options.response_spectra =
                        super::response_spectra::response_spectrum_vec_from_value(
                            STUDY_NAME, &pair[1],
                        )?;
                }
                "steps" => options.steps = step_vec_from_value(STUDY_NAME, &pair[1])?,
                "sequence" => {
                    options.sequence =
                        Some(super::sequences::sequence_from_value(STUDY_NAME, &pair[1])?)
                }
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
            || !self.initial_conditions.is_empty()
            || !self.response_spectra.is_empty()
            || !self.steps.is_empty()
            || self.sequence.is_some()
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
    excitations: Vec<super::excitations::GeometryBoundExcitation>,
    excitation_cases: Vec<ExcitationCase>,
    initial_conditions: Vec<super::initial_conditions::GeometryBoundInitialCondition>,
    response_spectra: Vec<runmat_analysis_core::ResponseSpectrum>,
    steps: Vec<ProcedureStep>,
    sequence: Option<AnalysisSequence>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<GeometryBoundInterface>,
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
        options.initial_conditions,
        options.response_spectra,
        options.steps,
        options.sequence,
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
            "signals" => options.signals = signal_vec_from_value(builtin, pair.value)?,
            "excitations" => {
                options.excitations =
                    super::excitations::bound_excitation_vec_from_value(builtin, pair.value)?
            }
            "excitationcases" | "cases" => {
                options.excitation_cases =
                    super::excitations::excitation_case_vec_from_value(builtin, pair.value)?
            }
            "initialconditions" => {
                options.initial_conditions =
                    super::initial_conditions::initial_condition_vec_from_value(
                        builtin, pair.value,
                    )?
            }
            "responsespectra" => {
                options.response_spectra =
                    super::response_spectra::response_spectrum_vec_from_value(builtin, pair.value)?
            }
            "steps" => options.steps = step_vec_from_value(builtin, pair.value)?,
            "sequence" => {
                options.sequence = Some(super::sequences::sequence_from_value(builtin, pair.value)?)
            }
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
    excitations: Vec<super::excitations::GeometryBoundExcitation>,
    excitation_cases: Vec<ExcitationCase>,
    initial_conditions: Vec<super::initial_conditions::GeometryBoundInitialCondition>,
    response_spectra: Vec<runmat_analysis_core::ResponseSpectrum>,
    steps: Vec<ProcedureStep>,
    sequence: Option<AnalysisSequence>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<GeometryBoundInterface>,
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
            .map(|bound| {
                bound.require_geometry(builtin, geometry)?;
                Ok(bound.action)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !excitation_cases.is_empty() {
        model.excitation_cases = excitation_cases;
    }
    if !initial_conditions.is_empty() {
        model.initial_conditions = initial_conditions
            .into_iter()
            .map(|bound| {
                bound.require_geometry(builtin, geometry)?;
                Ok(bound.condition)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !response_spectra.is_empty() {
        model.response_spectra = response_spectra;
    }
    if sequence.is_some() && !steps.is_empty() {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "Sequence and Steps are mutually exclusive",
        ));
    }
    if let Some(sequence) = sequence {
        model.sequence = sequence;
    } else if !steps.is_empty() {
        model.sequence =
            AnalysisSequence::from_steps(format!("{}.sequence", model.model_id.0), steps);
    }
    for domain in domains {
        match domain {
            DomainPayload::ThermoMechanical(domain) => model.thermo_mechanical = Some(domain),
            DomainPayload::ElectroThermal(domain) => model.electro_thermal = Some(domain),
            DomainPayload::Electromagnetic(domain) => model.electromagnetic = Some(domain),
            DomainPayload::Fluid(domain) => model.cfd = Some(domain),
        }
    }
    if !interfaces.is_empty() {
        model.interfaces.clear();
        for interface in interfaces {
            interface.require_geometry(builtin, geometry)?;
            model.interfaces.push(interface.interface);
        }
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

fn signal_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<PhysicalSignal>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| super::signals::physical_signal_from_value(builtin, item))
            .collect(),
        Value::Object(_) => Ok(vec![super::signals::physical_signal_from_value(
            builtin, value,
        )?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected a typed FEA signal object or cell array; got {other:?}"),
        )),
    }
}

pub(super) fn step_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ProcedureStep>> {
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

pub(super) fn domain_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<DomainPayload>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected typed FEA domain object or cell array; got {other:?}"),
            ))
        }
    };
    let domains = values
        .into_iter()
        .map(|value| {
            let Value::Object(object) = value else {
                unreachable!("domain inputs were classified above")
            };
            match object.class_name.as_str() {
                FEA_THERMOMECHANICAL_DOMAIN_CLASS => {
                    object_payload(builtin, value, FEA_THERMOMECHANICAL_DOMAIN_CLASS)
                        .map(DomainPayload::ThermoMechanical)
                }
                FEA_ELECTROTHERMAL_DOMAIN_CLASS => {
                    object_payload(builtin, value, FEA_ELECTROTHERMAL_DOMAIN_CLASS)
                        .map(DomainPayload::ElectroThermal)
                }
                FEA_ELECTROMAGNETIC_DOMAIN_CLASS => {
                    object_payload(builtin, value, FEA_ELECTROMAGNETIC_DOMAIN_CLASS)
                        .map(DomainPayload::Electromagnetic)
                }
                FEA_FLUID_DOMAIN_CLASS => {
                    object_payload(builtin, value, FEA_FLUID_DOMAIN_CLASS).map(DomainPayload::Fluid)
                }
                other => Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported FEA domain class `{other}`"),
                )),
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let mut families = HashSet::new();
    for domain in &domains {
        if !families.insert(domain.family_name()) {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!(
                    "duplicate {} FEA domain; provide at most one domain per physics family",
                    domain.family_name()
                ),
            ));
        }
    }
    Ok(domains)
}

fn interface_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<GeometryBoundInterface>> {
    let values = match value {
        Value::Cell(cell) => cell.data.iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected typed FEA interface object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| {
            let Value::Object(object) = value else {
                unreachable!("interface inputs were classified above")
            };
            let class_name = match object.class_name.as_str() {
                FEA_CONTACT_INTERFACE_CLASS => FEA_CONTACT_INTERFACE_CLASS,
                FEA_FLUID_STRUCTURE_INTERFACE_CLASS => FEA_FLUID_STRUCTURE_INTERFACE_CLASS,
                FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS => {
                    FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_CLASS
                }
                other => {
                    return Err(builtin_error(
                        builtin,
                        &ERROR_INPUT,
                        format!("unsupported FEA interface class `{other}`"),
                    ))
                }
            };
            let interface: GeometryBoundInterface = object_payload(builtin, value, class_name)?;
            interface.require_class(builtin, class_name)?;
            Ok(interface)
        })
        .collect()
}

pub(super) fn object_vec_from_value<T: DeserializeOwned>(
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
