use runmat_builtins::fea_contract::options::{
    FeaMaterialAssignmentOption, FeaModelOption, FeaStudyOption, FeaSweepOption,
};

use super::signals::bindings::{
    FrozenGeneratorInventory, FrozenStudyGeneratorInventory, FROZEN_GENERATORS_PROPERTY,
    FROZEN_STUDY_GENERATORS_PROPERTY,
};
use super::*;
use runmat_analysis_core::{
    validate_model_against_geometry, MaterialElasticityModel, PropertyRegionTarget, ThermalDomain,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundPropertyRegionTarget {
    geometry_id: String,
    geometry_revision: u32,
    target: PropertyRegionTarget,
}

#[cfg(test)]
impl GeometryBoundPropertyRegionTarget {
    pub(super) const fn target(&self) -> &PropertyRegionTarget {
        &self.target
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GeometryBoundMaterialAssignment {
    geometry_id: String,
    geometry_revision: u32,
    target: PropertyRegionTarget,
    expected_material_id: String,
    assigned_material_id: String,
    confidence: EvidenceConfidence,
}

impl GeometryBoundMaterialAssignment {
    fn require_geometry(
        &self,
        builtin: &'static str,
        geometry: &GeometryAsset,
    ) -> BuiltinResult<()> {
        crate::builtins::fea::constraints::require_geometry_binding(
            builtin,
            "material assignment",
            &self.geometry_id,
            self.geometry_revision,
            geometry,
        )
    }

    fn into_assignment(self) -> MaterialAssignment {
        MaterialAssignment {
            target: self.target,
            expected_material_id: self.expected_material_id,
            assigned_material_id: self.assigned_material_id,
            confidence: self.confidence,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum DomainPayload {
    Thermal(ThermalDomain),
    ThermoMechanical(ThermoMechanicalDomain),
    ElectroThermal(ElectroThermalDomain),
    Electromagnetic(ElectromagneticDomain),
    Fluid(CfdDomain),
}

impl DomainPayload {
    fn family_name(&self) -> &'static str {
        match self {
            Self::Thermal(_) => "thermal",
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

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComposedModel {
    pub(super) model: AnalysisModel,
    pub(super) frozen_generators: FrozenGeneratorInventory,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComposedStudy {
    pub(super) study: AnalysisStudySpec,
    pub(super) frozen_generators: FrozenGeneratorInventory,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComposedSweep {
    pub(super) sweep: AnalysisStudySweepSpec,
    pub(super) frozen_generators: FrozenStudyGeneratorInventory,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct RuntimeSignals {
    signals: Vec<PhysicalSignal>,
    frozen_generators: FrozenGeneratorInventory,
}

#[derive(Debug, Default)]
pub(super) struct StudyConstructorOptions {
    pub(super) execution: Option<AnalysisExecutionPlan>,
    pub(super) mesh: Option<crate::meshing::MeshingExecutionOutput>,
}

pub(super) fn create_study_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires id, geometry, and model arguments",
        ));
    }
    let study_id = scalar_string(&args[0], STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(STUDY_NAME, &args[1])?;
    let composed_model = model_from_value(STUDY_NAME, &args[2])?;
    let options = StudyConstructorOptions::parse(args.view().range(3..)?)?;
    let supplied_mesh = options.mesh.clone();
    let supplied_execution = options.execution.clone();
    let (model, frozen_generators) = (composed_model.model, composed_model.frozen_generators);
    require_model_geometry(STUDY_NAME, &model, &geometry)?;
    validate_model_against_geometry(&model, &CoordinateFrame::Global).map_err(|error| {
        builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            format!("FEA model failed study validation: {error}"),
        )
    })?;
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
        model_source_evidence: None,
        model,
        execution,
        meshing_settings: None,
        immutable_values: Vec::new(),
        solver_mesh_artifact_path: persisted_mesh
            .as_ref()
            .and_then(|mesh| mesh.artifact_path.clone()),
        meshing_evidence_artifact_path: persisted_mesh
            .as_ref()
            .and_then(|mesh| mesh.evidence_path.clone()),
    };
    study_to_object_with_generators(spec, frozen_generators)
}

impl StudyConstructorOptions {
    pub(super) fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                "fea.study options must be Name, Value pairs",
            ));
        }
        let mut options = Self::default();
        let mut seen = HashSet::new();
        for pair in args.chunks_exact(2) {
            let key = option_key(&pair[0], STUDY_NAME)?;
            let canonical = key.as_str();
            if !seen.insert(canonical.to_string()) {
                return Err(builtin_error(
                    STUDY_NAME,
                    &ERROR_INPUT,
                    format!("duplicate fea.study option `{canonical}`"),
                ));
            }
            match FeaStudyOption::from_normalized(key.as_str()) {
                Some(FeaStudyOption::Execution) => {
                    options.execution = Some(super::execution::execution_plan_from_value(
                        STUDY_NAME, &pair[1],
                    )?)
                }
                Some(FeaStudyOption::Mesh) => {
                    options.mesh = Some(crate::builtins::meshing::execution_output_from_value(
                        STUDY_NAME, &pair[1],
                    )?);
                }
                None => {
                    return Err(builtin_error(
                        STUDY_NAME,
                        &ERROR_INPUT,
                        format!("unsupported fea.study option `{key}`"),
                    ));
                }
            }
        }
        Ok(options)
    }
}

#[derive(Debug, Default)]
struct ModelConstructorOptions {
    frame: Option<CoordinateFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<GeometryBoundMaterialAssignment>,
    constraints: Vec<BoundConstraintPayload>,
    boundaries: Vec<GeometryBoundBoundary>,
    signals: Vec<PhysicalSignal>,
    frozen_generators: FrozenGeneratorInventory,
    excitations: Vec<super::excitations::GeometryBoundExcitation>,
    excitation_cases: Vec<ExcitationCase>,
    initial_conditions: Vec<super::initial_conditions::GeometryBoundInitialCondition>,
    response_spectra: Vec<runmat_analysis_core::ResponseSpectrum>,
    steps: Vec<ProcedureStep>,
    sequence: Option<AnalysisSequence>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<GeometryBoundInterface>,
}

pub(super) fn create_sweep_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    for pair in expect_name_value_tail(SWEEP_NAME, args.view().range(2..)?)? {
        match FeaSweepOption::from_normalized(pair.key.as_str()) {
            Some(FeaSweepOption::FailFast) => {
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
            None => {
                return Err(builtin_error(
                    SWEEP_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.sweep option `{}`", pair.key),
                ));
            }
        }
    }
    let spec = AnalysisStudySweepSpec {
        sweep_id,
        studies: studies.iter().map(|study| study.study.clone()).collect(),
        fail_fast,
    };
    let inventory = FrozenStudyGeneratorInventory::new(
        SWEEP_NAME,
        &studies
            .into_iter()
            .map(|study| (study.study, study.frozen_generators))
            .collect::<Vec<_>>(),
    )?;
    sweep_to_object_with_generators(spec, inventory)
}

pub(super) fn create_model_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires id and geometry arguments",
        ));
    }
    let model_id = scalar_string(&args[0], MODEL_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(MODEL_NAME, &args[1])?;
    let options = parse_model_constructor_options(MODEL_NAME, args.view().range(2..)?)?;
    let model = build_model_from_parts(
        MODEL_NAME,
        &geometry,
        model_id,
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
    model_to_object(ComposedModel {
        model,
        frozen_generators: options.frozen_generators,
    })
}

#[cfg(test)]
pub(super) fn static_test_step_value() -> Value {
    serializable_to_object(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_STRUCTURAL_STEP_CLASS,
        &ProcedureStep::structural_static("explicit_test_step"),
    )
    .expect("static test step should serialize")
}

#[cfg(test)]
pub(super) fn static_test_steps_value() -> Value {
    Value::Cell(
        runmat_value::CellArray::new(vec![static_test_step_value()], 1, 1)
            .expect("static test step cell should build"),
    )
}

fn parse_model_constructor_options(
    builtin: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ModelConstructorOptions> {
    let mut options = ModelConstructorOptions::default();
    for pair in expect_name_value_tail(builtin, args)? {
        match FeaModelOption::from_normalized(pair.key.as_str()) {
            Some(FeaModelOption::Frame) => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.frame = Some(parse_scalar_enum_for_builtin(builtin, &text, "Frame")?);
            }
            Some(FeaModelOption::Materials) => {
                options.materials = material_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::MaterialAssignments) => {
                options.material_assignments =
                    material_assignment_vec_from_value(builtin, pair.value)?;
            }
            Some(FeaModelOption::Constraints) => {
                options.constraints = constraint_vec_from_value(builtin, pair.value)?;
            }
            Some(FeaModelOption::Boundaries) => {
                options.boundaries = boundary_vec_from_value(builtin, pair.value)?;
            }
            Some(FeaModelOption::Signals) => {
                let signals = signal_vec_from_value(builtin, pair.value)?;
                options.signals = signals.signals;
                options.frozen_generators = signals.frozen_generators;
            }
            Some(FeaModelOption::Excitations) => {
                options.excitations =
                    super::excitations::bound_excitation_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::ExcitationCases) => {
                options.excitation_cases =
                    super::excitations::excitation_case_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::InitialConditions) => {
                options.initial_conditions =
                    super::initial_conditions::initial_condition_vec_from_value(
                        builtin, pair.value,
                    )?
            }
            Some(FeaModelOption::ResponseSpectra) => {
                options.response_spectra =
                    super::response_spectra::response_spectrum_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::Steps) => {
                options.steps = step_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::Sequence) => {
                options.sequence = Some(super::sequences::sequence_from_value(builtin, pair.value)?)
            }
            Some(FeaModelOption::Domains) => {
                options.domains = domain_vec_from_value(builtin, pair.value)?
            }
            Some(FeaModelOption::Interfaces) => {
                options.interfaces = interface_vec_from_value(builtin, pair.value)?
            }
            None => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported {builtin} option `{}`", pair.key),
                ));
            }
        }
    }
    Ok(options)
}

pub(super) fn create_material_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            "fea.material requires a material id",
        ));
    }
    let material_id = scalar_string(&args[0], MATERIAL_NAME, &ERROR_INPUT)?;
    let mut fields = json_fields_from_name_values(MATERIAL_NAME, args.view().range(1..)?)?;
    let mut authored_damping = None;
    for pair in expect_name_value_tail(MATERIAL_NAME, args.view().range(1..)?)? {
        let raw_name = scalar_string(pair.name, MATERIAL_NAME, &ERROR_INPUT)?;
        if canonical_field_name(&raw_name) == "damping" {
            authored_damping = Some(super::damping::mechanical_damping_from_value(
                MATERIAL_NAME,
                pair.value,
            )?);
        }
    }
    let name = fields
        .remove("name")
        .map(json_to_string)
        .transpose()?
        .unwrap_or_else(|| material_id.clone());
    let mass_density_kg_per_m3 =
        remove_optional_f64(&mut fields, MATERIAL_NAME, "mass_density_kg_per_m3")?;
    fields.remove("damping");
    let damping = authored_damping;
    let mechanical = if let Some(value) = fields.remove("mechanical") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "mechanical material model",
        )?)
    } else if fields.contains_key("youngs_modulus_pa") || fields.contains_key("poisson_ratio") {
        Some(MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "youngs_modulus_pa",
            )?),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "poisson_ratio",
            )?),
        })
    } else {
        None
    };
    let thermal =
        if let Some(value) = fields.remove("thermal") {
            Some(json_deserialize(
                MATERIAL_NAME,
                value,
                "thermal material model",
            )?)
        } else if [
            "reference_temperature_k",
            "modulus_temp_coeff_per_k",
            "conductivity_w_per_mk",
            "specific_heat_j_per_kgk",
            "expansion_coefficient_per_k",
        ]
        .iter()
        .any(|field| fields.contains_key(*field))
        {
            Some(MaterialThermalModel {
                reference_temperature_k: runmat_types::physical::Temperature::new(
                    remove_required_f64(&mut fields, MATERIAL_NAME, "reference_temperature_k")?,
                ),
                modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(
                    remove_required_f64(&mut fields, MATERIAL_NAME, "modulus_temp_coeff_per_k")?,
                ),
                conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(
                    remove_required_f64(&mut fields, MATERIAL_NAME, "conductivity_w_per_mk")?,
                ),
                specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(
                    remove_required_f64(&mut fields, MATERIAL_NAME, "specific_heat_j_per_kgk")?,
                ),
                expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(
                    remove_required_f64(&mut fields, MATERIAL_NAME, "expansion_coefficient_per_k")?,
                ),
            })
        } else {
            None
        };
    let electrical = if let Some(value) = fields.remove("electrical") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "electrical material model",
        )?)
    } else if [
        "conductivity_s_per_m",
        "resistive_heating_coefficient",
        "relative_permittivity",
        "relative_permeability",
        "conductivity_frequency_response",
    ]
    .iter()
    .any(|field| fields.contains_key(*field))
    {
        let mut electrical = serde_json::Map::new();
        move_known_fields(
            &mut fields,
            &mut electrical,
            &[
                "reference_temperature_k",
                "conductivity_s_per_m",
                "resistive_heating_coefficient",
                "relative_permittivity",
                "relative_permeability",
                "conductivity_frequency_response",
            ],
        );
        Some(json_deserialize(
            MATERIAL_NAME,
            serde_json::Value::Object(electrical),
            "electrical material model",
        )?)
    } else {
        None
    };
    let acoustic = if let Some(value) = fields.remove("acoustic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "acoustic material model",
        )?)
    } else if fields.contains_key("speed_of_sound_m_per_s") || fields.contains_key("damping_ratio")
    {
        Some(MaterialAcousticModel {
            speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "speed_of_sound_m_per_s",
            )?),
            damping_ratio: runmat_types::physical::DampingRatio::new(remove_required_f64(
                &mut fields,
                MATERIAL_NAME,
                "damping_ratio",
            )?),
        })
    } else {
        None
    };
    let plastic = if let Some(value) = fields.remove("plastic") {
        Some(json_deserialize(
            MATERIAL_NAME,
            value,
            "plastic material model",
        )?)
    } else {
        None
    };
    reject_unknown_fields(MATERIAL_NAME, fields)?;
    if mass_density_kg_per_m3.is_none()
        && mechanical.is_none()
        && damping.is_none()
        && thermal.is_none()
        && acoustic.is_none()
        && electrical.is_none()
        && plastic.is_none()
    {
        return Err(builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            "fea.material requires at least one explicit constitutive capability",
        ));
    }
    material_to_object(MaterialModel {
        fluid: None,
        material_id,
        name,
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: (mass_density_kg_per_m3)
            .map(runmat_types::physical::MassDensity::new),
        elasticity: mechanical.map(MaterialElasticityModel::IsotropicLinear),
        damping,
        thermal,
        acoustic,
        electrical,
        plastic,
    })
}

pub(super) fn create_material_assignment_object_from_args(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MATERIAL_ASSIGNMENT_NAME,
            &ERROR_INPUT,
            "fea.materialAssignment requires a fea.PropertyRegionTarget and fea.Material",
        ));
    }
    let target: GeometryBoundPropertyRegionTarget = object_payload(
        MATERIAL_ASSIGNMENT_NAME,
        &args[0],
        FEA_PROPERTY_REGION_TARGET_CLASS,
    )?;
    let material: MaterialModel =
        object_payload(MATERIAL_ASSIGNMENT_NAME, &args[1], FEA_MATERIAL_CLASS)?;
    let assigned_material_id = material.material_id;
    let mut expected_material_id = assigned_material_id.clone();
    let mut confidence = EvidenceConfidence::Verified;
    for pair in expect_name_value_tail(MATERIAL_ASSIGNMENT_NAME, args.view().range(2..)?)? {
        match FeaMaterialAssignmentOption::from_normalized(pair.key.as_str()) {
            Some(FeaMaterialAssignmentOption::ExpectedMaterial) => {
                let material: MaterialModel =
                    object_payload(MATERIAL_ASSIGNMENT_NAME, pair.value, FEA_MATERIAL_CLASS)?;
                expected_material_id = material.material_id;
            }
            Some(FeaMaterialAssignmentOption::Confidence) => {
                let text = scalar_string(pair.value, MATERIAL_ASSIGNMENT_NAME, &ERROR_INPUT)?;
                confidence =
                    parse_scalar_enum_for_builtin(MATERIAL_ASSIGNMENT_NAME, &text, "Confidence")?;
            }
            None => {
                return Err(builtin_error(
                    MATERIAL_ASSIGNMENT_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.materialAssignment option `{}`", pair.key),
                ));
            }
        }
    }
    material_assignment_to_object(GeometryBoundMaterialAssignment {
        geometry_id: target.geometry_id,
        geometry_revision: target.geometry_revision,
        target: target.target,
        expected_material_id,
        assigned_material_id,
        confidence,
    })
}

pub(super) fn create_property_region_target_object(
    builtin: &'static str,
    region: &Value,
    kind: fn(String) -> PropertyRegionTarget,
) -> BuiltinResult<Value> {
    let region = crate::builtins::geometry::region::region_ref_from_value(builtin, region)?;
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        FEA_PROPERTY_REGION_TARGET_CLASS,
        &GeometryBoundPropertyRegionTarget {
            geometry_id: region.geometry_id,
            geometry_revision: region.geometry_revision,
            target: kind(region.region_id),
        },
    )
}

pub(super) fn create_all_property_region_target_object(
    builtin: &'static str,
    geometry: &Value,
    target: PropertyRegionTarget,
) -> BuiltinResult<Value> {
    let geometry = geometry_asset_from_value(builtin, geometry)?;
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        FEA_PROPERTY_REGION_TARGET_CLASS,
        &GeometryBoundPropertyRegionTarget {
            geometry_id: geometry.geometry_id,
            geometry_revision: geometry.revision,
            target,
        },
    )
}

fn build_model_from_parts(
    builtin: &'static str,
    geometry: &GeometryAsset,
    model_id: String,
    frame: Option<CoordinateFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<GeometryBoundMaterialAssignment>,
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
    let mut model = empty_model(model_id, geometry);

    if let Some(frame) = frame {
        model.frame = frame;
    }
    if !materials.is_empty() {
        model.materials = materials;
        if material_assignments.is_empty() {
            model.material_assignments.clear();
        }
    }
    if !material_assignments.is_empty() {
        model.material_assignments = material_assignments
            .into_iter()
            .map(|assignment| {
                assignment.require_geometry(builtin, geometry)?;
                Ok(assignment.into_assignment())
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
    } else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "fea.model requires exactly one of Sequence or Steps",
        ));
    }
    for domain in domains {
        match domain {
            DomainPayload::Thermal(domain) => model.thermal = Some(domain),
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
        units: runmat_types::physical::CoordinateUnits::canonical(),
        frame: CoordinateFrame::Global,
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
        sequence: AnalysisSequence::from_steps(format!("{model_id}.sequence"), Vec::new()),
    }
}

fn require_model_geometry(
    builtin: &'static str,
    model: &AnalysisModel,
    geometry: &GeometryAsset,
) -> BuiltinResult<()> {
    if model.geometry_id != geometry.geometry_id || model.geometry_revision != geometry.revision {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "model `{}` is bound to geometry `{}@{}`, but the study uses `{}@{}`",
                model.model_id.0,
                model.geometry_id,
                model.geometry_revision,
                geometry.geometry_id,
                geometry.revision
            ),
        ));
    }
    Ok(())
}

fn material_to_object(material: MaterialModel) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_CLASS,
        &material,
    )
}

fn material_assignment_to_object(
    assignment: GeometryBoundMaterialAssignment,
) -> BuiltinResult<Value> {
    let mut object = serializable_to_object_value(
        MATERIAL_ASSIGNMENT_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_ASSIGNMENT_CLASS,
        &assignment,
    )?;
    // Same revision type as `geometry.Asset` and every other geometry-bound
    // FEA model object.
    crate::builtins::fea::object_api::replace(
        MATERIAL_ASSIGNMENT_NAME,
        &mut object,
        "geometry_revision",
        Value::Int(IntValue::U32(assignment.geometry_revision)),
    )?;
    Ok(Value::Object(object))
}

fn model_to_object(composed: ComposedModel) -> BuiltinResult<Value> {
    let frozen_generators = FrozenGeneratorInventory::new(
        MODEL_NAME,
        &composed.model.signals,
        composed.frozen_generators.bindings().to_vec(),
    )?;
    let mut value = serializable_to_object_preserving_integers(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_MODEL_CLASS,
        &composed.model,
        &[],
        &["geometry_revision", "revision"],
    )?;
    let Value::Object(object) = &mut value else {
        unreachable!("model serialization always returns an object")
    };
    frozen_generators.attach_to(object)?;
    Ok(value)
}

pub(super) fn study_to_object_with_generators(
    spec: AnalysisStudySpec,
    frozen_generators: FrozenGeneratorInventory,
) -> BuiltinResult<Value> {
    let signals = spec.model.signals.as_slice();
    let frozen_generators =
        FrozenGeneratorInventory::new(STUDY_NAME, signals, frozen_generators.bindings().to_vec())?;
    let mut value = serializable_to_object(STUDY_NAME, &ERROR_INTERNAL, FEA_STUDY_CLASS, &spec)?;
    let Value::Object(object) = &mut value else {
        unreachable!("study serialization always returns an object")
    };
    frozen_generators.attach_to(object)?;
    Ok(value)
}

pub(super) fn study_from_value_with_generators(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ComposedStudy> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {FEA_STUDY_CLASS} object"),
        ));
    };
    let sidecar = object.properties().get(FROZEN_GENERATORS_PROPERTY).cloned();
    let study: AnalysisStudySpec = super::object_payload_omitting(
        builtin,
        value,
        FEA_STUDY_CLASS,
        &[FROZEN_GENERATORS_PROPERTY],
    )?;
    let signals = study.model.signals.as_slice();
    let frozen_generators =
        FrozenGeneratorInventory::from_property_value(builtin, sidecar.as_ref(), signals)?;
    Ok(ComposedStudy {
        study,
        frozen_generators,
    })
}

pub(super) fn sweep_to_object_with_generators(
    spec: AnalysisStudySweepSpec,
    frozen_generators: FrozenStudyGeneratorInventory,
) -> BuiltinResult<Value> {
    let mut value = serializable_to_object(SWEEP_NAME, &ERROR_INTERNAL, FEA_SWEEP_CLASS, &spec)?;
    let Value::Object(object) = &mut value else {
        unreachable!("sweep serialization always returns an object")
    };
    frozen_generators.attach_to(object)?;
    Ok(value)
}

pub(super) fn sweep_from_value_with_generators(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ComposedSweep> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {FEA_SWEEP_CLASS} object"),
        ));
    };
    let sidecar = object
        .properties()
        .get(FROZEN_STUDY_GENERATORS_PROPERTY)
        .cloned();
    let sweep: AnalysisStudySweepSpec = super::object_payload_omitting(
        builtin,
        value,
        FEA_SWEEP_CLASS,
        &[FROZEN_STUDY_GENERATORS_PROPERTY],
    )?;
    let frozen_generators = FrozenStudyGeneratorInventory::from_property_value(
        builtin,
        sidecar.as_ref(),
        &sweep.studies,
    )?;
    Ok(ComposedSweep {
        sweep,
        frozen_generators,
    })
}

pub(super) fn model_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<ComposedModel> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected {FEA_MODEL_CLASS} object"),
        ));
    };
    let sidecar = object.properties().get(FROZEN_GENERATORS_PROPERTY).cloned();
    let model: AnalysisModel = super::object_payload_omitting(
        builtin,
        value,
        FEA_MODEL_CLASS,
        &[FROZEN_GENERATORS_PROPERTY],
    )?;
    let frozen_generators =
        FrozenGeneratorInventory::from_property_value(builtin, sidecar.as_ref(), &model.signals)?;
    Ok(ComposedModel {
        model,
        frozen_generators,
    })
}

fn study_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<ComposedStudy>> {
    let values = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("expected a typed FEA study object or cell array; got {other:?}"),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| study_from_value_with_generators(builtin, value))
        .collect()
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
) -> BuiltinResult<Vec<GeometryBoundMaterialAssignment>> {
    object_vec_from_value(builtin, value, FEA_MATERIAL_ASSIGNMENT_CLASS)
}

fn constraint_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<BoundConstraintPayload>> {
    let values = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            Value::Object(object)
                if object.class_name().display_name() == FEA_STRUCTURAL_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_STRUCTURAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundStructural)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_THERMAL_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_THERMAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundThermal)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_ELECTRICAL_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_ELECTRICAL_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundElectrical)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_MAGNETIC_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_MAGNETIC_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundMagnetic)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_ACOUSTIC_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_ACOUSTIC_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundAcoustic)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_FLUID_CONSTRAINT_CLASS =>
            {
                object_payload(builtin, value, FEA_FLUID_CONSTRAINT_CLASS)
                    .map(BoundConstraintPayload::BoundFluid)
            }
            Value::Object(object) => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported FEA constraint class `{}`", object.class_name()),
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
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            Value::Object(object)
                if object.class_name().display_name() == FEA_STRUCTURAL_BOUNDARY_CLASS =>
            {
                bound_boundary_payload(builtin, value, FEA_STRUCTURAL_BOUNDARY_CLASS)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_THERMAL_BOUNDARY_CLASS =>
            {
                bound_boundary_payload(builtin, value, FEA_THERMAL_BOUNDARY_CLASS)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_MAGNETIC_BOUNDARY_CLASS =>
            {
                bound_boundary_payload(builtin, value, FEA_MAGNETIC_BOUNDARY_CLASS)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_ACOUSTIC_BOUNDARY_CLASS =>
            {
                bound_boundary_payload(builtin, value, FEA_ACOUSTIC_BOUNDARY_CLASS)
            }
            Value::Object(object)
                if object.class_name().display_name() == FEA_FLUID_BOUNDARY_CLASS =>
            {
                bound_boundary_payload(builtin, value, FEA_FLUID_BOUNDARY_CLASS)
            }
            Value::Object(object) => Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported FEA boundary class `{}`", object.class_name()),
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

fn signal_vec_from_value(builtin: &'static str, value: &Value) -> BuiltinResult<RuntimeSignals> {
    let values = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
        Value::Object(_) => vec![value],
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected a typed FEA signal object or cell array; got {other:?}"),
        ))?,
    };
    let runtime = values
        .into_iter()
        .map(|item| super::signals::runtime_signal_from_value(builtin, item))
        .collect::<BuiltinResult<Vec<_>>>()?;
    let signals = runtime
        .iter()
        .map(|signal| signal.signal.clone())
        .collect::<Vec<_>>();
    let bindings = runtime
        .into_iter()
        .filter_map(|signal| signal.frozen_generator)
        .collect();
    Ok(RuntimeSignals {
        frozen_generators: FrozenGeneratorInventory::new(builtin, &signals, bindings)?,
        signals,
    })
}

pub(super) fn step_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ProcedureStep>> {
    let values = match value {
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            let class_name = match object.class_name().display_name() {
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
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            match object.class_name().display_name() {
                FEA_THERMAL_DOMAIN_CLASS => {
                    object_payload(builtin, value, FEA_THERMAL_DOMAIN_CLASS)
                        .map(DomainPayload::Thermal)
                }
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
        Value::Cell(cell) => cell.data().iter().collect::<Vec<_>>(),
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
            let class_name = match object.class_name().display_name() {
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
            .data()
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
