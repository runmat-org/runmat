use crate::analysis::analysis_create_model_op;
use crate::analysis::AnalysisCreateModelIntentSpec;
use crate::analysis::AnalysisCreateModelProfile;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    BOUNDARY_CONDITION_NAME, DOMAIN_NAME, FEA_BOUNDARY_CONDITION_CLASS, FEA_DOMAIN_CLASS,
    FEA_INTERFACE_CLASS, FEA_LOAD_CASE_CLASS, FEA_MATERIAL_ASSIGNMENT_CLASS, FEA_MATERIAL_CLASS,
    FEA_PAYLOAD_JSON_PROPERTY, FEA_RUN_OPTIONS_CLASS, FEA_STEP_CLASS, INTERFACE_NAME,
    LOAD_CASE_NAME, MATERIAL_ASSIGNMENT_NAME, MATERIAL_NAME, RUN_OPTIONS_NAME, STEP_NAME,
};
use crate::builtins::fea::errors::{builtin_error, operation_error};
use crate::builtins::fea::integer_serialization::{
    serializable_to_object, serializable_to_object_preserving_integers,
};
use crate::builtins::fea::options_json::json_deserialize;
use crate::builtins::fea::study::{DomainPayload, ModelDefaultsMode, RunOptionsPayload};
use crate::operations::OperationContext;
use crate::BuiltinResult;
use runmat_analysis_core::AnalysisInterface;
use runmat_analysis_core::AnalysisModel;
use runmat_analysis_core::AnalysisModelId;
use runmat_analysis_core::AnalysisStep;
use runmat_analysis_core::BoundaryCondition;
use runmat_analysis_core::ElectroThermalDomain;
use runmat_analysis_core::LoadCase;
use runmat_analysis_core::MaterialAssignment;
use runmat_analysis_core::MaterialModel;
use runmat_analysis_core::ReferenceFrame;
use runmat_analysis_core::ThermoMechanicalDomain;
use runmat_geometry_core::GeometryAsset;
use runmat_value::Value;

pub(in crate::builtins::fea) fn build_model_from_parts(
    builtin: &'static str,
    geometry: &GeometryAsset,
    model_id: String,
    profile: AnalysisCreateModelProfile,
    defaults: ModelDefaultsMode,
    frame: Option<ReferenceFrame>,
    materials: Vec<MaterialModel>,
    material_assignments: Vec<MaterialAssignment>,
    boundary_conditions: Vec<BoundaryCondition>,
    loads: Vec<LoadCase>,
    steps: Vec<AnalysisStep>,
    domains: Vec<DomainPayload>,
    interfaces: Vec<AnalysisInterface>,
) -> BuiltinResult<AnalysisModel> {
    let mut model = match defaults {
        ModelDefaultsMode::ProfileScaffold => analysis_create_model_op(
            geometry,
            AnalysisCreateModelIntentSpec {
                model_id: model_id.clone(),
                profile,
                prep_context: None,
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
    if !boundary_conditions.is_empty() {
        model.boundary_conditions = boundary_conditions
            .into_iter()
            .map(|mut bc| {
                bc.region_id = resolve_region_selector(builtin, &bc.region_id, geometry)?;
                Ok(bc)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !loads.is_empty() {
        model.loads = loads
            .into_iter()
            .map(|mut load| {
                load.region_id = resolve_region_selector(builtin, &load.region_id, geometry)?;
                Ok(load)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
    }
    if !steps.is_empty() {
        model.steps = steps;
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

pub(in crate::builtins::fea) fn empty_model(
    model_id: String,
    geometry: &GeometryAsset,
) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
        structural: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        boundary_conditions: Vec::new(),
        loads: Vec::new(),
        steps: Vec::new(),
    }
}

pub(in crate::builtins::fea) fn resolve_region_selector(
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

pub(in crate::builtins::fea) fn require_region_id(
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

pub(in crate::builtins::fea) fn material_to_object(
    material: MaterialModel,
) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_CLASS,
        &material,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn material_assignment_to_object(
    assignment: MaterialAssignment,
) -> BuiltinResult<Value> {
    serializable_to_object(
        MATERIAL_ASSIGNMENT_NAME,
        &ERROR_INTERNAL,
        FEA_MATERIAL_ASSIGNMENT_CLASS,
        &assignment,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn boundary_condition_to_object(
    bc: BoundaryCondition,
) -> BuiltinResult<Value> {
    serializable_to_object(
        BOUNDARY_CONDITION_NAME,
        &ERROR_INTERNAL,
        FEA_BOUNDARY_CONDITION_CLASS,
        &bc,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn load_case_to_object(load: LoadCase) -> BuiltinResult<Value> {
    serializable_to_object(
        LOAD_CASE_NAME,
        &ERROR_INTERNAL,
        FEA_LOAD_CASE_CLASS,
        &load,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn step_to_object(step: AnalysisStep) -> BuiltinResult<Value> {
    serializable_to_object(
        STEP_NAME,
        &ERROR_INTERNAL,
        FEA_STEP_CLASS,
        &step,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn domain_to_object(domain: DomainPayload) -> BuiltinResult<Value> {
    serializable_to_object_preserving_integers(
        DOMAIN_NAME,
        &ERROR_INTERNAL,
        FEA_DOMAIN_CLASS,
        &domain,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &["revision"],
    )
}

pub(in crate::builtins::fea) fn interface_to_object(
    interface: AnalysisInterface,
) -> BuiltinResult<Value> {
    serializable_to_object(
        INTERFACE_NAME,
        &ERROR_INTERNAL,
        FEA_INTERFACE_CLASS,
        &interface,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}

pub(in crate::builtins::fea) fn run_options_to_object(
    payload: RunOptionsPayload,
) -> BuiltinResult<Value> {
    serializable_to_object(
        RUN_OPTIONS_NAME,
        &ERROR_INTERNAL,
        FEA_RUN_OPTIONS_CLASS,
        &payload,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}
