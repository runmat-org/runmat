use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use runmat_analysis_core::{
    AnalysisInterface, AnalysisModel, AnalysisModelId, AnalysisStep, BoundaryCondition,
    BoundaryConditionKind, CfdDomain, ElectroThermalDomain, ElectromagneticDomain,
    EvidenceConfidence, LoadCase, LoadKind, MaterialAcousticModel, MaterialAssignment,
    MaterialElectricalModel, MaterialMechanicalModel, MaterialModel, MaterialPlasticModel,
    MaterialThermalModel, ReferenceFrame, ThermoMechanicalDomain,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{GeometryAsset, UnitSystem};
use runmat_geometry_io::GeometryImportOptions;
use serde::de::DeserializeOwned;
use serde::Deserialize;

use super::{
    analysis_create_model_op, AnalysisAcousticRunOptions, AnalysisCfdRunOptions,
    AnalysisChtRunOptions, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisElectromagneticRunOptions, AnalysisFsiRunOptions, AnalysisModalRunOptions,
    AnalysisNonlinearRunOptions, AnalysisRunKind, AnalysisRunOptions, AnalysisStudySpec,
    AnalysisStudySweepSpec, AnalysisThermalRunOptions, AnalysisTransientRunOptions,
};
use crate::operations::OperationContext;

const FEA_DOCUMENT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum FeaResolvedDocument {
    Study(Box<AnalysisStudySpec>),
    Sweep(AnalysisStudySweepSpec),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RawFeaDocument {
    Study(Box<FeaStudyDocument>),
    Sweep(FeaSweepDocument),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaSweepDocument {
    version: u32,
    id: String,
    #[serde(default = "default_fail_fast")]
    fail_fast: bool,
    studies: Vec<FeaStudyDocument>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStudyDocument {
    version: u32,
    id: String,
    geometry: FeaGeometryDocument,
    model: FeaModelDocument,
    run: FeaRunDocument,
    #[serde(default)]
    regions: BTreeMap<String, FeaRegionDocument>,
    #[serde(default)]
    materials: BTreeMap<String, FeaMaterialDocument>,
    #[serde(default)]
    material_assignments: Vec<FeaMaterialAssignmentDocument>,
    #[serde(default)]
    boundary_conditions: Vec<FeaBoundaryConditionDocument>,
    #[serde(default)]
    loads: Vec<FeaLoadDocument>,
    #[serde(default)]
    steps: Vec<FeaStepDocument>,
    #[serde(default)]
    domains: FeaDomainsDocument,
    #[serde(default)]
    interfaces: Vec<AnalysisInterface>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaGeometryDocument {
    path: PathBuf,
    #[serde(default = "default_units")]
    units: UnitSystem,
    #[serde(default)]
    import: FeaGeometryImportDocument,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaGeometryImportDocument {
    #[serde(default)]
    max_triangles: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaModelDocument {
    #[serde(default)]
    id: Option<String>,
    profile: AnalysisCreateModelProfile,
    #[serde(default)]
    frame: Option<ReferenceFrame>,
    #[serde(default)]
    defaults: FeaModelDefaultsMode,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum FeaModelDefaultsMode {
    #[default]
    ProfileScaffold,
    None,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRegionDocument {
    selector: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialDocument {
    #[serde(default)]
    name: Option<String>,
    mechanical: MaterialMechanicalModel,
    #[serde(default)]
    thermal: Option<MaterialThermalModel>,
    #[serde(default)]
    acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    plastic: Option<MaterialPlasticModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaMaterialAssignmentDocument {
    region: String,
    material: String,
    #[serde(default)]
    expected_material: Option<String>,
    #[serde(default = "default_assignment_confidence")]
    confidence: EvidenceConfidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaBoundaryConditionDocument {
    id: String,
    region: String,
    #[serde(alias = "type")]
    kind: BoundaryConditionKind,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaLoadDocument {
    id: String,
    region: String,
    #[serde(rename = "type", alias = "kind")]
    load_type: FeaLoadType,
    #[serde(default)]
    vector: Option<[f64; 3]>,
    #[serde(default)]
    magnitude_pa: Option<f64>,
    #[serde(default)]
    current_a: Option<f64>,
    #[serde(default)]
    phase_rad: Option<f64>,
    #[serde(default)]
    amplitude_scale: Option<f64>,
    #[serde(default)]
    volumetric_w_per_m3: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeaLoadType {
    Force,
    Pressure,
    BodyForce,
    CurrentDensity,
    CoilCurrent,
    HeatSource,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaStepDocument {
    id: String,
    kind: runmat_analysis_core::AnalysisStepKind,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaDomainsDocument {
    #[serde(default)]
    thermo_mechanical: Option<ThermoMechanicalDomain>,
    #[serde(default)]
    electro_thermal: Option<ElectroThermalDomain>,
    #[serde(default)]
    electromagnetic: Option<ElectromagneticDomain>,
    #[serde(default)]
    cfd: Option<CfdDomain>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaRunDocument {
    #[serde(default)]
    kind: Option<AnalysisRunKind>,
    #[serde(default = "default_backend")]
    backend: ComputeBackend,
    #[serde(default)]
    options: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResolvedStudyParts {
    spec: AnalysisStudySpec,
}

pub fn is_fea_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("fea"))
}

pub async fn load_fea_document_from_path_async(path: &Path) -> Result<FeaResolvedDocument, String> {
    if !is_fea_file_path(path) {
        return Err(format!(
            "unsupported FEA document extension: {}",
            path.display()
        ));
    }
    let input = runmat_filesystem::read_to_string_async(path)
        .await
        .map_err(|err| format!("failed to read FEA document {}: {err}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
    parse_and_resolve_fea_document(&input, base_dir).await
}

pub async fn parse_and_resolve_fea_document(
    input: &str,
    base_dir: &Path,
) -> Result<FeaResolvedDocument, String> {
    let raw = serde_yaml::from_str::<RawFeaDocument>(input)
        .map_err(|err| format!("failed to parse FEA YAML: {err}"))?;
    match raw {
        RawFeaDocument::Study(study) => {
            let resolved = resolve_study(*study, base_dir).await?;
            Ok(FeaResolvedDocument::Study(Box::new(resolved.spec)))
        }
        RawFeaDocument::Sweep(sweep) => resolve_sweep(sweep, base_dir).await,
    }
}

async fn resolve_sweep(
    sweep: FeaSweepDocument,
    base_dir: &Path,
) -> Result<FeaResolvedDocument, String> {
    validate_version(sweep.version)?;
    if sweep.id.trim().is_empty() {
        return Err("sweep id must be non-empty".to_string());
    }
    let mut studies = Vec::with_capacity(sweep.studies.len());
    for study in sweep.studies {
        studies.push(resolve_study(study, base_dir).await?.spec);
    }
    Ok(FeaResolvedDocument::Sweep(AnalysisStudySweepSpec {
        sweep_id: sweep.id,
        studies,
        fail_fast: sweep.fail_fast,
    }))
}

async fn resolve_study(
    study: FeaStudyDocument,
    base_dir: &Path,
) -> Result<ResolvedStudyParts, String> {
    validate_version(study.version)?;
    if study.id.trim().is_empty() {
        return Err("study id must be non-empty".to_string());
    }

    let geometry = load_geometry(&study.geometry, base_dir).await?;
    let model_id = study
        .model
        .id
        .clone()
        .unwrap_or_else(|| format!("{}_model", sanitize_id(&study.id)));
    let intent = AnalysisCreateModelIntentSpec {
        model_id: model_id.clone(),
        profile: study.model.profile,
        prep_context: None,
    };
    let model = resolve_model(&study, &geometry, &intent)?;
    let run_kind = resolve_run_kind(study.model.profile, &study.run)?;
    let run_options = resolve_run_options(&study.run, run_kind)?;
    let spec = AnalysisStudySpec {
        study_id: study.id,
        geometry,
        create_model_intent: intent,
        model,
        run_kind,
        backend: study.run.backend,
        linear_static_run_options: run_options.linear_static,
        modal_run_options: run_options.modal,
        acoustic_run_options: run_options.acoustic,
        thermal_run_options: run_options.thermal,
        transient_run_options: run_options.transient,
        cfd_run_options: run_options.cfd,
        cht_run_options: run_options.cht,
        fsi_run_options: run_options.fsi,
        nonlinear_run_options: run_options.nonlinear,
        electromagnetic_run_options: run_options.electromagnetic,
    };
    Ok(ResolvedStudyParts { spec })
}

async fn load_geometry(
    geometry: &FeaGeometryDocument,
    base_dir: &Path,
) -> Result<GeometryAsset, String> {
    let path = resolve_document_path(base_dir, &geometry.path);
    let bytes = runmat_filesystem::read_async(&path)
        .await
        .map_err(|err| format!("failed to read geometry file {}: {err}", path.display()))?;
    let options = GeometryImportOptions {
        max_triangles: geometry.import.max_triangles.or(Some(16_000_000)),
        budget_policy: runmat_geometry_io::GeometryImportBudgetPolicy::Strict,
        units: geometry.units,
        tessellation_profile: Default::default(),
        relative_deflection: false,
    };
    crate::geometry::geometry_load_with_options_op(
        &path.to_string_lossy(),
        &bytes,
        options,
        OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|err| {
        format!(
            "failed to load geometry {}: {}",
            path.display(),
            err.message
        )
    })
}

fn resolve_model(
    study: &FeaStudyDocument,
    geometry: &GeometryAsset,
    intent: &AnalysisCreateModelIntentSpec,
) -> Result<Option<AnalysisModel>, String> {
    if !has_explicit_model_data(study)
        && study.model.defaults == FeaModelDefaultsMode::ProfileScaffold
    {
        return Ok(None);
    }

    let mut model = match study.model.defaults {
        FeaModelDefaultsMode::ProfileScaffold => {
            analysis_create_model_op(geometry, intent.clone(), OperationContext::new(None, None))
                .map(|envelope| envelope.data)
                .map_err(|err| format!("failed to create FEA model scaffold: {}", err.message))?
        }
        FeaModelDefaultsMode::None => empty_model(intent.model_id.clone(), geometry),
    };

    if let Some(frame) = &study.model.frame {
        model.frame = frame.clone();
    }
    if !study.materials.is_empty() {
        model.materials = study
            .materials
            .iter()
            .map(|(id, material)| resolve_material(id, material))
            .collect();
    }
    if !study.material_assignments.is_empty() {
        model.material_assignments = study
            .material_assignments
            .iter()
            .map(|assignment| resolve_material_assignment(assignment, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if !study.boundary_conditions.is_empty() {
        model.boundary_conditions = study
            .boundary_conditions
            .iter()
            .map(|bc| resolve_boundary_condition(bc, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if !study.loads.is_empty() {
        model.loads = study
            .loads
            .iter()
            .map(|load| resolve_load(load, geometry, &study.regions))
            .collect::<Result<Vec<_>, _>>()?;
    }
    if !study.steps.is_empty() {
        model.steps = study
            .steps
            .iter()
            .map(|step| AnalysisStep {
                step_id: step.id.clone(),
                kind: step.kind.clone(),
            })
            .collect();
    }
    if study.domains.thermo_mechanical.is_some() {
        model.thermo_mechanical = study.domains.thermo_mechanical.clone();
    }
    if study.domains.electro_thermal.is_some() {
        model.electro_thermal = study.domains.electro_thermal.clone();
    }
    if study.domains.electromagnetic.is_some() {
        model.electromagnetic = study.domains.electromagnetic.clone();
    }
    if study.domains.cfd.is_some() {
        model.cfd = study.domains.cfd.clone();
    }
    if !study.interfaces.is_empty() {
        model.interfaces = study.interfaces.clone();
    }

    Ok(Some(model))
}

fn resolve_material(id: &str, material: &FeaMaterialDocument) -> MaterialModel {
    MaterialModel {
        material_id: id.to_string(),
        name: material.name.clone().unwrap_or_else(|| id.to_string()),
        mechanical: material.mechanical.clone(),
        thermal: material.thermal.clone().unwrap_or_default(),
        acoustic: material.acoustic.clone(),
        electrical: material.electrical.clone(),
        plastic: material.plastic.clone(),
    }
}

fn resolve_material_assignment(
    assignment: &FeaMaterialAssignmentDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<MaterialAssignment, String> {
    let region_id = resolve_region_ref(&assignment.region, geometry, aliases)?;
    Ok(MaterialAssignment {
        region_id,
        expected_material_id: assignment
            .expected_material
            .clone()
            .unwrap_or_else(|| assignment.material.clone()),
        assigned_material_id: assignment.material.clone(),
        confidence: assignment.confidence,
    })
}

fn resolve_boundary_condition(
    bc: &FeaBoundaryConditionDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<BoundaryCondition, String> {
    Ok(BoundaryCondition {
        bc_id: bc.id.clone(),
        region_id: resolve_region_ref(&bc.region, geometry, aliases)?,
        kind: bc.kind.clone(),
    })
}

fn resolve_load(
    load: &FeaLoadDocument,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<LoadCase, String> {
    let kind = match load.load_type {
        FeaLoadType::Force => {
            let [fx, fy, fz] = load_vector(load, "force")?;
            LoadKind::Force { fx, fy, fz }
        }
        FeaLoadType::Pressure => LoadKind::Pressure {
            magnitude_pa: required_f64(load.magnitude_pa, "pressure.magnitude_pa")?,
        },
        FeaLoadType::BodyForce => {
            let [gx, gy, gz] = load_vector(load, "body_force")?;
            LoadKind::BodyForce { gx, gy, gz }
        }
        FeaLoadType::CurrentDensity => {
            let [jx, jy, jz] = load_vector(load, "current_density")?;
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad: load.phase_rad.unwrap_or_default(),
                amplitude_scale: load.amplitude_scale.unwrap_or(1.0),
            }
        }
        FeaLoadType::CoilCurrent => LoadKind::CoilCurrent {
            current_a: required_f64(load.current_a, "coil_current.current_a")?,
            phase_rad: load.phase_rad.unwrap_or_default(),
            amplitude_scale: load.amplitude_scale.unwrap_or(1.0),
        },
        FeaLoadType::HeatSource => LoadKind::HeatSource {
            volumetric_w_per_m3: required_f64(
                load.volumetric_w_per_m3,
                "heat_source.volumetric_w_per_m3",
            )?,
        },
    };
    Ok(LoadCase {
        load_id: load.id.clone(),
        region_id: resolve_region_ref(&load.region, geometry, aliases)?,
        kind,
    })
}

#[derive(Debug, Default)]
struct ResolvedRunOptions {
    linear_static: Option<AnalysisRunOptions>,
    modal: Option<AnalysisModalRunOptions>,
    acoustic: Option<AnalysisAcousticRunOptions>,
    thermal: Option<AnalysisThermalRunOptions>,
    transient: Option<AnalysisTransientRunOptions>,
    cfd: Option<AnalysisCfdRunOptions>,
    cht: Option<AnalysisChtRunOptions>,
    fsi: Option<AnalysisFsiRunOptions>,
    nonlinear: Option<AnalysisNonlinearRunOptions>,
    electromagnetic: Option<AnalysisElectromagneticRunOptions>,
}

fn resolve_run_kind(
    profile: AnalysisCreateModelProfile,
    run: &FeaRunDocument,
) -> Result<AnalysisRunKind, String> {
    let derived = profile.derived_run_kind();
    if let Some(explicit) = run.kind {
        if explicit != derived {
            return Err(format!(
                "run.kind {:?} does not match the solver selected by model.profile {:?}; omit run.kind unless you need an advanced matching solver override",
                explicit, profile
            ));
        }
    }
    Ok(derived)
}

fn resolve_run_options(
    run: &FeaRunDocument,
    run_kind: AnalysisRunKind,
) -> Result<ResolvedRunOptions, String> {
    let Some(options) = run.options.clone() else {
        return Ok(ResolvedRunOptions::default());
    };
    let mut resolved = ResolvedRunOptions::default();
    match run_kind {
        AnalysisRunKind::LinearStatic => {
            resolved.linear_static = Some(parse_options(options, "linear_static options")?);
        }
        AnalysisRunKind::Modal => {
            resolved.modal = Some(parse_options(options, "modal options")?);
        }
        AnalysisRunKind::Acoustic => {
            resolved.acoustic = Some(parse_options(options, "acoustic options")?);
        }
        AnalysisRunKind::Thermal => {
            resolved.thermal = Some(parse_options(options, "thermal options")?);
        }
        AnalysisRunKind::Transient => {
            resolved.transient = Some(parse_options(options, "transient options")?);
        }
        AnalysisRunKind::Cfd => {
            resolved.cfd = Some(parse_options(options, "cfd options")?);
        }
        AnalysisRunKind::Cht => {
            resolved.cht = Some(parse_options(options, "cht options")?);
        }
        AnalysisRunKind::Fsi => {
            resolved.fsi = Some(parse_options(options, "fsi options")?);
        }
        AnalysisRunKind::Nonlinear => {
            resolved.nonlinear = Some(parse_options(options, "nonlinear options")?);
        }
        AnalysisRunKind::Electromagnetic => {
            resolved.electromagnetic = Some(parse_options(options, "electromagnetic options")?);
        }
    }
    Ok(resolved)
}

fn parse_options<T: DeserializeOwned>(
    options: serde_yaml::Value,
    label: &str,
) -> Result<T, String> {
    serde_yaml::from_value(options).map_err(|err| format!("invalid {label}: {err}"))
}

fn empty_model(model_id: String, geometry: &GeometryAsset) -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId(model_id),
        geometry_id: geometry.geometry_id.clone(),
        geometry_revision: geometry.revision,
        units: geometry.units,
        frame: ReferenceFrame::Global,
        materials: Vec::new(),
        material_assignments: Vec::new(),
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

fn has_explicit_model_data(study: &FeaStudyDocument) -> bool {
    !study.materials.is_empty()
        || !study.material_assignments.is_empty()
        || !study.boundary_conditions.is_empty()
        || !study.loads.is_empty()
        || !study.steps.is_empty()
        || study.domains.thermo_mechanical.is_some()
        || study.domains.electro_thermal.is_some()
        || study.domains.electromagnetic.is_some()
        || study.domains.cfd.is_some()
        || !study.interfaces.is_empty()
        || study.model.frame.is_some()
}

fn resolve_region_ref(
    reference: &str,
    geometry: &GeometryAsset,
    aliases: &BTreeMap<String, FeaRegionDocument>,
) -> Result<String, String> {
    if let Some(alias) = aliases.get(reference) {
        return resolve_region_selector(&alias.selector, geometry);
    }
    resolve_region_selector(reference, geometry)
}

fn resolve_region_selector(selector: &str, geometry: &GeometryAsset) -> Result<String, String> {
    if let Some(id) = selector
        .strip_prefix("id:")
        .or_else(|| selector.strip_prefix("region:"))
    {
        return require_region_id(id, geometry);
    }
    if let Some(tag) = selector.strip_prefix("tag:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.tag.as_deref() == Some(tag))
            .map(|region| region.region_id.clone())
            .ok_or_else(|| format!("region tag `{tag}` was not found in geometry"));
    }
    if let Some(name) = selector.strip_prefix("name:") {
        return geometry
            .regions
            .iter()
            .find(|region| region.name == name)
            .map(|region| region.region_id.clone())
            .ok_or_else(|| format!("region name `{name}` was not found in geometry"));
    }
    require_region_id(selector, geometry)
}

fn require_region_id(region_id: &str, geometry: &GeometryAsset) -> Result<String, String> {
    geometry
        .regions
        .iter()
        .find(|region| region.region_id == region_id)
        .map(|region| region.region_id.clone())
        .ok_or_else(|| format!("region id `{region_id}` was not found in geometry"))
}

fn load_vector(load: &FeaLoadDocument, label: &str) -> Result<[f64; 3], String> {
    load.vector
        .ok_or_else(|| format!("{label} load requires vector: [x, y, z]"))
}

fn required_f64(value: Option<f64>, label: &str) -> Result<f64, String> {
    value.ok_or_else(|| format!("{label} is required"))
}

fn resolve_document_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn validate_version(version: u32) -> Result<(), String> {
    if version == FEA_DOCUMENT_VERSION {
        Ok(())
    } else {
        Err(format!(
            "unsupported FEA document version {version}; expected {FEA_DOCUMENT_VERSION}"
        ))
    }
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn default_units() -> UnitSystem {
    UnitSystem::Meter
}

fn default_backend() -> ComputeBackend {
    ComputeBackend::Cpu
}

fn default_fail_fast() -> bool {
    true
}

fn default_assignment_confidence() -> EvidenceConfidence {
    EvidenceConfidence::Verified
}
