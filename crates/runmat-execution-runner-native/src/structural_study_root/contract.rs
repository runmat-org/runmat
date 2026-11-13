//! Immutable authored root and client-held result authority.
use super::*;
use runmat_analysis_fea::accelerator_admission::AcceleratorPlacement;

/// Family-specific planning carried by the one portable authored study root.
///
/// Every authored family travels through the same root contract; only the family planning and
/// the composition selected from it differ after exact geometry and meshing complete.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "policy",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum StudyRootPlanning {
    Structural(runmat_analysis_fea::study::StructuralStudyPlanningPolicy),
    Thermal(runmat_analysis_fea::study::ThermalStudyPlanningPolicy),
    Electrical(runmat_analysis_fea::study::ElectricalStudyPlanningPolicy),
    Magnetic(runmat_analysis_fea::study::MagneticStudyPlanningPolicy),
    Maxwell(runmat_analysis_fea::study::MaxwellStudyPlanningPolicy),
    Acoustic(runmat_analysis_fea::study::AcousticStudyPlanningPolicy),
    AcousticResponse(runmat_analysis_fea::study::AcousticResponseStudyPlanningPolicy),
    Fluid(runmat_analysis_fea::study::FluidStudyPlanningPolicy),
    NonlinearSolid(Box<NonlinearSolidStudyPlanning>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidStudyPlanning {
    pub controls: runmat_analysis_fea::solve::nonlinear::NonlinearSolveOptions,
    pub accepted_block_limit: u64,
}

const _: () = assert!(
    std::mem::size_of::<StudyRootPlanning>() < std::mem::size_of::<NonlinearSolidStudyPlanning>()
);

impl StudyRootPlanning {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        match self {
            Self::Structural(policy) => policy.numerics.validate().map_err(protocol),
            Self::NonlinearSolid(policy) => {
                if policy.accepted_block_limit == 0 || policy.controls.solver_mesh.is_some() {
                    return Err(protocol("nonlinear solid root planning is invalid"));
                }
                Ok(())
            }
            Self::Thermal(_)
            | Self::Electrical(_)
            | Self::Magnetic(_)
            | Self::Maxwell(_)
            | Self::Acoustic(_)
            | Self::AcousticResponse(_)
            | Self::Fluid(_) => Ok(()),
        }
    }

    /// Admitted analysis backend; families without an accelerator envelope plan on the host.
    pub fn backend(&self) -> runmat_analysis_fea::ComputeBackend {
        match self {
            Self::Structural(policy) => policy.backend,
            Self::Thermal(policy) => policy.backend,
            Self::Electrical(_)
            | Self::Magnetic(_)
            | Self::Maxwell(_)
            | Self::Acoustic(_)
            | Self::AcousticResponse(_)
            | Self::Fluid(_)
            | Self::NonlinearSolid(_) => runmat_analysis_fea::ComputeBackend::Cpu,
        }
    }

    pub fn structural(&self) -> Option<runmat_analysis_fea::study::StructuralStudyPlanningPolicy> {
        match self {
            Self::Structural(policy) => Some(*policy),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StructuralStudyRootRequest {
    pub(super) schema_version: u16,
    pub(super) authored_study_digest: Digest,
    pub(super) frozen_project_digest: Option<Digest>,
    pub(super) study: RealizedStudy,
    pub(super) materials: Vec<StructuralStudyMaterial>,
    pub(super) planning: StudyRootPlanning,
    pub(super) local_execution: LocalExecutionPlan,
    /// Execution policy `memory_per_worker`: every placement (local or remote) plans each
    /// worker within it. Absent when not authored, so roots without it keep their bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) memory_per_worker_bytes: Option<u64>,
    /// Execution policy accelerator admission (`accelerators = required` is strict): the worker
    /// admits the study's placement under it before any meshing or solve. Absent for the
    /// default fallback policy, so roots without it keep their bytes.
    #[serde(
        default,
        skip_serializing_if = "runmat_analysis_core::AcceleratorAdmissionPolicy::is_fallback"
    )]
    pub(super) accelerator_admission: runmat_analysis_core::AcceleratorAdmissionPolicy,
    pub(super) analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub(super) input_materializations: Vec<InputMaterializationPlanRequest>,
    pub(super) geometry_meshing: ProgramExecutionDescriptor,
    pub(super) geometry_argument_count: u32,
    pub(super) materialization_argument_count: u32,
    pub(super) initial_fields: Vec<initial_fields::InitialFieldInput>,
    pub(super) initial_field_argument_count: u32,
}

impl StructuralStudyRootRequest {
    pub(super) fn validate(&self) -> NativeExecutionResult<()> {
        self.planning.validate()?;
        initial_fields::validate_inputs(&self.study, &self.initial_fields)?;
        self.geometry_meshing
            .validate_for_portable_host()
            .map_err(protocol)?;
        if self.schema_version != 4
            || self.authored_study_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
            || self.study.module_path.trim().is_empty()
            || self.study.study_id.trim().is_empty()
            || self.geometry_argument_count == 0
            || self.memory_per_worker_bytes == Some(0)
            || self.geometry_meshing.artifact.form != ExecutableForm::MeshingWorkload
            || self.geometry_meshing.recipe.execution_mode
                != crate::GEOMETRY_MESHING_ROOT_EXECUTION_MODE
            || self.geometry_meshing.requested_outputs != 1
        {
            return Err(protocol("structural study root request is invalid"));
        }
        let mut identities = BTreeSet::new();
        if self
            .materials
            .iter()
            .any(|entry| !identities.insert(entry.reference.clone()))
        {
            return Err(protocol(
                "structural study material identities are not unique",
            ));
        }
        let materials = self
            .materials
            .iter()
            .map(|entry| entry.material.clone())
            .collect::<Vec<_>>();
        runmat_analysis_core::validate_material_definitions(&materials).map_err(protocol)?;
        validate_accelerator_requests(
            self.planning.backend(),
            &self.analysis_accelerator_requests,
        )?;
        let materializations = self
            .input_materializations
            .iter()
            .map(InputMaterializationPlanRequest::to_plan)
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        validate_input_materialization_plan_inventory(
            &self.study,
            &materializations,
            &self.geometry_meshing.recipe.program_revision,
        )?;
        let reference_count = materializations
            .iter()
            .flat_map(|prepared| prepared.input_references().iter())
            .map(|reference| reference.id)
            .collect::<BTreeSet<_>>()
            .len();
        if usize::try_from(self.materialization_argument_count).map_err(protocol)?
            != reference_count
        {
            return Err(protocol(
                "structural study input-reference count differs from its materialization plans",
            ));
        }
        Ok(())
    }

    pub(super) fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(ROOT_PREFIX, ROOT_DOMAIN, self, CONTRACT_LIMITS).map_err(protocol)
    }

    pub(super) fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let request =
            decode_contract(ROOT_PREFIX, ROOT_DOMAIN, bytes, CONTRACT_LIMITS).map_err(protocol)?;
        Self::validate(&request)?;
        Ok(request)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StructuralStudyMaterial {
    pub(super) reference: AuthoredMaterialRef,
    pub(super) material: MaterialModel,
}

pub struct StructuralStudyRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub planning: StudyRootPlanning,
    pub local_execution: LocalExecutionPlan,
    /// The execution policy's `memory_per_worker`, when authored.
    pub memory_per_worker_bytes: Option<u64>,
    /// The execution policy's accelerator admission (`accelerators = required` is strict).
    pub accelerator_admission: runmat_analysis_core::AcceleratorAdmissionPolicy,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationPlan>,
    pub initial_fields: Vec<runmat_analysis_fea::execution::InitialFieldBinding>,
    pub geometry_meshing: PreparedGeometryMeshingRoot,
}

/// Client-held admission contract for a geometry-derived structural study result.
///
/// Workloads are intentionally absent: exact topology is not available until the remote geometry
/// and meshing stages finish. The importer admits the published compiled study against these
/// immutable inputs, derives the workloads locally, and then applies the ordinary exact-result
/// validation path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: Vec<StructuralStudyMaterialExpectation>,
    pub planning: StudyRootPlanning,
    pub local_execution: LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationPlanExpectation>,
    pub initial_fields: Vec<runmat_analysis_fea::execution::InitialFieldBinding>,
    pub geometry_request: runmat_analysis_execution::GeometryMaterializationRequest,
    pub expected_meshing_request: runmat_meshing_core::MeshingRequest,
    /// The placement the submitting host admitted; settled when the result is published.
    #[serde(default, skip_serializing_if = "AcceleratorPlacement::is_default")]
    pub accelerator_placement: AcceleratorPlacement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyMaterialExpectation {
    pub reference: AuthoredMaterialRef,
    pub material: MaterialModel,
}

impl StructuralStudyResultExpectation {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        initial_fields::validate_bindings(&self.study, &self.initial_fields)?;
        self.artifact_access.validate().map_err(protocol)?;
        self.planning.validate()?;
        self.expected_meshing_request.validate().map_err(protocol)?;
        self.geometry_request.validate().map_err(protocol)?;
        if self.authored_study_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
        {
            return Err(protocol("structural study result expectation is invalid"));
        }
        if self.study.module_path.trim().is_empty()
            || self.study.study_id.trim().is_empty()
            || self.materials.iter().enumerate().any(|(index, material)| {
                self.materials[..index]
                    .iter()
                    .any(|existing| existing.reference == material.reference)
            })
        {
            return Err(protocol(
                "structural study result source contract is invalid",
            ));
        }
        runmat_analysis_core::validate_material_definitions(
            &self
                .materials
                .iter()
                .map(|entry| entry.material.clone())
                .collect::<Vec<_>>(),
        )
        .map_err(protocol)?;
        let materials = self
            .materials
            .iter()
            .map(|entry| (entry.reference.clone(), entry.material.clone()))
            .collect::<BTreeMap<_, _>>();
        let expected_source_digest =
            runmat_analysis_fea::study::structural_study_source_digest(&self.study, &materials)
                .map_err(protocol)?;
        let geometry_definition = runmat_geometry_core::GeometryDefinitionRef {
            module_path: self.geometry_request.definition.module_path.clone(),
            geometry_id: self.geometry_request.definition.geometry_id.clone(),
        };
        if expected_source_digest != self.authored_study_digest
            || self.study.geometry.definition != geometry_definition
            || self.study.geometry.source != self.geometry_request.definition.source
            || self.study.geometry.bindings != self.geometry_request.bindings
        {
            return Err(protocol(
                "structural study result source identity differs from its geometry or materials",
            ));
        }
        validate_accelerator_requests(
            self.planning.backend(),
            &self.analysis_accelerator_requests,
        )?;
        validate_input_plan_expectation_inventory(
            &self.study,
            &self.input_materializations,
            &self.program_revision,
        )?;
        Ok(())
    }
}
