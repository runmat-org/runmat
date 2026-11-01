//! Authored family results for publication, from local or remote completions.
//!
//! The coordinators, completion roots and publication owners are shared by every placement.
//! This module only binds a completed family to the object source that holds its closure.
use runmat_execution::value::ValueRef;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};
use runmat_runtime::analysis::{
    AcousticResponseStudyCompletion, AcousticResponseStudyObjectSource, AcousticStudyCompletion,
    AcousticStudyObjectSource, ElectricalStudyCompletion, ElectricalStudyObjectSource,
    FluidStudyCompletion, FluidStudyObjectSource, MagneticStudyCompletion,
    MagneticStudyObjectSource, MaxwellStudyCompletion, MaxwellStudyObjectSource,
    NonlinearSolidStudyCompletion, NonlinearSolidStudyObjectSource, ThermalStudyCompletion,
    ThermalStudyObjectSource,
};

use super::CompletedFeaStudyResult;
use crate::{CompletedFamilyStudy, StudyFamilyKind};

macro_rules! completed_family {
    ($(#[$meta:meta])* $name:ident, $completion:ty, $source:ty) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            pub completion: $completion,
            pub source: $source,
            pub mesh: SolverMeshArtifact,
            pub meshing_evidence: MeshingEvidence,
        }
    };
}

completed_family!(
    /// Completed standalone thermal study bound to its authenticated object closure.
    CompletedAuthoredThermalStudy,
    ThermalStudyCompletion,
    ThermalStudyObjectSource
);
completed_family!(
    /// Completed electrostatic or steady-conduction study.
    CompletedAuthoredElectricalStudy,
    ElectricalStudyCompletion,
    ElectricalStudyObjectSource
);
completed_family!(
    /// Completed magnetostatic study.
    CompletedAuthoredMagneticStudy,
    MagneticStudyCompletion,
    MagneticStudyObjectSource
);
completed_family!(
    /// Completed harmonic Maxwell study.
    CompletedAuthoredMaxwellStudy,
    MaxwellStudyCompletion,
    MaxwellStudyObjectSource
);
completed_family!(
    /// Completed harmonic acoustic study.
    CompletedAuthoredAcousticStudy,
    AcousticStudyCompletion,
    AcousticStudyObjectSource
);
completed_family!(
    /// Completed acoustic cavity-mode or transient study.
    CompletedAuthoredAcousticResponseStudy,
    AcousticResponseStudyCompletion,
    AcousticResponseStudyObjectSource
);
completed_family!(
    /// Completed steady fluid study.
    CompletedAuthoredFluidStudy,
    FluidStudyCompletion,
    FluidStudyObjectSource
);
completed_family!(
    /// Completed J2 nonlinear solid study.
    CompletedAuthoredNonlinearSolidStudy,
    NonlinearSolidStudyCompletion,
    NonlinearSolidStudyObjectSource
);

/// Bind a locally executed family to the session store that retains its task outputs.
pub(crate) fn local_result(
    completed: CompletedFamilyStudy,
    source: crate::NativeProgramObjectSource,
    store: &mut impl CacheExport,
    mesh: SolverMeshArtifact,
    meshing_evidence: MeshingEvidence,
) -> Result<CompletedFeaStudyResult, String> {
    macro_rules! bind {
        ($variant:ident, $name:ident, $convert:path, $completed:expr) => {{
            let (completion, source) = $convert(*$completed, source)?;
            CompletedFeaStudyResult::$variant(Box::new($name {
                completion,
                source,
                mesh,
                meshing_evidence,
            }))
        }};
    }
    Ok(match completed {
        CompletedFamilyStudy::Thermal(completed) => bind!(
            Thermal,
            CompletedAuthoredThermalStudy,
            crate::analysis_placement::convert_completed_thermal_study,
            completed
        ),
        CompletedFamilyStudy::Electrical(completed) => bind!(
            Electrical,
            CompletedAuthoredElectricalStudy,
            crate::analysis_electrical::source::convert,
            completed
        ),
        CompletedFamilyStudy::Magnetic(completed) => bind!(
            Magnetic,
            CompletedAuthoredMagneticStudy,
            crate::analysis_magnetic::source::convert,
            completed
        ),
        CompletedFamilyStudy::Maxwell(completed) => bind!(
            Maxwell,
            CompletedAuthoredMaxwellStudy,
            crate::analysis_maxwell::source::convert,
            completed
        ),
        CompletedFamilyStudy::Acoustic(completed) => bind!(
            Acoustic,
            CompletedAuthoredAcousticStudy,
            crate::analysis_acoustic::source::convert,
            completed
        ),
        CompletedFamilyStudy::AcousticResponse(completed) => bind!(
            AcousticResponse,
            CompletedAuthoredAcousticResponseStudy,
            crate::analysis_acoustic_response::source::convert,
            completed
        ),
        CompletedFamilyStudy::Fluid(completed) => bind!(
            Fluid,
            CompletedAuthoredFluidStudy,
            crate::analysis_fluid::source::convert,
            completed
        ),
        CompletedFamilyStudy::NonlinearSolid(completed) => {
            let (completion, source) =
                crate::analysis_nonlinear::source::convert(*completed, source, store)?;
            CompletedFeaStudyResult::Nonlinear(Box::new(CompletedAuthoredNonlinearSolidStudy {
                completion,
                source,
                mesh,
                meshing_evidence,
            }))
        }
    })
}

/// Object-inventory and resident-memory ceilings for an imported family result.
pub struct FamilyResultImportLimits {
    pub objects: ObjectInventoryLimits,
    pub maximum_resident_bytes: u64,
}

/// Import a transported family completion root from its downloaded object closure.
///
/// The importer replays every phase from the canonical root and authenticates each response
/// against its closure; nothing from the executing host is trusted beyond those objects.
pub fn imported_family_result(
    kind: StudyFamilyKind,
    source: impl CacheImport + Send + Sync + 'static,
    root: &ValueRef,
    access: &ArtifactAccess,
    import_limits: FamilyResultImportLimits,
    mesh: SolverMeshArtifact,
    meshing_evidence: MeshingEvidence,
) -> Result<CompletedFeaStudyResult, String> {
    let FamilyResultImportLimits {
        objects: limits,
        maximum_resident_bytes,
    } = import_limits;
    let digest = root.logical_digest;
    macro_rules! bind {
        ($variant:ident, $name:ident, $import:path, $convert:path) => {{
            let completed =
                $import(&source, root, digest, access, limits).map_err(|e| e.to_string())?;
            let (completion, source) = $convert(completed, source)?;
            CompletedFeaStudyResult::$variant(Box::new($name {
                completion,
                source,
                mesh,
                meshing_evidence,
            }))
        }};
    }
    Ok(match kind {
        StudyFamilyKind::Thermal => bind!(
            Thermal,
            CompletedAuthoredThermalStudy,
            runmat_analysis_execution::import_thermal_study_root,
            crate::thermal_study_completion_from_source
        ),
        StudyFamilyKind::Electrical => bind!(
            Electrical,
            CompletedAuthoredElectricalStudy,
            runmat_analysis_execution::import_electrical_study_root,
            crate::electrical_study_completion_from_source
        ),
        StudyFamilyKind::Magnetic => bind!(
            Magnetic,
            CompletedAuthoredMagneticStudy,
            runmat_analysis_execution::import_magnetic_study_root,
            crate::magnetic_study_completion_from_source
        ),
        StudyFamilyKind::Maxwell => bind!(
            Maxwell,
            CompletedAuthoredMaxwellStudy,
            runmat_analysis_execution::import_maxwell_study_root,
            crate::maxwell_study_completion_from_source
        ),
        StudyFamilyKind::Acoustic => bind!(
            Acoustic,
            CompletedAuthoredAcousticStudy,
            runmat_analysis_execution::import_acoustic_study_root,
            crate::acoustic_study_completion_from_source
        ),
        StudyFamilyKind::AcousticResponse => bind!(
            AcousticResponse,
            CompletedAuthoredAcousticResponseStudy,
            runmat_analysis_execution::import_acoustic_response_study_root,
            crate::acoustic_response_study_completion_from_source
        ),
        StudyFamilyKind::Fluid => bind!(
            Fluid,
            CompletedAuthoredFluidStudy,
            runmat_analysis_execution::import_fluid_study_root,
            crate::fluid_study_completion_from_source
        ),
        StudyFamilyKind::NonlinearSolid => {
            let (completion, source) = crate::nonlinear_solid_study_completion_from_source(
                source,
                root.clone(),
                digest,
                access.clone(),
                limits,
                maximum_resident_bytes,
            )?;
            CompletedFeaStudyResult::Nonlinear(Box::new(CompletedAuthoredNonlinearSolidStudy {
                completion,
                source,
                mesh,
                meshing_evidence,
            }))
        }
    })
}
