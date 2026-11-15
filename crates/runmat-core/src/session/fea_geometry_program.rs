//! The geometry program a serial (browser) host materializes for one checked study.
//! Study preparation and the study geometry preview build it the same way.
use super::{CheckedFeaSessionSource, FeaStudyPreparationError};
use runmat_analysis_execution::{
    GeometryMaterializationRequest, GeometryProviderProfile,
    PreparedGeometryMaterializationProgram, PreparedGeometrySource,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_geometry_core::{GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionSource};
use runmat_runtime::analysis::{FeaRootSelector, ResolvedFeaRoot};
use std::collections::BTreeMap;

pub(super) struct SerialGeometryProgram {
    pub request: GeometryMaterializationRequest,
    pub program: PreparedGeometryMaterializationProgram,
    pub source: PreparedGeometrySource,
}

/// The program that materializes one study's geometry for display, with the exact objects it
/// reads. The host executes it with `materialize_preview_serial`.
pub struct FeaStudyGeometryPreviewProgram {
    pub program: PreparedGeometryMaterializationProgram,
    pub source: PreparedGeometrySource,
}

impl CheckedFeaSessionSource {
    pub(super) async fn serial_geometry_program(
        &self,
        geometry: &GeometryDefinition,
        bindings: BTreeMap<String, GeometryAuthoringValue>,
        profile: GeometryProviderProfile,
        revision: &ProgramRevision,
        access: &ArtifactAccess,
    ) -> Result<SerialGeometryProgram, FeaStudyPreparationError> {
        let source = match &geometry.source {
            GeometryDefinitionSource::Constructed(_) => PreparedGeometrySource::constructed(),
            GeometryDefinitionSource::Imported(imported) => {
                let logical =
                    runmat_runtime::analysis::LogicalSourcePath::new(imported.logical_path.clone())
                        .map_err(|e| e.to_string())?;
                let path =
                    runmat_runtime::analysis::physical_source_path(&self.source_root, &logical)
                        .map_err(|e| e.to_string())?;
                let bytes = self.read_source(&path).await?;
                if Digest::sha256(&bytes).bytes() != imported.content_digest.bytes() {
                    return Err("The geometry changed after checking. Refresh the simulation before running.".into());
                }
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| "Geometry filename is not valid UTF-8.".to_owned())?;
                PreparedGeometrySource::imported(name, &bytes, 4 << 20, access)
                    .map_err(|e| e.to_string())?
            }
            _ => {
                return Err(FeaStudyPreparationError::RequiresDesktop(
                    "Run this generated geometry in the desktop app.",
                ))
            }
        };
        let request = runmat_analysis_execution::prepare_geometry_materialization_request(
            geometry.clone(),
            bindings,
            &source,
            profile,
            Default::default(),
        )
        .map_err(|e| e.to_string())?;
        let program = runmat_analysis_execution::prepare_geometry_materialization_program(
            request.clone(),
            revision.clone(),
            access,
            &source.roots,
        )
        .map_err(|e| e.to_string())?;
        Ok(SerialGeometryProgram {
            request,
            program,
            source,
        })
    }

    /// Prepares the geometry program of study `study_id` for a display preview in a serial
    /// exact host. Imported and constructed geometry are supported; generated and derived
    /// geometry need the desktop host.
    pub async fn prepare_fea_study_geometry_preview(
        &self,
        study_id: &str,
        profile: GeometryProviderProfile,
        authority: &str,
    ) -> Result<FeaStudyGeometryPreviewProgram, String> {
        let selector = FeaRootSelector::Study {
            id: study_id.to_string(),
        };
        let ResolvedFeaRoot::Study(resolved) = self
            .resolve_root(Some(&selector))
            .map_err(|error| error.to_string())?
        else {
            return Err(format!("`{study_id}` did not resolve to an FEA study"));
        };
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .map_err(|error| error.to_string())?;
        let revision = self.program_revision()?;
        let access = runmat_analysis_execution::checked_fea_artifact_access(authority);
        let prepared = self
            .serial_geometry_program(
                &resolved.geometry,
                realized.geometry.bindings,
                profile,
                &revision,
                &access,
            )
            .await
            .map_err(|error| error.to_string())?;
        Ok(FeaStudyGeometryPreviewProgram {
            program: prepared.program,
            source: prepared.source,
        })
    }
}
