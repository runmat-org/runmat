//! Fast advisory planning from checked source and existing preview measurements.
//! No geometry materialization, solver mesh, assembly, or numerical solve occurs here.
//!
//! The check estimates the solver mesh from the imported preview's measure and the mesh
//! settings, then the static solid plan's memory from those counts, so a study too large for
//! its target is refused with numbers before Run instead of after meshing.
use super::fea_browser_budget::{BrowserResourceLimit, BrowserResourceTerm};
use super::CheckedFeaSessionSource;
use runmat_analysis_fea::planning::resource::estimate_static_solid_from_counts;
use runmat_geometry_core::GeometryDefinitionSource;
use runmat_meshing_core::{estimate_solid_mesh_size, MeshDimension, MeshingRequestSettings};
use runmat_runtime::analysis::{AnalysisResourceLimit, ResolvedFeaRoot};

/// The code of a native refusal at check time.
pub const RESOURCE_LIMIT_CODE: &str = "RM.FEA.RUN_STUDY.RESOURCE_LIMIT";
const NATIVE_NEXT_ACTION: &str =
    "Coarsen the mesh (a larger target_size or maximum_deviation) or run on a machine with more memory.";

pub struct FeaExecutionPreflight {
    /// The check-time mesh and memory estimate; absent without a measured imported preview.
    pub estimate: Option<FeaCheckEstimate>,
    pub browser_supported: bool,
    pub desktop_reason: Option<&'static str>,
}

/// The solver mesh a study's settings are estimated to make, and the memory its static solve
/// is estimated to need. An estimate, not an allocation certificate: execution plans the exact
/// mesh and admits it again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeaCheckEstimate {
    pub element_count: u64,
    pub node_count: u64,
    /// Resident bytes of the static multigrid plan with the prepared mesh; absent for studies
    /// other than one linear static solid step.
    pub memory_bytes: Option<u64>,
    /// Scratch bytes of the same plan.
    pub scratch_bytes: Option<u64>,
    /// The study's authored resident ceiling; `u64::MAX` without one.
    pub authored_resident_bytes: u64,
}

impl FeaCheckEstimate {
    /// The admitted budget of a target offering `platform_bytes`: min(authored, platform).
    pub fn admitted_bytes(&self, platform_bytes: u64) -> u64 {
        self.authored_resident_bytes.min(platform_bytes)
    }

    /// The browser refusal when the estimate exceeds one numerical worker's budget.
    pub fn browser_limit(&self, instance_budget_bytes: u64) -> Option<BrowserResourceLimit> {
        let maximum = self.admitted_bytes(instance_budget_bytes);
        let check = |term, required: Option<u64>| {
            required.and_then(|required| BrowserResourceLimit::check(term, required, maximum).err())
        };
        check(BrowserResourceTerm::EstimatedMemory, self.memory_bytes)
            .or_else(|| check(BrowserResourceTerm::EstimatedScratch, self.scratch_bytes))
    }

    /// The native refusal when the estimate exceeds the memory this machine admits a run.
    pub fn native_limit(&self, machine_bytes: u64) -> Option<AnalysisResourceLimit> {
        let maximum = self.admitted_bytes(machine_bytes);
        [
            (BrowserResourceTerm::EstimatedMemory, self.memory_bytes),
            (BrowserResourceTerm::EstimatedScratch, self.scratch_bytes),
        ]
        .into_iter()
        .find_map(|(term, required)| {
            let required = required.filter(|required| *required > maximum)?;
            Some(AnalysisResourceLimit {
                code: RESOURCE_LIMIT_CODE.to_owned(),
                term: term.name().to_owned(),
                required_bytes: Some(required),
                maximum_bytes: maximum,
                next_action: NATIVE_NEXT_ACTION.to_owned(),
            })
        })
    }
}

impl CheckedFeaSessionSource {
    pub fn fea_execution_preflight(&self) -> Result<FeaExecutionPreflight, String> {
        let root = self.resolve_root(None).map_err(|e| e.to_string())?;
        let ResolvedFeaRoot::Study(resolved) = root else {
            return Ok(FeaExecutionPreflight {
                estimate: None,
                browser_supported: false,
                desktop_reason: Some("Run simulation groups in the desktop app."),
            });
        };
        let measure = match &resolved.geometry.source {
            GeometryDefinitionSource::Imported(imported) => imported.preview_measure.as_ref(),
            _ => None,
        };
        let counts =
            measure.and_then(|measure| solid_mesh_counts(measure, &resolved.mesh.settings));
        let element_estimate = |memory: Option<(u64, u64)>, authored: u64| {
            counts.map(|counts| FeaCheckEstimate {
                element_count: counts.tetrahedra,
                node_count: counts.nodes(),
                memory_bytes: memory.map(|(memory, _)| memory),
                scratch_bytes: memory.map(|(_, scratch)| scratch),
                authored_resident_bytes: authored,
            })
        };
        if !resolved.study.inputs.is_empty()
            || resolved.study.assessments.iter().any(|a| {
                matches!(
                    a.rule,
                    runmat_analysis_core::AssessmentRule::Function { .. }
                )
            })
        {
            return Ok(FeaExecutionPreflight {
                estimate: element_estimate(None, u64::MAX),
                browser_supported: false,
                desktop_reason: Some(
                    "Run simulations with custom input or assessment functions in the desktop app.",
                ),
            });
        }
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .map_err(|e| e.to_string())?;
        let supported = realized.sequence.steps.len() == 1
            && realized.sequence.steps.iter().all(|step| {
                step.family == runmat_analysis_core::PhysicsFamily::Structural
                    && step.procedure == runmat_analysis_core::AnalysisProcedure::Static
            })
            && matches!(
                resolved.geometry.source,
                GeometryDefinitionSource::Imported(_) | GeometryDefinitionSource::Constructed(_)
            );
        let authored = realized
            .sequence
            .steps
            .iter()
            .map(|step| step.accuracy.resources.resident_byte_budget())
            .min()
            .unwrap_or(u64::MAX);
        let memory = counts
            .filter(|_| supported)
            .and_then(|counts| estimate_static_solid_from_counts(counts).ok())
            .map(|plan| (plan.memory_bytes, plan.scratch_bytes));
        Ok(FeaExecutionPreflight {
            estimate: element_estimate(memory, authored),
            browser_supported: supported,
            desktop_reason: (!supported).then_some("Run this simulation type in the desktop app."),
        })
    }
}

fn solid_mesh_counts(
    measure: &runmat_geometry_core::PreviewMeasure,
    settings: &MeshingRequestSettings,
) -> Option<runmat_meshing_core::SolidMeshCounts> {
    (settings.dimension == MeshDimension::Solid && settings.cross_section.is_none())
        .then(|| estimate_solid_mesh_size(measure, settings))
        .flatten()
        .map(|estimate| estimate.counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(memory: u64, scratch: u64, authored: u64) -> FeaCheckEstimate {
        FeaCheckEstimate {
            element_count: 124_318,
            node_count: 182_210,
            memory_bytes: Some(memory),
            scratch_bytes: Some(scratch),
            authored_resident_bytes: authored,
        }
    }

    #[test]
    fn the_estimate_is_refused_against_min_of_authored_and_platform() {
        let fits = estimate(1_000, 500, u64::MAX);
        assert!(fits.browser_limit(1_000).is_none());
        assert!(fits.native_limit(1_000).is_none());
        let browser = estimate(2_000, 500, u64::MAX).browser_limit(1_000).unwrap();
        assert_eq!(browser.term, BrowserResourceTerm::EstimatedMemory);
        assert_eq!(
            (browser.required_bytes, browser.maximum_bytes),
            (Some(2_000), 1_000)
        );
        let contract = browser.resource_limit();
        assert_eq!(contract.code, super::super::BROWSER_RESOURCE_LIMIT_CODE);
        assert_eq!(contract.term, "estimated_memory");
        // The authored ceiling binds below the platform.
        let native = estimate(900, 950, 800).native_limit(4_000).unwrap();
        assert_eq!(native.code, RESOURCE_LIMIT_CODE);
        assert_eq!(
            (native.term.as_str(), native.required_bytes),
            ("estimated_memory", Some(900))
        );
        assert_eq!(native.maximum_bytes, 800);
        let scratch = estimate(500, 2_000, u64::MAX).native_limit(1_000).unwrap();
        assert_eq!(scratch.term, "estimated_scratch");
        let unknown = FeaCheckEstimate {
            memory_bytes: None,
            scratch_bytes: None,
            ..fits
        };
        assert!(unknown.browser_limit(1).is_none() && unknown.native_limit(1).is_none());
    }
}
