//! Fast advisory planning from checked source and existing preview measurements.
//! No geometry materialization, solver mesh, assembly, or numerical solve occurs here.
use super::CheckedFeaSessionSource;
use runmat_geometry_core::GeometryDefinitionSource;
use runmat_runtime::analysis::ResolvedFeaRoot;

pub struct FeaExecutionPreflight {
    /// A coarse boxed-grid workspace estimate, never an allocation certificate.
    pub approximate_memory_bytes: Option<u64>,
    pub browser_supported: bool,
    pub desktop_reason: Option<&'static str>,
}
impl CheckedFeaSessionSource {
    pub fn fea_execution_preflight(&self) -> Result<FeaExecutionPreflight, String> {
        let root = self.resolve_root(None).map_err(|e| e.to_string())?;
        let ResolvedFeaRoot::Study(resolved) = root else {
            return Ok(FeaExecutionPreflight {
                approximate_memory_bytes: None,
                browser_supported: false,
                desktop_reason: Some("Run simulation groups in the desktop app."),
            });
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
                approximate_memory_bytes: None,
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
        let supported = resolved.study.inputs.is_empty()
            && !resolved.study.assessments.iter().any(|a| {
                matches!(
                    a.rule,
                    runmat_analysis_core::AssessmentRule::Function { .. }
                )
            })
            && realized.sequence.steps.len() == 1
            && realized.sequence.steps.iter().all(|step| {
                step.family == runmat_analysis_core::PhysicsFamily::Structural
                    && step.procedure == runmat_analysis_core::AnalysisProcedure::Static
            })
            && matches!(
                resolved.geometry.source,
                GeometryDefinitionSource::Imported(_) | GeometryDefinitionSource::Constructed(_)
            );
        let extent = match &resolved.geometry.source {
            GeometryDefinitionSource::Imported(imported) => imported.preview_extent_m,
            _ => None,
        };
        Ok(FeaExecutionPreflight {
            approximate_memory_bytes: supported
                .then(|| {
                    approximate_workspace(
                        extent?,
                        resolved
                            .mesh
                            .settings
                            .target_edge_length_m
                            .canonical_value(),
                        resolved.mesh.settings.element_order,
                    )
                })
                .flatten(),
            browser_supported: supported,
            desktop_reason: (!supported).then_some("Run this simulation type in the desktop app."),
        })
    }
}

// Reference tetrahedral grid over the preview's bounding box. It intentionally
// includes empty space; curvature, local refinement and factorization may change
// actual demand in either direction. This is UI guidance, not a mesh-size bound.
fn approximate_workspace(
    extent: [f64; 3],
    edge: f64,
    order: runmat_meshing_core::ElementOrder,
) -> Option<u64> {
    if !edge.is_finite() || edge <= 0.0 || extent.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return None;
    }
    let cells = extent.map(|size| (size / edge).ceil().max(1.0));
    let quadratic = order == runmat_meshing_core::ElementOrder::Tet10;
    let nodes = cells.iter().map(|n| n + 1.0).product::<f64>() * if quadratic { 8.0 } else { 1.0 };
    let tetrahedra = cells.iter().product::<f64>() * 6.0;
    // A 3-DOF reference CSR stencil (27 neighbors), row pointers, 12 vectors,
    // mesh coordinates/connectivity, and two resident working copies.
    let nodal = 3.0 * (81.0 * 16.0 + 8.0 + 12.0 * 8.0) + 24.0;
    let connectivity = if quadratic { 80.0 } else { 32.0 };
    let bytes = (nodes * nodal + tetrahedra * connectivity) * 2.0;
    if !bytes.is_finite() {
        return Some(9_007_199_254_740_991);
    }
    Some(bytes.ceil().clamp(1.0, 9_007_199_254_740_991.0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refinement_and_order_increase_advisory_memory_and_invalid_scale_is_unknown() {
        use runmat_meshing_core::ElementOrder::*;
        let coarse = approximate_workspace([0.1; 3], 0.01, Tet4).unwrap();
        let fine = approximate_workspace([0.1; 3], 0.005, Tet4).unwrap();
        assert!(fine > coarse * 6);
        assert!(approximate_workspace([0.1; 3], 0.01, Tet10).unwrap() > coarse * 5);
        assert_eq!(approximate_workspace([f64::NAN; 3], 0.01, Tet4), None);
        assert_eq!(approximate_workspace([0.1; 3], 0.0, Tet4), None);
        assert!(
            approximate_workspace([f64::MAX; 3], 1e-10, Tet10).unwrap() <= 9_007_199_254_740_991
        );
    }
}
