use std::collections::BTreeMap;

use crate::{
    AnalysisOperationCatalog, BoundFeaProgram, CheckedFeaDeclaration, ExecutionProfileSource,
    FeaDiagnostic, FeaSymbolKey, GeometryInspectionSource, LogicalSourcePath,
};

pub(super) struct LoweredProducts {
    pub geometry_inspections:
        BTreeMap<(LogicalSourcePath, String), super::CheckedGeometryInspection>,
    pub studies: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredStudy>,
    pub sweeps: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredSweep>,
    pub execution_policies:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredExecutionPolicy>,
    pub execution_bindings: BTreeMap<
        (LogicalSourcePath, FeaSymbolKey),
        runmat_analysis_core::AuthoredExecutionPolicyRef,
    >,
    pub materials: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::MaterialModel>,
    pub geometries:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_geometry_core::GeometryDefinition>,
    pub meshes: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_meshing_core::MeshDefinition>,
    pub models: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredModel>,
}

pub(super) fn lower_products<
    C: AnalysisOperationCatalog + GeometryInspectionSource + ExecutionProfileSource + ?Sized,
>(
    catalog: &C,
    bound: &BoundFeaProgram,
    declarations: &mut BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredProducts {
    let geometry_inspections = super::inspect_geometry_declarations(catalog, bound, diagnostics);
    let geometry_probe = super::lower_geometry_definitions(
        bound,
        declarations,
        &geometry_inspections,
        &mut Vec::new(),
    );
    super::cad_parameter_value::resolve(bound, declarations, &geometry_probe);
    let geometries =
        super::lower_geometry_definitions(bound, declarations, &geometry_inspections, diagnostics);
    let materials = super::lower_materials(bound, declarations, &geometry_inspections, diagnostics);
    let meshes = super::lower_mesh_definitions(bound, declarations, &geometries, diagnostics);
    let models = super::lower_models(bound, declarations, diagnostics);
    let execution_policies =
        super::execution_policy::lower_execution_policies(catalog, declarations, diagnostics);
    let studies = super::lower_studies(
        bound,
        declarations,
        &models,
        &geometries,
        &meshes,
        diagnostics,
    );
    let sweeps = super::lower_sweeps(bound, declarations, &studies, diagnostics);
    let execution_bindings = super::execution_binding::bind_execution_policies(
        bound,
        declarations,
        &execution_policies,
        &studies,
        &sweeps,
        diagnostics,
    );
    LoweredProducts {
        geometry_inspections,
        studies,
        sweeps,
        execution_policies,
        execution_bindings,
        materials,
        geometries,
        meshes,
        models,
    }
}
