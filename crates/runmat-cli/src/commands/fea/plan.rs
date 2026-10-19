use anyhow::Result;
use runmat_analysis_core::{
    AuthoredExecutionPolicy, AuthoredMaterialRef, AuthoredModel, AuthoredStudy, AuthoredSweep,
    MaterialModel,
};
use runmat_geometry_core::GeometryDefinition;
use runmat_meshing_core::MeshDefinition;
use runmat_runtime::analysis::ResolvedFeaRoot;
use serde::Serialize;

use crate::presentation;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum FeaPlanPhase {
    MaterializeGeometry,
    GenerateExactMesh,
    CompileStudy,
    PlanNumericalWorkloads,
    ExecuteProcedures,
    RecoverOutputs,
    PublishEvidence,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaLogicalPlan {
    schema_version: u16,
    root_kind: &'static str,
    root_id: String,
    study: AuthoredStudy,
    sweep: Option<AuthoredSweep>,
    sweep_case_count: Option<usize>,
    geometry: GeometryDefinition,
    geometry_dependencies: Vec<GeometryDefinition>,
    mesh: MeshDefinition,
    model_closure: Vec<AuthoredModel>,
    material_catalog: Vec<FeaPlanMaterial>,
    execution: Option<AuthoredExecutionPolicy>,
    phases: Vec<FeaPlanPhase>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
struct FeaPlanMaterial {
    source: AuthoredMaterialRef,
    definition: MaterialModel,
}

pub(super) fn print(root: &ResolvedFeaRoot, json: bool) -> Result<()> {
    let plan = logical_plan(root)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }
    let styles = presentation::stdout();
    println!(
        "{} {} {}",
        styles.success("FEA plan"),
        plan.root_kind,
        styles.identifier(&plan.root_id)
    );
    println!(
        "  {} {}::{}",
        styles.label("geometry:"),
        plan.geometry.module_path,
        plan.geometry.geometry_id
    );
    println!(
        "  {} {}::{}",
        styles.label("mesh:"),
        plan.mesh.module_path,
        plan.mesh.mesh_id
    );
    if let Some(count) = plan.sweep_case_count {
        println!("  {} {count}", styles.label("cases:"));
    }
    println!("  {}", styles.label("phases:"));
    for (index, phase) in plan.phases.iter().enumerate() {
        println!("    {}. {phase:?}", index + 1);
    }
    Ok(())
}

fn logical_plan(root: &ResolvedFeaRoot) -> Result<FeaLogicalPlan> {
    let (
        study,
        sweep,
        sweep_case_count,
        geometry,
        geometry_dependencies,
        mesh,
        model_closure,
        material_catalog,
        execution,
    ) = match root {
        ResolvedFeaRoot::Study(root) => (
            root.study.clone(),
            None,
            None,
            root.geometry.clone(),
            root.geometry_dependencies.clone(),
            root.mesh.clone(),
            root.models.values().cloned().collect(),
            root.material_catalog
                .iter()
                .map(|(source, definition)| FeaPlanMaterial {
                    source: source.clone(),
                    definition: definition.clone(),
                })
                .collect(),
            root.execution_policy.clone(),
        ),
        ResolvedFeaRoot::Sweep(root) => (
            root.study.study.clone(),
            Some(root.sweep.clone()),
            Some(root.sweep.expand_cases(&root.study.study)?.len()),
            root.study.geometry.clone(),
            root.study.geometry_dependencies.clone(),
            root.study.mesh.clone(),
            root.study.models.values().cloned().collect(),
            root.study
                .material_catalog
                .iter()
                .map(|(source, definition)| FeaPlanMaterial {
                    source: source.clone(),
                    definition: definition.clone(),
                })
                .collect(),
            root.study.execution_policy.clone(),
        ),
    };
    Ok(FeaLogicalPlan {
        schema_version: 1,
        root_kind: root.kind(),
        root_id: root.id().to_string(),
        study,
        sweep,
        sweep_case_count,
        geometry,
        geometry_dependencies,
        mesh,
        model_closure,
        material_catalog,
        execution,
        phases: vec![
            FeaPlanPhase::MaterializeGeometry,
            FeaPlanPhase::GenerateExactMesh,
            FeaPlanPhase::CompileStudy,
            FeaPlanPhase::PlanNumericalWorkloads,
            FeaPlanPhase::ExecuteProcedures,
            FeaPlanPhase::RecoverOutputs,
            FeaPlanPhase::PublishEvidence,
        ],
    })
}
