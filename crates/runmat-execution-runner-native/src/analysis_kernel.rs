use runmat_analysis_execution::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernel, AnalysisKernelInvocation,
    AnalysisKernelOutput,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, StructuralLinearWorkload,
    StructuralLinearWorkloadResult,
};
use runmat_analysis_fea::{run_linear_static_with_options, LinearStaticSolveOptions};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAnalysisKernel;

impl AnalysisKernel for NativeAnalysisKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        if invocation.kind != AnalysisWorkKind::Solve {
            return Err(AnalysisBridgeError::Invalid(
                "native analysis kernel does not yet implement this work kind".into(),
            ));
        }
        let state = unique_input(&invocation, AnalysisNumericalObjectKind::SemanticState)?;
        let mesh = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
        if invocation.inputs.len() != 2
            || invocation.host.workload.outputs != vec![AnalysisNumericalObjectKind::SemanticState]
        {
            return Err(AnalysisBridgeError::Invalid(
                "structural linear solve requires one semantic state, one mesh, and one semantic-state output".into(),
            ));
        }
        let solve = StructuralLinearWorkload::from_canonical_bytes(&state.payload())?;
        let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let run = run_linear_static_with_options(
            &solve.model,
            solve.backend,
            LinearStaticSolveOptions {
                formulation_requirement: solve.formulation_requirement,
                preconditioner_kind: solve.preconditioner,
                algebra_backend_kind: solve.algebra_backend,
                solver_mesh_artifact_path: None,
                solver_mesh: Some(mesh),
                require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
                thermo_mechanical_context: solve.thermo_mechanical_context,
                electro_thermal_context: solve.electro_thermal_context,
            },
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let payload = StructuralLinearWorkloadResult {
            schema_version: 1,
            run,
        }
        .canonical_bytes()?;
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::SemanticState,
            logical_identity: invocation.host.workload.execution_plan.digest(),
            payload,
        }])
    }
}

fn unique_input<'a>(
    invocation: &'a AnalysisKernelInvocation<'_>,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&'a runmat_analysis_execution::PreparedAnalysisObjects> {
    let mut matches = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == kind);
    let input = matches.next().ok_or_else(|| {
        AnalysisBridgeError::Invalid(format!("analysis workload is missing its {kind:?} input"))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisBridgeError::Invalid(format!(
            "analysis workload has multiple {kind:?} inputs"
        )));
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_analysis_execution::{
        AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
        AnalysisWorkloadRequest,
    };
    use runmat_analysis_fea::execution::{
        AccuracyIdentity, ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
        NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity, StepIdentity,
        StudyIdentity,
    };
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::value::ValueRefKind;
    use runmat_execution::Digest;
    use runmat_execution_artifact::object::ObjectInventoryLimits;
    use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    use super::*;

    fn digest(label: &str) -> Digest {
        Digest::sha256(label)
    }

    fn identities() -> (StudyIdentity, ExecutionPlanIdentity) {
        let study = StudyIdentity::new(digest("study"), None).unwrap();
        let sequence = SequenceIdentity::new(study, digest("sequence")).unwrap();
        let step = StepIdentity::new(sequence, digest("step")).unwrap();
        let constraints =
            ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
        let problem =
            ProblemIdentity::new(step, digest("model"), None, constraints, Vec::new()).unwrap();
        let discretization =
            DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
        let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
        let accuracy = AccuracyIdentity::new(step, digest("accuracy")).unwrap();
        let numerical = NumericalPlanIdentity::new(operator, digest("solver"), accuracy).unwrap();
        (
            study,
            ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap(),
        )
    }

    #[test]
    fn native_kernel_executes_path_free_structural_linear_workload() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let solve = StructuralLinearWorkload {
            schema_version: 1,
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            require_solver_mesh_for_solid: false,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        };
        let state = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::SemanticState,
            digest("state"),
            &solve.canonical_bytes().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        for (face_index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance.push(PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: if face_index == 0 { "root" } else { "tip" }.into(),
                assembly_path: vec!["root".into()],
            });
            face.provenance.sort();
        }
        if let Some(assignment) = solve.model.material_assignments.first() {
            let region = PersistentEntityId {
                kind: PersistentEntityKind::Region,
                source_topology_id: assignment.region_id.clone(),
                assembly_path: vec!["root".into()],
            };
            mesh.topology.volume_elements[0].region_id = region.clone();
            mesh.topology.regions[0].region_id = region;
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        let mesh = runmat_analysis_execution::prepare_analysis_objects(
            AnalysisNumericalObjectKind::Mesh,
            digest("mesh"),
            &mesh.canonical_encode().unwrap(),
            4096,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let inputs = vec![state, mesh];
        let workload_inputs = inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect();
        let (study, execution_plan) = identities();
        let workload = AnalysisWorkloadRequest::new(
            study,
            execution_plan,
            AnalysisWorkKind::Solve,
            workload_inputs,
            vec![AnalysisNumericalObjectKind::SemanticState],
            BTreeSet::from([AnalysisCapabilityRequirement::Host {
                abi: runmat_analysis_execution::ANALYSIS_HOST_ABI.into(),
            }]),
        )
        .unwrap();
        let host = AnalysisHostWorkload::new(
            workload,
            ArtifactAccess {
                authorization_scope: "study:test".into(),
                encryption_context: digest("encryption"),
            },
        )
        .unwrap();
        let outputs = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::Solve,
                host: &host,
                inputs: &inputs,
            })
            .unwrap();
        let result =
            StructuralLinearWorkloadResult::from_canonical_bytes(&outputs[0].payload).unwrap();
        assert!(!result.run.fields_are_empty());
        assert!(result
            .run
            .diagnostics
            .iter()
            .all(|diagnostic| !diagnostic.message.contains("artifact_path")));
    }
}
