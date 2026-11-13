//! Retained transport witness, not native-worker or independent continuum qualification.
use super::*;
use crate::{AnalysisHostResponse, AnalysisKernel, AnalysisKernelInvocation, AnalysisKernelOutput};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ElectricalNumericalReceiptArtifact,
};
use runmat_execution_artifact::{
    cache::CacheExport, ArtifactResult, LogicalObject, ObjectNamespace,
};
use std::collections::BTreeMap;

#[derive(Default)]
struct MemoryObjects(BTreeMap<Digest, Vec<u8>>);
impl CacheImport for MemoryObjects {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
            digest,
            self.0.get(&digest).map(Vec::as_slice),
            maximum,
            &mut |_| Ok(()),
        )
    }
}
impl CacheExport for MemoryObjects {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        if self
            .0
            .insert(object.descriptor.digest, object.bytes.clone())
            .is_some_and(|previous| previous != object.bytes)
        {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "test object collision".into(),
            ));
        }
        Ok(())
    }
}
struct AcceptedOutputs(Vec<crate::PreparedAnalysisObjects>);
impl AnalysisKernel for AcceptedOutputs {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::SolveElectricalStatic);
        Ok(self
            .0
            .iter()
            .cloned()
            .map(AnalysisKernelOutput::prepared)
            .collect())
    }
}

#[test]
fn exact_equation_order_result_receipt_profiles_survive_cold_import_and_reject_missing_objects() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut input = preparation();
            input.mesh = canonical_tetrahedron_solver_mesh(order);
            input.product.compiled.model.electromagnetic =
                Some(ElectromagneticDomain::Static { equation });
            input.product.compiled.properties =
                compile_property_map(&input.product.compiled.model, Some(&input.mesh)).unwrap();
            input.product.compiled.solver_mesh_digest = input.mesh.canonical_digest;
            input.product.workload =
                plan_electrical_study(&input.product.compiled, input.policy.clone()).unwrap();
            let workload = input.product.workload.clone();
            let result = workload.execute(&input.mesh).unwrap();
            let receipt =
                ElectricalNumericalReceiptArtifact::new(&workload, &result, &input.mesh).unwrap();
            let limits = input.output_policy.inventory_limits();
            let chunk = input.output_policy.chunk_bytes();
            let objects = |kind, identity, bytes: Vec<u8>| {
                crate::prepare_analysis_objects(
                    kind,
                    identity,
                    &bytes,
                    chunk,
                    ObjectNamespace::ResultValue,
                    limits,
                )
                .unwrap()
            };
            let result_objects = objects(
                AnalysisNumericalObjectKind::SolverResult,
                result.source_result_identity().unwrap().digest(),
                result.canonical_bytes().unwrap(),
            );
            let profiles = result
                .state_publication_profiles(result_objects.manifest.payload_digest)
                .unwrap();
            let accepted = AcceptedOutputs(vec![
                result_objects,
                objects(
                    AnalysisNumericalObjectKind::NumericalReceipt,
                    receipt.identity().digest(),
                    receipt.canonical_bytes().unwrap(),
                ),
                objects(
                    AnalysisNumericalObjectKind::StatePublicationProfiles,
                    profiles.identity(),
                    profiles.canonical_bytes().unwrap(),
                ),
            ]);
            let mut study = ElectricalStudyCoordinator::new(input).unwrap();
            let task = study.prepare_solve().unwrap();
            let mut store = MemoryObjects::default();
            task.program.write_to(&mut store).unwrap();
            let executed = crate::execute_serial_analysis(
                &task.program.request,
                &mut store,
                &accepted,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &executed)
                .unwrap()
                .program_response();
            // Copy only returned result closure, excluding producer inputs/program objects.
            let cold = MemoryObjects(
                executed
                    .result_objects
                    .iter()
                    .map(|object| {
                        (
                            object.logical_digest,
                            store.0[&object.logical_digest].clone(),
                        )
                    })
                    .collect(),
            );
            let imported =
                import_electrical_static_result(&cold, response.clone(), task, limits).unwrap();
            assert_eq!(imported.result, result);
            assert_eq!(imported.numerical_receipt, receipt);
            let foreign_payload = Digest::sha256("foreign retained payload");
            assert!(result.state_publication_profiles(foreign_payload).is_err());
            // Construct a metadata-valid catalog associated with another payload. The
            // retained-result helper must reject it above; import must independently
            // reject it even when an untrusted worker emits a well-formed object.
            let foreign_profiles = runmat_analysis_fea::execution::StatePublicationProfiles::new(
                result.producer(),
                result.source_result_identity().unwrap(),
                Digest::from_bytes(*result.mesh_digest().bytes()),
                foreign_payload,
                profiles.entries().to_vec(),
            )
            .unwrap();
            let mut foreign = accepted.0.clone();
            foreign[2] = objects(
                AnalysisNumericalObjectKind::StatePublicationProfiles,
                foreign_profiles.identity(),
                foreign_profiles.canonical_bytes().unwrap(),
            );
            let executed_foreign = crate::execute_serial_analysis(
                &task.program.request,
                &mut store,
                &AcceptedOutputs(foreign),
                chunk,
                limits,
            )
            .unwrap();
            let foreign_response = AnalysisHostResponse::completed(&task.host, &executed_foreign)
                .unwrap()
                .program_response();
            assert!(
                import_electrical_static_result(&store, foreign_response, task, limits).is_err()
            );
            for object in &executed.result_objects {
                let mut missing = MemoryObjects(cold.0.clone());
                missing.0.remove(&object.logical_digest);
                assert!(
                    import_electrical_static_result(&missing, response.clone(), task, limits)
                        .is_err()
                );
            }
            // Model the scheduler's admitted request: the numerical plan remains
            // exact, with the portable host/process capabilities added at submission.
            let mut allocation = task.resource_plan.request.clone();
            allocation
                .required_capabilities
                .extend(task.host.workload.generic_capabilities().unwrap());
            allocation
                .required_capabilities
                .insert(runmat_execution::resource::Capability::ProcessIsolation);
            let accounting =
                crate::tests::phase_execution(&task.host, &task.program, &allocation, &response);
            study.complete_solve(&cold, response, accounting).unwrap();
            // Native fields are recovered (and observations evaluated) before accuracy.
            assert_eq!(study.phase(), ElectricalStudyPhase::ReadyForFieldRecovery);
            assert_eq!(study.result().unwrap().result, result);
        }
    }
}
