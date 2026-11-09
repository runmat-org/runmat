//! Retained magnetic result authority and its exact native fields. Validation
//! compares borrowed values; it neither solves nor reconstructs missing physics.
use super::*;
use crate::execution::magnetic_static::fields::{
    native_magnetic_entity_counts, native_magnetic_field_definition, native_magnetic_fields,
};
use crate::execution::{FieldRecoveryRequest, FieldRecoverySource, MagnetostaticSolverResult};
use crate::results::field::{RecoveredFieldArtifact, RecoveredFieldSample};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalMagnetostaticStatePayload {
    pub result: Box<MagnetostaticSolverResult>,
    pub recovery_request: FieldRecoveryRequest,
    pub fields: RecoveredFieldArtifact,
}
impl CanonicalMagnetostaticStatePayload {
    pub fn validate_for(
        &self,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        producer: StepIdentity,
    ) -> AnalysisExecutionResult<()> {
        self.result.validate()?;
        self.recovery_request.validate()?;
        self.fields.validate().map_err(invalid)?;
        let expected_request = self.result.native_field_request_from_result()?;
        if self.result.producer() != producer
            || self.result.source_result_identity()? != source_result
            || expected_request.topology.discretization_digest() != topology_digest
            || self.recovery_request != expected_request
            || self.fields.fields().len() != expected_request.plan.requirements.len()
            || self.fields.source != FieldRecoverySource::SolverResult(source_result)
            || self.fields.topology_digest != topology_digest
            || self.fields.recovery_request != expected_request.identity()
            || self.fields.entity_counts != native_magnetic_entity_counts(&self.result)
        {
            return Err(invalid(
                "magnetic state retained authority/source/topology or native inventory differs",
            ));
        }
        for ((field, native), requirement) in self
            .fields
            .fields()
            .iter()
            .zip(native_magnetic_fields(&self.result))
            .zip(&expected_request.plan.requirements)
        {
            field_validation::validate_portable_field(field.storage())?;
            let expected = native_magnetic_field_definition(native.field_id, requirement)?;
            let values = field
                .storage()
                .as_host_f64()
                .ok_or_else(|| invalid("magnetic native fields must be host arrays"))?;
            if field.definition() != &expected
                || field.sample() != RecoveredFieldSample::Final
                || field.storage().shape != native.shape
                || values.iter().copied().ne(native.values.values())
            {
                return Err(invalid(
                    "magnetic state field definition, sampling or retained physical values differ",
                ));
            }
        }
        Ok(())
    }
}
impl CanonicalSemanticState {
    pub fn from_magnetostatic_publication(
        request: &SemanticStatePublicationRequest,
        result: &MagnetostaticSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_publication_request(request)?;
        let recovery_request = result.native_field_request_from_result()?;
        let fields = result.recover_native_fields_from_result(&recovery_request)?;
        let payload = CanonicalMagnetostaticStatePayload {
            result: Box::new(result.clone()),
            recovery_request,
            fields,
        };
        payload.validate_for(
            request.topology_digest,
            request.source_result,
            request.producer,
        )?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Magnetostatic {
                state: Box::new(payload),
            },
        })
    }
}
fn invalid(reason: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{MagnetostaticWorkload, SemanticStatePublicationProfileSource};
    use crate::physics::magnetic::test_support::authored_model;
    use crate::topology::test_meshes::slab_mesh;
    use runmat_analysis_core::{ProcedureStepId, StateId};

    fn result() -> MagnetostaticSolverResult {
        let mut model = authored_model(2.0);
        model.sequence.steps[0].state_outputs = ["first", "second"]
            .into_iter()
            .map(|id| StateOutput {
                state_id: StateId(id.into()),
                kind: StateKind::ElectromagneticStatic,
                completeness: StateCompleteness::Complete,
                persistence: StatePersistence::Canonical,
            })
            .collect();
        MagnetostaticWorkload {
            model,
            step_id: ProcedureStepId("potential".into()),
            relative_residual_target: 1e-10,
        }
        .execute(&slab_mesh(2))
        .unwrap()
    }

    #[test]
    fn magnetic_detached_publication_preserves_exact_fields_and_each_authored_declaration() {
        let result = result();
        let cold =
            MagnetostaticSolverResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
                .unwrap();
        drop(result);
        let requests = cold.state_publication_requests().unwrap();
        assert_eq!(requests.len(), 2);
        for request in &requests {
            let state =
                CanonicalSemanticState::from_magnetostatic_publication(request, &cold).unwrap();
            let bytes = state.canonical_bytes().unwrap();
            let imported = CanonicalSemanticState::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(imported, state);
            let payload = imported.payload.magnetostatic().unwrap();
            for (field, native) in payload
                .fields
                .fields()
                .iter()
                .zip(native_magnetic_fields(&cold))
            {
                assert!(field
                    .storage()
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .copied()
                    .eq(native.values.values()));
            }
            let borrowed = request
                .encoding_profile(
                    SemanticStatePublicationProfileSource::MagnetostaticNativeFields(&cold),
                )
                .unwrap();
            let actual = runmat_canonical_codec::profile_serialized(
                &state,
                CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
            )
            .unwrap();
            assert_eq!(borrowed, actual);
        }
        assert!(cold
            .state_publication_profiles(Digest::sha256(cold.canonical_bytes().unwrap()))
            .is_ok());
    }

    #[test]
    fn magnetic_publication_rejects_source_declaration_and_sampling_substitution() {
        let result = result();
        let request = result.state_publication_requests().unwrap().remove(0);
        let mut foreign = request.clone();
        foreign.declaration.state_id = StateId("invented".into());
        assert!(result.validate_publication_request(&foreign).is_err());
        foreign = request.clone();
        foreign.source = SemanticStatePublicationSource::StaticElectricalNativeFields;
        assert!(result.validate_publication_request(&foreign).is_err());
        foreign = request.clone();
        foreign.topology_digest = Digest::sha256(b"foreign sampling");
        assert!(result.validate_publication_request(&foreign).is_err());
        foreign = request.clone();
        foreign.source_result = SolverResultIdentity::from_step_result(
            request.producer,
            Digest::sha256(b"foreign result"),
        )
        .unwrap();
        assert!(result.validate_publication_request(&foreign).is_err());
        let state =
            CanonicalSemanticState::from_magnetostatic_publication(&request, &result).unwrap();
        let payload = state.payload.magnetostatic().unwrap();
        assert!(payload
            .validate_for(
                Digest::sha256(b"raw mesh is not sampled topology"),
                request.source_result,
                request.producer
            )
            .is_err());
    }

    #[test]
    fn magnetic_state_rejects_reidentified_but_physically_substituted_native_values() {
        let result = result();
        let request = result.state_publication_requests().unwrap().remove(0);
        let state =
            CanonicalSemanticState::from_magnetostatic_publication(&request, &result).unwrap();
        let mut payload = state.payload.magnetostatic().unwrap().clone();
        let mut fields = payload.fields.fields().to_vec();
        let original = &fields[0];
        let mut values = original.storage().as_host_f64().unwrap().to_vec();
        values[0] += 1.0;
        fields[0] = crate::results::field::RecoveredField::new(
            original.definition().clone(),
            original.sample(),
            runmat_analysis_core::AnalysisField::host_f64(
                original.definition().field_id(),
                original.definition().layout(),
                original.storage().shape.clone(),
                values,
            ),
        )
        .unwrap();
        payload.fields = RecoveredFieldArtifact::new(
            payload.fields.recovery_request,
            payload.fields.source,
            payload.fields.topology_digest,
            payload.fields.entity_counts,
            fields,
        )
        .unwrap();
        payload.fields.validate().unwrap();
        assert!(payload
            .validate_for(
                request.topology_digest,
                request.source_result,
                request.producer
            )
            .is_err());
    }
}
