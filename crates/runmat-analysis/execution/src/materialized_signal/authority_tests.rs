//! Numerical identity alone cannot authorize changed callable or immutable-input provenance.
use super::*;
use crate::callable::executable::{admit_named_signal_generator_from_executable, tests as fixture};
use runmat_analysis_core::{
    GeneratorArgument, GeneratorEffectPolicy, SignalArtifactDomain, SignalArtifactRef,
};
use runmat_analysis_fea::execution::{CanonicalSignalRealizationInput, SignalInputAdmission};
use runmat_execution::value::{
    ValueAdmissionOwnership, ValueAdmissionReceipt, ValueAdmissionTransfer, ValueCopyAccounting,
    EXTERNALIZED_VALUE_SCHEMA, VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
};
use runmat_types::{
    MutationFact, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeSet;

fn artifact_admission() -> SignalInputAdmission {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    fact.mutation = MutationFact::Immutable;
    SignalInputAdmission {
        artifact_id: "magnitude".into(),
        logical_digest: Digest::sha256("immutable-magnitude"),
        value_schema: EXTERNALIZED_VALUE_SCHEMA.into(),
        receipt: ValueAdmissionReceipt {
            schema_version: VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
            source: fact.clone(),
            target: fact,
            ownership: ValueAdmissionOwnership::Moved,
            conversion: None,
            transfer: ValueAdmissionTransfer::None,
            copies: ValueCopyAccounting::default(),
        },
    }
}

fn authority_fixture(
    with_repeated_artifact: bool,
) -> (
    PhysicalSignal,
    AdmittedSignalGenerator,
    CanonicalSignalRealization,
) {
    let mut authored = fixture::signal();
    let mut analysis = fixture::analysis();
    let mut input_admissions = Vec::new();
    if with_repeated_artifact {
        let admission = artifact_admission();
        admission.validate().unwrap();
        let artifact = SignalArtifactRef {
            artifact_id: admission.artifact_id.clone(),
            content_digest: admission.logical_digest.to_string(),
            schema: admission.value_schema.clone(),
            domain: SignalArtifactDomain::Constant,
            topology_digest: None,
            sample_count: 1,
        };
        let PhysicalSignal::Force(force) = &mut authored else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut force.representation else {
            unreachable!()
        };
        generator.invocation.arguments = vec![
            GeneratorArgument::Artifact(artifact.clone()),
            GeneratorArgument::Artifact(artifact),
        ];
        generator.invocation.effects = GeneratorEffectPolicy::DeclaredImmutableArtifacts;
        let scalar_parameter = analysis.functions[0].callable.outputs[0].clone();
        analysis.functions[0]
            .callable
            .parameters
            .extend([scalar_parameter.clone(), scalar_parameter]);
        input_admissions.push(admission);
    }
    let admitted = admit_named_signal_generator_from_executable(
        &authored,
        fixture::revision(),
        fixture::callable(),
        &fixture::envelope(&analysis),
        &BTreeSet::new(),
    )
    .unwrap();
    let generator = authored.generator().unwrap();
    let realization = CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
        materialization: admitted.materialization_identity,
        signal_id: authored.signal_id().clone(),
        quantity: authored.quantity(),
        shape: authored.shape().clone(),
        topology_entity_count: None,
        domain: generator.domain.clone(),
        knots: admitted
            .sample_points
            .canonical_values()
            .iter()
            .map(|point| CanonicalSignalKnot {
                abscissa: point.to_bits(),
                continuity: None,
            })
            .collect(),
        precision: authored.precision(),
        values: CanonicalSignalValues::real([1.0; 5]),
        interpolation: generator.interpolation,
        extrapolation: generator.extrapolation,
        evidence: runmat_analysis_core::SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 5,
        },
        provenance: admitted.provenance.clone(),
        input_admissions,
    })
    .unwrap();
    (authored, admitted, realization)
}

fn reopen(realization: &CanonicalSignalRealization) -> CanonicalSignalRealization {
    CanonicalSignalRealization::from_canonical_bytes(&realization.canonical_bytes().unwrap())
        .unwrap()
}

#[test]
fn cold_realization_cannot_change_provenance_without_changing_numerical_identity() {
    let (authored, admitted, realization) = authority_fixture(false);
    materialize_signal_for_solver(&authored, &admitted, &realization).unwrap();
    for field in 0..3 {
        let mut forged = realization.clone();
        match field {
            0 => forged.provenance.project_revision = "foreign-project".into(),
            1 => forged.provenance.callable_id = "foreign-callable".into(),
            _ => forged.provenance.frozen_input_digest = "foreign-inputs".into(),
        }
        let reopened = reopen(&forged);
        assert_eq!(reopened.identity, realization.identity);
        assert!(materialize_signal_for_solver(&authored, &admitted, &reopened).is_err());
    }
}

#[test]
fn cold_realization_rejects_spurious_admissions_for_artifact_free_generators() {
    let (authored, admitted, realization) = authority_fixture(false);
    let mut forged = realization.clone();
    forged.input_admissions.push(artifact_admission());
    let reopened = reopen(&forged);
    assert_eq!(reopened.identity, realization.identity);
    assert!(materialize_signal_for_solver(&authored, &admitted, &reopened).is_err());
}

#[test]
fn repeated_immutable_arguments_admit_one_exact_inventory_and_reject_missing_or_substitution() {
    let (authored, admitted, realization) = authority_fixture(true);
    assert_eq!(authored.generator().unwrap().invocation.arguments.len(), 2);
    assert_eq!(realization.input_admissions.len(), 1);
    materialize_signal_for_solver(&authored, &admitted, &reopen(&realization)).unwrap();
    for field in 0..3 {
        let mut forged = realization.clone();
        match field {
            0 => forged.input_admissions.clear(),
            1 => forged.input_admissions[0].artifact_id = "foreign-artifact".into(),
            _ => forged.input_admissions[0].logical_digest = Digest::sha256("foreign-content"),
        }
        let reopened = reopen(&forged);
        assert_eq!(reopened.identity, realization.identity);
        assert!(materialize_signal_for_solver(&authored, &admitted, &reopened).is_err());
    }
    let mut changed_schema = authored;
    let PhysicalSignal::Force(force) = &mut changed_schema else {
        unreachable!()
    };
    let SignalRepresentation::Generator(generator) = &mut force.representation else {
        unreachable!()
    };
    let GeneratorArgument::Artifact(artifact) = &mut generator.invocation.arguments[1] else {
        unreachable!()
    };
    artifact.schema = "foreign-schema".into();
    assert!(materialize_signal_for_solver(&changed_schema, &admitted, &realization).is_err());
}
