use std::collections::BTreeSet;

use runmat_analysis_core::{
    Force, GeneratorDomain, GeneratorEffectPolicy, GeneratorProvenance, PhysicalSignal, Signal,
    SignalApproximationEvidence, SignalArtifactDomain, SignalArtifactRef, SignalExtrapolation,
    SignalGenerator, SignalId, SignalInterpolation, SignalOrdinate, SignalPrecision,
    SignalProvenance, SignalRepresentation, SignalSamplingPolicy, SignalShape,
};
use runmat_analysis_fea::execution::{GeneratorIdentity, SignalMaterializationIdentity};
use runmat_execution::value::{
    DenseValue, ElementType, InlineValue, ValueAdmissionOwnership, ValueAdmissionReceipt,
    ValueAdmissionTransfer, ValueCopyAccounting, ValuePayload,
    VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, BundleCallable, ProgramExecutionObject};
use runmat_types::{
    MutationFact, NumericClass, NumericDomain, NumericFact, ProgramFunctionId, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use crate::{
    canonical_tabulated_signal_content_digest, materialize_signal_for_solver,
    validate_signal_input_payload,
};

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn signal(shape: SignalShape, precision: SignalPrecision) -> PhysicalSignal {
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("load".into()),
        shape,
        frame: None,
        representation: SignalRepresentation::Generator(SignalGenerator {
            callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                qualified_name: "loads.pulse".into(),
            },
            arguments: vec![GeneratorArgument::Real(2.0)],
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 5,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
            effects: GeneratorEffectPolicy::Pure,
        }),
        precision,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    })
}

fn admitted() -> AdmittedSignalGenerator {
    let revision = ProgramRevision::new(
        digest("graph"),
        digest("source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap();
    AdmittedSignalGenerator {
        program_revision: revision,
        callable: BundleCallable {
            owner_identity: "project:loads".into(),
            qualified_name: "loads.pulse".into(),
            source_digest: digest("loads.m"),
        },
        executable_source_digest: digest("executable-source"),
        function: ProgramFunctionId(7),
        captures: Vec::new(),
        generator_identity: GeneratorIdentity::new(
            digest("callable"),
            digest("captures"),
            digest("generator"),
        )
        .unwrap(),
        materialization_identity: SignalMaterializationIdentity::new(
            digest("signal"),
            None,
            digest("sampling"),
        )
        .unwrap(),
        sample_points: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        required_capabilities: BTreeSet::new(),
        provenance: GeneratorProvenance {
            project_revision: "project-revision".into(),
            callable_id: "callable-id".into(),
            frozen_input_digest: "input-digest".into(),
        },
    }
}

fn dense_f64(shape: Vec<u64>, values: &[f64]) -> ValuePayload {
    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape,
        little_endian_data: values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
    })))
}

fn moved_real64_receipt(shape: Vec<Option<usize>>) -> ValueAdmissionReceipt {
    let mut target = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    );
    target.mutation = MutationFact::Immutable;
    ValueAdmissionReceipt {
        schema_version: VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
        source: target.clone(),
        target,
        ownership: ValueAdmissionOwnership::Moved,
        conversion: None,
        transfer: ValueAdmissionTransfer::None,
        copies: ValueCopyAccounting::default(),
    }
}

fn output(chunk: &SignalGeneratorChunk, value: ValuePayload) -> SignalGeneratorChunkOutput {
    SignalGeneratorChunkOutput {
        chunk_identity: chunk.identity,
        value,
    }
}

#[test]
fn planning_chunks_large_histories_and_preserves_typed_argument_order() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted(),
        None,
        RESULT_ENVELOPE_BYTES + 2 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].first_sample, 0);
    assert_eq!(chunks[1].first_sample, 2);
    assert_eq!(chunks[2].first_sample, 4);
    assert_eq!(
        chunks[0].sample_points().collect::<Vec<_>>(),
        vec![0.0, 0.25]
    );
    assert_eq!(chunks[0].arguments.len(), 2);
    assert!(matches!(
        chunks[0].arguments[1],
        ValuePayload::Inline(ref value)
            if matches!(value.as_ref(), InlineValue::F64Bits(bits) if f64::from_bits(*bits) == 2.0)
    ));
    assert_ne!(chunks[0].identity, chunks[1].identity);
}

#[test]
fn planning_places_frozen_closure_captures_before_the_sample_axis() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let mut admitted = admitted();
    admitted.captures = vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
        7.0_f64.to_bits(),
    )))];
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 16 * 8,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap();
    assert!(chunks.chunks.iter().all(|chunk| {
        chunk.arguments.first() == admitted.captures.first()
            && matches!(
                chunk.arguments.get(1),
                Some(ValuePayload::Inline(value)) if matches!(value.as_ref(), InlineValue::Dense(_))
            )
    }));
    validate_signal_generator_chunks(&signal, &admitted, &chunks.chunks).unwrap();
}

#[test]
fn scalar_chunks_compose_into_one_canonical_realization() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 2 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let outputs = vec![
        output(&chunks[0], dense_f64(vec![2, 1], &[10.0, 20.0])),
        output(&chunks[1], dense_f64(vec![1, 2], &[30.0, 40.0])),
        output(&chunks[2], dense_f64(vec![1], &[50.0])),
    ];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &[], &chunks, &outputs)
            .unwrap();
    let CanonicalSignalValues::Real { values } = realization.values else {
        panic!("expected real realization")
    };
    assert_eq!(
        values.into_iter().map(f64::from_bits).collect::<Vec<_>>(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0]
    );
    assert_eq!(realization.evidence.sample_count, 5);
}

#[test]
fn matrix_output_is_reordered_from_column_major_to_sample_major() {
    let mut signal = signal(SignalShape::Vector3, SignalPrecision::Real64);
    let PhysicalSignal::Force(value) = &mut signal else {
        unreachable!()
    };
    value.frame = Some(runmat_analysis_core::CoordinateFrame::Global);
    let mut admitted = admitted();
    admitted.sample_points = vec![0.0, 1.0];
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 2 * 3 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let outputs = vec![output(
        &chunks[0],
        dense_f64(vec![2, 3], &[1.0, 2.0, 10.0, 20.0, 100.0, 200.0]),
    )];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &[], &chunks, &outputs)
            .unwrap();
    let CanonicalSignalValues::Real { values } = realization.values else {
        panic!("expected real realization")
    };
    assert_eq!(
        values.into_iter().map(f64::from_bits).collect::<Vec<_>>(),
        vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0]
    );
}

#[test]
fn canonicalization_rejects_substituted_chunks_wrong_shapes_and_nonfinite_values() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let mut substituted = vec![output(
        &chunks[0],
        dense_f64(vec![5, 1], &[1.0, 2.0, 3.0, 4.0, 5.0]),
    )];
    substituted[0].chunk_identity = digest("other-chunk");
    assert!(canonicalize_signal_generator_outputs(
        &signal,
        &admitted,
        None,
        &[],
        &chunks,
        &substituted,
    )
    .is_err());

    let wrong_shape = vec![output(
        &chunks[0],
        dense_f64(vec![1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]),
    )];
    assert!(canonicalize_signal_generator_outputs(
        &signal,
        &admitted,
        None,
        &[],
        &chunks,
        &wrong_shape,
    )
    .is_err());

    let nonfinite = vec![output(
        &chunks[0],
        dense_f64(vec![5, 1], &[1.0, 2.0, f64::NAN, 4.0, 5.0]),
    )];
    assert!(canonicalize_signal_generator_outputs(
        &signal,
        &admitted,
        None,
        &[],
        &chunks,
        &nonfinite,
    )
    .is_err());
}

#[test]
fn transported_chunk_plan_rejects_axis_and_authored_argument_tampering() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;

    let mut changed_axis = chunks.clone();
    changed_axis[0].sample_points[1] = 0.5_f64.to_bits();
    assert!(validate_signal_generator_chunks(&signal, &admitted, &changed_axis).is_err());

    let mut changed_argument = chunks;
    changed_argument[0].arguments[1] =
        ValuePayload::Inline(Box::new(InlineValue::F64Bits(3.0_f64.to_bits())));
    assert!(validate_signal_generator_chunks(&signal, &admitted, &changed_argument).is_err());
}

#[test]
fn artifact_authority_does_not_change_logical_chunk_identity() {
    let value = dense_f64(vec![2, 1], &[4.0, 8.0]);
    let first = ProgramExecutionObject::externalized_value(
        &ArtifactAccess {
            authorization_scope: "run-a".into(),
            encryption_context: digest("key-a"),
        },
        runmat_execution::value::ValueRefKind::DriverObject,
        value.clone(),
    )
    .unwrap();
    let second = ProgramExecutionObject::externalized_value(
        &ArtifactAccess {
            authorization_scope: "run-b".into(),
            encryption_context: digest("key-b"),
        },
        runmat_execution::value::ValueRefKind::DriverObject,
        value,
    )
    .unwrap();
    let mut signal = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let generator = match &mut signal {
        PhysicalSignal::Force(signal) => match &mut signal.representation {
            SignalRepresentation::Generator(generator) => generator,
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    generator.arguments = vec![GeneratorArgument::Artifact(SignalArtifactRef {
        artifact_id: "immutable-vector".into(),
        content_digest: first.reference.logical_digest.to_string(),
        schema: "runmat-value-payload-v1".into(),
        domain: SignalArtifactDomain::Constant,
        topology_digest: None,
        sample_count: 1,
    })];
    generator.effects = GeneratorEffectPolicy::DeclaredImmutableArtifacts;

    let first_chunks = plan_signal_generator_chunks(
        &signal,
        &admitted(),
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| {
            Ok(ResolvedSignalArtifact {
                value: ValuePayload::Object(Box::new(first.reference.clone())),
                admission: moved_real64_receipt(vec![Some(2), Some(1)]),
            })
        },
    )
    .unwrap();
    let second_chunks = plan_signal_generator_chunks(
        &signal,
        &admitted(),
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| {
            Ok(ResolvedSignalArtifact {
                value: ValuePayload::Object(Box::new(second.reference.clone())),
                admission: moved_real64_receipt(vec![Some(2), Some(1)]),
            })
        },
    )
    .unwrap();

    assert_ne!(
        first_chunks.chunks[0].arguments,
        second_chunks.chunks[0].arguments
    );
    assert_eq!(
        first_chunks.chunks[0].identity,
        second_chunks.chunks[0].identity
    );
    assert_eq!(
        first_chunks.input_admissions,
        second_chunks.input_admissions
    );
    validate_signal_generator_chunks(&signal, &admitted(), &first_chunks.chunks).unwrap();
    validate_signal_generator_chunks(&signal, &admitted(), &second_chunks.chunks).unwrap();
}

#[test]
fn immutable_input_receipt_must_match_the_closed_payload_shape() {
    let payload = dense_f64(vec![2, 1], &[17.0, 19.0]);
    let admission = SignalInputAdmission {
        artifact_id: "immutable-vector".into(),
        logical_digest: payload.logical_digest().unwrap(),
        value_schema: runmat_execution::value::EXTERNALIZED_VALUE_SCHEMA.into(),
        receipt: moved_real64_receipt(vec![Some(2), Some(1)]),
    };
    validate_signal_input_payload(&admission, &payload).unwrap();

    let mut substituted = admission;
    substituted.receipt.target.shape = ShapeFact::from(vec![Some(1), Some(2)]);
    assert!(validate_signal_input_payload(&substituted, &payload).is_err());
}

#[test]
fn declared_single_precision_is_canonicalized_at_that_precision() {
    let signal = signal(SignalShape::Scalar, SignalPrecision::Real32);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let input = 1.0 / 3.0;
    let outputs = vec![output(&chunks[0], dense_f64(vec![5, 1], &[input; 5]))];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &[], &chunks, &outputs)
            .unwrap();
    let CanonicalSignalValues::Real { values } = realization.values else {
        panic!("expected real realization")
    };
    assert_eq!(f64::from_bits(values[0]), f64::from(input as f32));
}

#[test]
fn canonical_realization_becomes_a_typed_solver_table_without_a_callable() {
    let authored = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &authored,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let outputs = vec![output(
        &chunks[0],
        dense_f64(vec![5, 1], &[10.0, 20.0, 30.0, 40.0, 50.0]),
    )];
    let realization =
        canonicalize_signal_generator_outputs(&authored, &admitted, None, &[], &chunks, &outputs)
            .unwrap();
    let materialized = materialize_signal_for_solver(&authored, &admitted, &realization).unwrap();
    assert_eq!(
        canonical_tabulated_signal_content_digest(&materialized).unwrap(),
        realization.content_digest,
        "generated and tabulated forms must share one numerical content identity"
    );
    let PhysicalSignal::Force(signal) = materialized else {
        panic!("expected force signal")
    };
    let SignalRepresentation::Tabulated(table) = signal.representation else {
        panic!("solver input must be a typed table")
    };
    assert_eq!(
        table.axis,
        runmat_analysis_core::SignalAxis::TimeSeconds(vec![0.0, 0.25, 0.5, 0.75, 1.0])
    );
    assert!(matches!(
        &table.values[2],
        SignalOrdinate::Real(runmat_analysis_core::SignalValue::Scalar(value))
            if value.newtons == 30.0
    ));
    assert!(matches!(signal.provenance, SignalProvenance::Generated(_)));

    let mut substituted = admitted.clone();
    substituted.materialization_identity =
        SignalMaterializationIdentity::new(digest("different-signal"), None, digest("sampling"))
            .unwrap();
    assert!(materialize_signal_for_solver(&authored, &substituted, &realization).is_err());
}

#[test]
fn table_content_identity_rejects_semantics_a_generator_cannot_represent() {
    let authored = signal(SignalShape::Scalar, SignalPrecision::Real64);
    let admitted = admitted();
    let chunks = plan_signal_generator_chunks(
        &authored,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    let realization = canonicalize_signal_generator_outputs(
        &authored,
        &admitted,
        None,
        &[],
        &chunks,
        &[output(
            &chunks[0],
            dense_f64(vec![5, 1], &[10.0, 20.0, 30.0, 40.0, 50.0]),
        )],
    )
    .unwrap();
    let mut table = materialize_signal_for_solver(&authored, &admitted, &realization).unwrap();
    let PhysicalSignal::Force(signal) = &mut table else {
        unreachable!()
    };
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    table
        .breakpoints
        .push(runmat_analysis_core::SignalBreakpoint {
            abscissa: 0.5,
            side: runmat_analysis_core::BreakpointSide::Both,
        });
    assert!(
        canonical_tabulated_signal_content_digest(&PhysicalSignal::Force(signal.clone())).is_err()
    );
}
