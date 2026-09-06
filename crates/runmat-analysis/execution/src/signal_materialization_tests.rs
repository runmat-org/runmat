use std::collections::BTreeSet;

use runmat_analysis_core::{
    Force, GeneratorDomain, GeneratorEffectPolicy, GeneratorProvenance, PhysicalSignal, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalGenerator, SignalId,
    SignalInterpolation, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSamplingPolicy, SignalShape,
};
use runmat_analysis_fea::execution::{GeneratorIdentity, SignalMaterializationIdentity};
use runmat_execution::value::{DenseValue, ElementType, InlineValue, ValuePayload};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_types::ProgramFunctionId;

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
            qualified_function: "loads.pulse".into(),
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
        function: ProgramFunctionId(7),
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
            Vec::new(),
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
    .unwrap();
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
    .unwrap();
    let outputs = vec![
        output(&chunks[0], dense_f64(vec![2, 1], &[10.0, 20.0])),
        output(&chunks[1], dense_f64(vec![1, 2], &[30.0, 40.0])),
        output(&chunks[2], dense_f64(vec![1], &[50.0])),
    ];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &outputs).unwrap();
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
    .unwrap();
    let outputs = vec![output(
        &chunks[0],
        dense_f64(vec![2, 3], &[1.0, 2.0, 10.0, 20.0, 100.0, 200.0]),
    )];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &outputs).unwrap();
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
    .unwrap();
    let mut substituted = vec![output(
        &chunks[0],
        dense_f64(vec![5, 1], &[1.0, 2.0, 3.0, 4.0, 5.0]),
    )];
    substituted[0].chunk_identity = digest("other-chunk");
    assert!(
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &substituted,)
            .is_err()
    );

    let wrong_shape = vec![output(
        &chunks[0],
        dense_f64(vec![1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]),
    )];
    assert!(
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &wrong_shape,)
            .is_err()
    );

    let nonfinite = vec![output(
        &chunks[0],
        dense_f64(vec![5, 1], &[1.0, 2.0, f64::NAN, 4.0, 5.0]),
    )];
    assert!(
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &nonfinite,)
            .is_err()
    );
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
    .unwrap();
    let input = 1.0 / 3.0;
    let outputs = vec![output(&chunks[0], dense_f64(vec![5, 1], &[input; 5]))];
    let realization =
        canonicalize_signal_generator_outputs(&signal, &admitted, None, &chunks, &outputs).unwrap();
    let CanonicalSignalValues::Real { values } = realization.values else {
        panic!("expected real realization")
    };
    assert_eq!(f64::from_bits(values[0]), f64::from(input as f32));
}
