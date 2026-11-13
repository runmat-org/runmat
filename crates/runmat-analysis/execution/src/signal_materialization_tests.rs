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
    let frame = if matches!(shape, SignalShape::Scalar) {
        None
    } else {
        Some(runmat_analysis_core::CoordinateFrame::Global)
    };
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("load".into()),
        shape,
        frame,
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: runmat_analysis_core::GeneratorInvocation {
                callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                    qualified_name: "loads.pulse".into(),
                },
                arguments: vec![GeneratorArgument::Real(2.0)],
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 5,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
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
        schema_version: 2,
        value_layout: crate::CanonicalGeneratorValueLayout::SampleMajorRowMajor,
        program_revision: revision,
        callable: BundleCallable {
            owner_identity: "project:loads".into(),
            qualified_name: "loads.pulse".into(),
            source_digest: digest("loads.m"),
        },
        executable_source_digest: digest("executable-source"),
        compiled_binding: crate::named_callable::CompiledCallableBinding::fixture(
            "loads.pulse",
            "loads.pulse",
            ProgramFunctionId(7),
            digest("executable-source"),
        ),
        function: ProgramFunctionId(7),
        frozen_inputs: crate::FrozenGeneratorInputs::DirectCaptures(Vec::new()),
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
        sample_points: runmat_types::physical::Quantity::try_from_descriptor(
            runmat_analysis_core::GeneratorDomain::TimeSeconds
                .axis_descriptor(None)
                .unwrap(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
        )
        .unwrap(),
        required_capabilities: BTreeSet::new(),
        provenance: GeneratorProvenance {
            project_revision: "project-revision".into(),
            callable_id: "callable-id".into(),
            frozen_input_digest: "input-digest".into(),
        },
    }
}

#[test]
fn frozen_admission_requires_explicit_canonical_value_layout() {
    let mut wire = serde_json::to_value(admitted()).unwrap();
    assert_eq!(wire["value_layout"], "sample_major_row_major");
    wire.as_object_mut().unwrap().remove("value_layout");
    assert!(serde_json::from_value::<AdmittedSignalGenerator>(wire.clone()).is_err());
    wire["value_layout"] = serde_json::Value::String("sample_major_column_major".into());
    assert!(serde_json::from_value::<AdmittedSignalGenerator>(wire).is_err());
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

fn output(chunk: &GeneratorChunk, value: ValuePayload) -> GeneratorChunkOutput {
    GeneratorChunkOutput {
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
    admitted.frozen_inputs =
        crate::FrozenGeneratorInputs::DirectCaptures(vec![ValuePayload::Inline(Box::new(
            InlineValue::F64Bits(7.0_f64.to_bits()),
        ))]);
    let chunks = plan_signal_generator_chunks(
        &signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 16 * 8,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap();
    assert!(chunks.chunks.iter().all(|chunk| {
        chunk.arguments.first() == admitted.frozen_inputs.direct_captures().first()
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
fn topology_expectation_authenticates_mesh_and_reopened_realization_rows() {
    use runmat_analysis_core::{BodyForceDensity, FieldLocation, Temperature};
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for thermal in [false, true] {
            let width = if thermal { 1 } else { 3 };
            let PhysicalSignal::Force(template) = signal(
                SignalShape::TopologyField {
                    topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
                    location: FieldLocation::Node,
                    component_count: width,
                },
                SignalPrecision::Real64,
            ) else {
                unreachable!()
            };
            let authored = if thermal {
                PhysicalSignal::Temperature(Signal::<Temperature> {
                    signal_id: template.signal_id,
                    shape: template.shape,
                    frame: template.frame,
                    representation: match template.representation {
                        SignalRepresentation::Generator(generator) => {
                            SignalRepresentation::Generator(generator)
                        }
                        _ => unreachable!(),
                    },
                    precision: template.precision,
                    evidence: template.evidence,
                    provenance: template.provenance,
                })
            } else {
                PhysicalSignal::BodyForceDensity(Signal::<BodyForceDensity> {
                    signal_id: template.signal_id,
                    shape: template.shape,
                    frame: template.frame,
                    representation: match template.representation {
                        SignalRepresentation::Generator(generator) => {
                            SignalRepresentation::Generator(generator)
                        }
                        _ => unreachable!(),
                    },
                    precision: template.precision,
                    evidence: template.evidence,
                    provenance: template.provenance,
                })
            };
            let admitted = admitted();
            let binding = crate::SignalTopologyBinding::derive(&authored, &mesh)
                .unwrap()
                .unwrap();
            let expected = crate::SignalMaterializationExpectation {
                authored: authored.clone(),
                admitted: admitted.clone(),
                topology_binding: Some(binding.clone()),
            };
            expected.validate_for_mesh(&mesh).unwrap();
            let mut invalid_contract = expected.clone();
            match &mut invalid_contract.authored {
                PhysicalSignal::Temperature(signal) => {
                    signal.evidence.max_absolute_error = Some(f64::NAN);
                }
                PhysicalSignal::BodyForceDensity(signal) => signal.frame = None,
                _ => unreachable!(),
            }
            assert!(invalid_contract.validate().is_err());
            let mut missing = expected.clone();
            missing.topology_binding = None;
            assert!(missing.validate().is_err());
            let foreign = canonical_tetrahedron_solver_mesh(if order == ElementOrder::Tet4 {
                ElementOrder::Tet10
            } else {
                ElementOrder::Tet4
            });
            assert!(expected.validate_for_mesh(&foreign).is_err());
            let planned = plan_signal_generator_chunks(
                &authored,
                &admitted,
                Some(binding.entity_count()),
                RESULT_ENVELOPE_BYTES + 5 * binding.scalar_component_count().unwrap() * 16,
                |_| panic!("no artifact argument expected"),
            )
            .unwrap();
            assert_eq!(planned.chunks.len(), 1);
            let shape = vec![5, binding.entity_count() as u64, width as u64];
            let outputs = [output(
                &planned.chunks[0],
                dense_f64(
                    shape,
                    &vec![2.0; 5 * binding.scalar_component_count().unwrap()],
                ),
            )];
            let realized = canonicalize_signal_generator_outputs(
                &authored,
                &admitted,
                Some(binding.entity_count()),
                &[],
                &planned.chunks,
                &outputs,
            )
            .unwrap();
            let reopened = CanonicalSignalRealization::from_canonical_bytes(
                &realized.canonical_bytes().unwrap(),
            )
            .unwrap();
            let reopened_expected: crate::SignalMaterializationExpectation =
                serde_json::from_slice(&serde_json::to_vec(&expected).unwrap()).unwrap();
            reopened_expected.validate_for_mesh(&mesh).unwrap();
            reopened_expected.validate_realization(&reopened).unwrap();
            materialize_signal_for_solver(&authored, &admitted, &reopened).unwrap();
            let mut wrong = reopened;
            wrong.topology_entity_count = Some(binding.entity_count() as u64 + 1);
            assert!(reopened_expected.validate_realization(&wrong).is_err());
        }
    }
}

#[test]
fn generated_nodal_vectors_preserve_each_node_component_and_sample() {
    use runmat_analysis_core::FieldLocation;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let nodes = mesh.topology.nodes.len();
        let authored = signal(
            SignalShape::TopologyField {
                topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
                location: FieldLocation::Node,
                component_count: 3,
            },
            SignalPrecision::Real64,
        );
        let mut admitted = admitted();
        admitted.sample_points = runmat_types::physical::Quantity::try_from_descriptor(
            GeneratorDomain::TimeSeconds.axis_descriptor(None).unwrap(),
            vec![0.0, 1.0],
        )
        .unwrap();
        let chunks = plan_signal_generator_chunks(
            &authored,
            &admitted,
            Some(nodes),
            RESULT_ENVELOPE_BYTES + 2 * nodes * 3 * 16,
            |_| panic!("no artifact argument expected"),
        )
        .unwrap()
        .chunks;
        // Literal MATLAB storage: sample varies fastest, then node, then component.
        let mut matlab = Vec::new();
        for component in 0..3 {
            for node in 0..nodes {
                for sample in 0..2 {
                    matlab.push((1000 * sample + 10 * node + component) as f64);
                }
            }
        }
        let outputs = [output(
            &chunks[0],
            dense_f64(vec![2, nodes as u64, 3], &matlab),
        )];
        let realized = canonicalize_signal_generator_outputs(
            &authored,
            &admitted,
            Some(nodes),
            &[],
            &chunks,
            &outputs,
        )
        .unwrap();
        let materialized = materialize_signal_for_solver(&authored, &admitted, &realized).unwrap();
        let PhysicalSignal::Force(value) = materialized else {
            unreachable!()
        };
        let SignalRepresentation::Tabulated(table) = value.representation else {
            unreachable!()
        };
        for (sample, knot) in table.knots.iter().enumerate() {
            let runmat_analysis_core::SignalKnotValue::Continuous(SignalOrdinate::Real(
                runmat_analysis_core::SignalValue::Tensor { shape, values },
            )) = &knot.value
            else {
                unreachable!()
            };
            assert_eq!(shape, &vec![nodes, 3]);
            for node in 0..nodes {
                for component in 0..3 {
                    assert_eq!(
                        values[3 * node + component].canonical_value(),
                        (1000 * sample + 10 * node + component) as f64
                    );
                }
            }
        }
    }
}

#[test]
fn matrix_output_is_reordered_from_column_major_to_sample_major() {
    let mut signal = signal(SignalShape::Vector3, SignalPrecision::Real64);
    let PhysicalSignal::Force(value) = &mut signal else {
        unreachable!()
    };
    value.frame = Some(runmat_analysis_core::CoordinateFrame::Global);
    let mut admitted = admitted();
    admitted.sample_points = runmat_types::physical::Quantity::try_from_descriptor(
        GeneratorDomain::TimeSeconds.axis_descriptor(None).unwrap(),
        vec![0.0, 1.0],
    )
    .unwrap();
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
    generator.invocation.arguments = vec![GeneratorArgument::Artifact(SignalArtifactRef {
        artifact_id: "immutable-vector".into(),
        content_digest: first.reference.logical_digest.to_string(),
        schema: "runmat-value-payload-v1".into(),
        domain: SignalArtifactDomain::Constant,
        topology_digest: None,
        sample_count: 1,
    })];
    generator.invocation.effects = GeneratorEffectPolicy::DeclaredImmutableArtifacts;

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
        table.domain,
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds
    );
    assert_eq!(
        table
            .knots
            .iter()
            .map(|knot| knot.abscissa)
            .collect::<Vec<_>>(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    assert!(matches!(
        &table.knots[2].value,
        runmat_analysis_core::SignalKnotValue::Continuous(SignalOrdinate::Real(runmat_analysis_core::SignalValue::Scalar(value)))
            if value.canonical_value() == 30.0
    ));
    assert!(matches!(signal.provenance, SignalProvenance::Generated(_)));

    let mut substituted = admitted.clone();
    substituted.materialization_identity =
        SignalMaterializationIdentity::new(digest("different-signal"), None, digest("sampling"))
            .unwrap();
    assert!(materialize_signal_for_solver(&authored, &substituted, &realization).is_err());
}

fn materialize_constant_vector_signal(
    authored: &PhysicalSignal,
    components: [f64; 3],
    identity_label: &str,
) -> PhysicalSignal {
    let mut generator_signal = authored.clone();
    let PhysicalSignal::Force(generator_template) =
        signal(SignalShape::Vector3, SignalPrecision::Real64)
    else {
        unreachable!()
    };
    let SignalRepresentation::Generator(mut generator) = generator_template.representation else {
        unreachable!()
    };
    generator.invocation.arguments = vec![GeneratorArgument::Real(components.iter().sum())];
    match &mut generator_signal {
        PhysicalSignal::Force(signal) => {
            signal.representation = SignalRepresentation::Generator(generator.clone())
        }
        PhysicalSignal::Moment(signal) => {
            signal.representation = SignalRepresentation::Generator(generator)
        }
        _ => panic!("the static beam test supports only force and moment signals"),
    }
    let mut admitted = admitted();
    admitted.generator_identity = GeneratorIdentity::new(
        digest(&format!("{identity_label}.callable")),
        digest(&format!("{identity_label}.captures")),
        digest(&format!("{identity_label}.generator")),
    )
    .unwrap();
    admitted.materialization_identity = SignalMaterializationIdentity::new(
        digest(&format!("{identity_label}.signal")),
        None,
        digest("sampling"),
    )
    .unwrap();
    let chunks = plan_signal_generator_chunks(
        &generator_signal,
        &admitted,
        None,
        RESULT_ENVELOPE_BYTES + 5 * 3 * 16,
        |_| panic!("no artifact argument expected"),
    )
    .unwrap()
    .chunks;
    // Dense generator output is column-major: five samples of each vector component.
    let values = components
        .into_iter()
        .flat_map(|component| std::iter::repeat_n(component, 5))
        .collect::<Vec<_>>();
    let outputs = vec![output(&chunks[0], dense_f64(vec![5, 3], &values))];
    let realization = canonicalize_signal_generator_outputs(
        &generator_signal,
        &admitted,
        None,
        &[],
        &chunks,
        &outputs,
    )
    .unwrap();
    let generated =
        materialize_signal_for_solver(&generator_signal, &admitted, &realization).unwrap();
    assert_eq!(
        canonical_tabulated_signal_content_digest(&generated).unwrap(),
        realization.content_digest,
    );
    generated
}

#[test]
fn generated_force_and_moment_match_explicit_static_beam_and_analytic_balance() {
    use runmat_analysis_fea::{
        excitation::SpatialProjectionBalance,
        fixtures::{fixture_model, FixtureId},
        run_linear_static_with_options, ComputeBackend, FeaRunResult, LinearStaticSolveOptions,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_ROTATION,
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    };

    fn field<'a>(result: &'a FeaRunResult, name: &str) -> &'a [f64] {
        result.field(name).unwrap().as_host_f64().unwrap()
    }

    let baseline_model = fixture_model(FixtureId::StructuralBeamForceAndMomentReference);
    assert_eq!(baseline_model.signals.len(), 2);
    let generated_signals = baseline_model
        .signals
        .iter()
        .map(|authored| match authored {
            PhysicalSignal::Force(_) => {
                materialize_constant_vector_signal(authored, [0.0, 500.0, 0.0], "tip.force")
            }
            PhysicalSignal::Moment(_) => {
                materialize_constant_vector_signal(authored, [0.0, 0.0, 90.0], "tip.moment")
            }
            _ => panic!("beam fixture has only force and moment"),
        })
        .collect::<Vec<_>>();
    assert!(generated_signals.iter().all(|signal| match signal {
        PhysicalSignal::Force(signal) =>
            matches!(signal.provenance, SignalProvenance::Generated(_)),
        PhysicalSignal::Moment(signal) => {
            matches!(signal.provenance, SignalProvenance::Generated(_))
        }
        _ => false,
    }));
    let mut explicit_signals = generated_signals.clone();
    for signal in &mut explicit_signals {
        match signal {
            PhysicalSignal::Force(signal) => signal.provenance = SignalProvenance::Authored,
            PhysicalSignal::Moment(signal) => signal.provenance = SignalProvenance::Authored,
            _ => unreachable!(),
        }
    }

    let mut explicit_model = baseline_model.clone();
    explicit_model.signals = explicit_signals;
    let mut generated_model = baseline_model.clone();
    generated_model.signals = generated_signals;

    let options = LinearStaticSolveOptions {
        ..LinearStaticSolveOptions::default()
    };
    let baseline =
        run_linear_static_with_options(&baseline_model, ComputeBackend::Cpu, options.clone())
            .expect("constant authored force and moment must solve");
    let explicit =
        run_linear_static_with_options(&explicit_model, ComputeBackend::Cpu, options.clone())
            .expect("explicit constant table must solve");
    let generated = run_linear_static_with_options(&generated_model, ComputeBackend::Cpu, options)
        .expect("verified generated constant tables must solve");

    for field_id in [
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_ROTATION,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    ] {
        let expected = baseline.field(field_id).expect("baseline field");
        assert_eq!(
            explicit.field(field_id),
            Some(expected),
            "explicit {field_id}"
        );
        assert_eq!(
            generated.field(field_id),
            Some(expected),
            "generated {field_id}"
        );
    }
    assert_eq!(
        explicit.spatial_projection_evidence,
        baseline.spatial_projection_evidence
    );
    assert_eq!(
        generated.spatial_projection_evidence,
        baseline.spatial_projection_evidence
    );
    assert_eq!(
        explicit.structural_static_balance,
        baseline.structural_static_balance
    );
    assert_eq!(
        generated.structural_static_balance,
        baseline.structural_static_balance
    );

    // Euler-Bernoulli cantilever, L=1 m and EI_z=(200e9 Pa)(6.4e-9 m^4)=1280 N m².
    // The independent tip response is F L³/(3EI)+M L²/(2EI),
    // theta=F L²/(2EI)+M L/EI, and stored energy is (F u + M theta)/2.
    let tip_y = 500.0 / (3.0 * 1280.0) + 90.0 / (2.0 * 1280.0);
    let tip_theta_z = 500.0 / (2.0 * 1280.0) + 90.0 / 1280.0;
    let energy = 0.5 * (500.0 * tip_y + 90.0 * tip_theta_z);
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-9,
            "actual={actual} expected={expected}"
        );
    };
    close(
        field(&generated, FEA_FIELD_STRUCTURAL_DISPLACEMENT)[4],
        tip_y,
    );
    close(
        field(&generated, FEA_FIELD_STRUCTURAL_ROTATION)[5],
        tip_theta_z,
    );
    close(
        field(&generated, FEA_FIELD_STRUCTURAL_REACTION_FORCE)[1],
        -500.0,
    );
    close(
        field(&generated, FEA_FIELD_STRUCTURAL_REACTION_MOMENT)[2],
        -590.0,
    );
    close(
        field(&generated, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0],
        energy,
    );

    assert_eq!(generated.spatial_projection_evidence.len(), 2);
    let mut projected_force_y = 0.0;
    let mut projected_moment_z = 0.0;
    for evidence in &generated.spatial_projection_evidence {
        evidence.validate().unwrap();
        assert!(evidence.accepted);
        let SpatialProjectionBalance::VectorResultant {
            projected_resultant,
            projected_moment,
            ..
        } = evidence.balance
        else {
            panic!("beam vector loading needs vector projection evidence")
        };
        projected_force_y += projected_resultant.real[1];
        // Projection moments are about each evidence reference point; shift to the
        // global origin before adding the force and direct moment contributions.
        projected_moment_z += projected_moment.real[2]
            + evidence.reference_point_m[0] * projected_resultant.real[1]
            - evidence.reference_point_m[1] * projected_resultant.real[0];
    }
    close(projected_force_y, 500.0);
    close(projected_moment_z, 590.0);
}

#[path = "signal_materialization_tests/solid_static.rs"]
mod solid_static;

#[path = "signal_materialization_tests/complex_frequency.rs"]
mod complex_frequency;
#[path = "signal_materialization_tests/volume_flow.rs"]
mod volume_flow;

#[test]
fn table_content_identity_includes_typed_jump_semantics() {
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
    let left = match table.knots[2].value.clone() {
        runmat_analysis_core::SignalKnotValue::Continuous(value) => value,
        _ => unreachable!(),
    };
    table.knots[2] = runmat_analysis_core::SignalKnot::jump(
        0.5,
        left,
        SignalOrdinate::Real(runmat_analysis_core::SignalValue::Scalar(Force::new(31.0))),
        runmat_analysis_core::SignalPointContinuity::Right,
    );
    let jump_digest =
        canonical_tabulated_signal_content_digest(&PhysicalSignal::Force(signal.clone())).unwrap();
    assert_ne!(jump_digest, realization.content_digest);
}
