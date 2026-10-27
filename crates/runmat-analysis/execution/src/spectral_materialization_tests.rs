use std::collections::BTreeSet;

use runmat_analysis_core::{
    CoordinateFrame, DirectionalCombination, GeneratorCallable, GeneratorEffectPolicy,
    GeneratorInvocation, GeneratorProvenance, ModalCombination, ModalTruncationCorrection,
    ModalTruncationPolicy, NumericalDataSource, RandomExcitationChannel, RandomExcitationChannelId,
    RandomExcitationId, RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, ResponseSpectrum, ResponseSpectrumId,
    RigidResponsePolicy, SpectralDensityConvention, SpectrumAbscissa, SpectrumConstruction,
    SpectrumDamping, SpectrumDirection, SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation,
    SpectrumInterpolation, SpectrumResponse,
};
use runmat_analysis_fea::execution::{GeneratorIdentity, NumericalMaterializationIdentity};
use runmat_execution::value::{DenseValue, ElementType, InlineValue, ValuePayload};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_types::ProgramFunctionId;

use super::*;
use crate::{AdmittedGeneratorInvocation, GeneratorChunkOutput};

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn invocation(name: &str) -> GeneratorInvocation {
    GeneratorInvocation {
        callable: GeneratorCallable::ProjectFunction {
            qualified_name: name.into(),
        },
        arguments: Vec::new(),
        effects: GeneratorEffectPolicy::Pure,
    }
}

fn admitted(
    name: &str,
    kind: NumericalGeneratorKind,
    sample_points: Vec<f64>,
) -> AdmittedNumericalGenerator {
    let generator_identity =
        GeneratorIdentity::new(digest("callable"), digest("captures"), digest(name)).unwrap();
    AdmittedNumericalGenerator {
        schema_version: 2,
        invocation: AdmittedGeneratorInvocation {
            program_revision: ProgramRevision::new(
                digest("graph"),
                digest("source"),
                ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab")
                    .unwrap(),
            )
            .unwrap(),
            callable: BundleCallable {
                owner_identity: "project:spectra".into(),
                qualified_name: name.into(),
                source_digest: digest("spectra.m"),
            },
            executable_source_digest: digest("executable"),
            compiled_binding: crate::named_callable::CompiledCallableBinding::fixture(
                name,
                name,
                ProgramFunctionId(11),
                digest("executable"),
            ),
            function: ProgramFunctionId(11),
            frozen_inputs: crate::FrozenGeneratorInputs::DirectCaptures(Vec::new()),
            generator_identity,
            required_capabilities: BTreeSet::new(),
            provenance: GeneratorProvenance {
                project_revision: "project-revision".into(),
                callable_id: "callable-id".into(),
                frozen_input_digest: "frozen-inputs".into(),
            },
        },
        materialization_identity: NumericalMaterializationIdentity::new(
            digest("authored-contract"),
            generator_identity,
        )
        .unwrap(),
        kind,
        sample_points: runmat_types::physical::Quantity::try_from_descriptor(
            runmat_analysis_core::GeneratorDomain::FrequencyHertz
                .axis_descriptor(None)
                .unwrap(),
            sample_points,
        )
        .unwrap(),
    }
}

fn response() -> ResponseSpectrum {
    ResponseSpectrum {
        spectrum_id: ResponseSpectrumId("design".into()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![5.0, 10.0, 20.0])),
        ordinates: NumericalDataSource::Generator(invocation("loads.design_spectrum")),
        damping: SpectrumDamping::Ratio(0.05),
        response: SpectrumResponse::AbsoluteAcceleration,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::Reject,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        },
        modal_combination: ModalCombination::Srss,
        directional_combination: DirectionalCombination::Srss,
        truncation: ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: ModalTruncationCorrection::None,
        },
        rigid_response: RigidResponsePolicy::Disabled,
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
    }
}

fn random() -> RandomExcitationSpectrum {
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("base".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(vec![10.0, 20.0]),
        channels: vec![
            RandomExcitationChannel {
                channel_id: RandomExcitationChannelId("x".into()),
                quantity: RandomExcitationQuantity::BaseAcceleration,
                direction: SpectrumDirection {
                    frame: CoordinateFrame::Global,
                    vector: [1.0, 0.0, 0.0],
                },
            },
            RandomExcitationChannel {
                channel_id: RandomExcitationChannelId("y".into()),
                quantity: RandomExcitationQuantity::BaseAcceleration,
                direction: SpectrumDirection {
                    frame: CoordinateFrame::Global,
                    vector: [0.0, 1.0, 0.0],
                },
            },
        ],
        spectral_density: NumericalDataSource::Generator(invocation("loads.random_spectrum")),
        convention: SpectralDensityConvention::OneSidedPerHertz,
        interpolation: RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: RandomSpectrumExtrapolation::Reject,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: (None)
                .map(runmat_types::physical::AccelerationPowerSpectralDensity::new),
            equivalent_degrees_of_freedom: None,
        },
    }
}

fn dense_real(shape: Vec<u64>, values: &[f64]) -> ValuePayload {
    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape,
        little_endian_data: values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
    })))
}

fn dense_complex(shape: Vec<u64>, values: &[(f64, f64)]) -> ValuePayload {
    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::ComplexF64,
        shape,
        little_endian_data: values
            .iter()
            .flat_map(|(real, imaginary)| {
                [
                    real.to_bits().to_le_bytes(),
                    imaginary.to_bits().to_le_bytes(),
                ]
                .concat()
            })
            .collect(),
    })))
}

fn output(chunk: &GeneratorChunk, value: ValuePayload) -> GeneratorChunkOutput {
    GeneratorChunkOutput {
        chunk_identity: chunk.identity,
        value,
    }
}

#[test]
fn response_spectrum_materializes_chunked_real_ordinates_and_round_trips() {
    let authored = response();
    let admitted = admitted(
        "loads.design_spectrum",
        NumericalGeneratorKind::ResponseSpectrum {
            spectrum_id: authored.spectrum_id.clone(),
        },
        vec![5.0, 10.0, 20.0],
    );
    let plan = plan_response_spectrum_generator_chunks(&authored, &admitted, 4096 + 2 * 16, |_| {
        panic!("no artifact arguments")
    })
    .unwrap();
    assert!(plan.chunks.len() > 1);
    let values = [1.2, 2.5, 0.7];
    let outputs = plan
        .chunks
        .iter()
        .map(|chunk| {
            let begin = chunk.first_sample;
            let end = begin + chunk.sample_points.len();
            output(
                chunk,
                dense_real(
                    vec![chunk.sample_points.len() as u64, 1],
                    &values[begin..end],
                ),
            )
        })
        .collect::<Vec<_>>();
    let materialized = materialize_response_spectrum(
        &authored,
        &admitted,
        &plan.input_admissions,
        &plan.chunks,
        &outputs,
    )
    .unwrap();
    assert_eq!(materialized.resolved_ordinates().unwrap(), values);
    let bytes = canonical_materialized_response_spectrum_bytes(&materialized).unwrap();
    assert_eq!(
        decode_materialized_response_spectrum(&bytes).unwrap(),
        materialized
    );
}

#[test]
fn random_spectrum_preserves_complex_matrix_order_and_rejects_nonhermitian_output() {
    let authored = random();
    let admitted = admitted(
        "loads.random_spectrum",
        NumericalGeneratorKind::RandomSpectrum {
            excitation_id: authored.excitation_id.clone(),
            channel_count: 2,
        },
        vec![10.0, 20.0],
    );
    let plan =
        plan_random_spectrum_generator_chunks(&authored, &admitted, 4096 + 2 * 4 * 16, |_| {
            panic!("no artifact arguments")
        })
        .unwrap();
    assert_eq!(plan.chunks.len(), 1);

    // MATLAB column-major N-by-C-by-C storage. Each sample is the Hermitian matrix
    // [[2, +0.5i], [-0.5i, 3]] (and a scaled second sample).
    let values = [
        (2.0, 0.0),
        (4.0, 0.0),
        (0.0, -0.5),
        (0.0, -1.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (3.0, 0.0),
        (6.0, 0.0),
    ];
    let outputs = vec![output(
        &plan.chunks[0],
        dense_complex(vec![2, 2, 2], &values),
    )];
    let materialized = materialize_random_spectrum(
        &authored,
        &admitted,
        &plan.input_admissions,
        &plan.chunks,
        &outputs,
    )
    .unwrap();
    let matrices = materialized.resolved_spectral_density().unwrap();
    assert_eq!(
        matrices[0].canonical_coefficient(2, 0, 1),
        ComplexSpectralDensity::new(0.0, 0.5)
    );
    assert_eq!(
        matrices[0].canonical_coefficient(2, 1, 0),
        ComplexSpectralDensity::new(0.0, -0.5)
    );
    let bytes = canonical_materialized_random_spectrum_bytes(&materialized).unwrap();
    assert_eq!(
        decode_materialized_random_spectrum(&bytes).unwrap(),
        materialized
    );

    let mut invalid = values;
    invalid[4] = (0.0, 0.25);
    let invalid_outputs = vec![output(
        &plan.chunks[0],
        dense_complex(vec![2, 2, 2], &invalid),
    )];
    assert!(materialize_random_spectrum(
        &authored,
        &admitted,
        &plan.input_admissions,
        &plan.chunks,
        &invalid_outputs,
    )
    .unwrap_err()
    .to_string()
    .contains("Hermitian"));
}

#[test]
fn numerical_materialization_rejects_wrong_shape_and_substituted_chunk_identity() {
    let authored = response();
    let admitted = admitted(
        "loads.design_spectrum",
        NumericalGeneratorKind::ResponseSpectrum {
            spectrum_id: authored.spectrum_id.clone(),
        },
        vec![5.0, 10.0, 20.0],
    );
    let plan = plan_response_spectrum_generator_chunks(&authored, &admitted, 4096 + 3 * 16, |_| {
        panic!("no artifact arguments")
    })
    .unwrap();
    let wrong_shape = vec![output(
        &plan.chunks[0],
        dense_real(vec![3, 2], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
    )];
    assert!(materialize_response_spectrum(
        &authored,
        &admitted,
        &plan.input_admissions,
        &plan.chunks,
        &wrong_shape,
    )
    .is_err());

    let substituted = vec![GeneratorChunkOutput {
        chunk_identity: digest("substituted"),
        value: dense_real(vec![3, 1], &[1.0, 2.0, 3.0]),
    }];
    assert!(materialize_response_spectrum(
        &authored,
        &admitted,
        &plan.input_admissions,
        &plan.chunks,
        &substituted,
    )
    .is_err());
}

#[test]
fn spectral_admission_rejects_axis_substitution_and_unversioned_receipts() {
    let authored = response();
    let admission = admitted(
        "loads.design_spectrum",
        NumericalGeneratorKind::ResponseSpectrum {
            spectrum_id: authored.spectrum_id.clone(),
        },
        vec![5.0, 10.0, 20.0],
    );
    let check = |admitted: &AdmittedNumericalGenerator| {
        plan_response_spectrum_generator_chunks(&authored, admitted, 8192, |_| {
            panic!("no artifact arguments")
        })
    };
    assert!(check(&admission).is_ok());
    let mut changed = admission.clone();
    changed.sample_points = runmat_types::physical::Quantity::try_from_descriptor(
        runmat_analysis_core::GeneratorDomain::TimeSeconds
            .axis_descriptor(None)
            .unwrap(),
        vec![5.0, 10.0, 20.0],
    )
    .unwrap();
    assert!(check(&changed).is_err());
    changed.sample_points = runmat_types::physical::Quantity::try_from_descriptor(
        admission.sample_points.descriptor(),
        vec![5.0, 11.0, 20.0],
    )
    .unwrap();
    assert!(check(&changed).is_err());
    let mut changed = admission.clone();
    changed.schema_version = 1;
    assert!(check(&changed).is_err());
    let mut wire = serde_json::to_value(&admission).unwrap();
    wire.as_object_mut().unwrap().remove("schema_version");
    assert!(serde_json::from_value::<AdmittedNumericalGenerator>(wire).is_err());
}
