use std::collections::BTreeMap;

use runmat_analysis_core::{
    GeneratorArgument, PhysicalSignal, SignalApproximationEvidence, SignalArtifactRef,
    SignalPrecision, SignalShape,
};
use runmat_analysis_fea::execution::{
    CanonicalSignalKnot, CanonicalSignalRealization, CanonicalSignalRealizationInput,
    CanonicalSignalValues,
};
use runmat_execution::value::{DenseValue, ElementType, InlineValue, ValueLimits, ValuePayload};
use runmat_execution::Digest;

use crate::signal_input::resolve_signal_artifact;
use crate::{
    AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult, ResolvedSignalArtifact,
    SignalInputAdmission,
};

const CHUNK_IDENTITY_PREFIX: &[u8] = b"runmat-analysis-signal-chunk/v1\0";
pub(crate) const RESULT_ENVELOPE_BYTES: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorChunk {
    pub ordinal: u32,
    pub first_sample: usize,
    pub sample_points: Vec<u64>,
    pub arguments: Vec<ValuePayload>,
    pub identity: Digest,
}

impl GeneratorChunk {
    pub fn sample_points(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.sample_points
            .iter()
            .map(|point| f64::from_bits(*point))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratorChunkOutput {
    pub chunk_identity: Digest,
    pub value: ValuePayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedGenerator {
    pub chunks: Vec<GeneratorChunk>,
    pub input_admissions: Vec<SignalInputAdmission>,
}

/// Split one admitted generator into deterministic, independently schedulable calls.
///
/// The first argument is the chunk's canonical axis vector. Authored arguments follow in their
/// declared order. Artifact resolution remains with the runtime because it owns authorization and
/// encryption context; the analysis bridge only verifies the returned portable value contract.
pub fn plan_signal_generator_chunks(
    signal: &PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
    topology_entity_count: Option<usize>,
    maximum_inline_result_bytes: usize,
    resolve_artifact: impl FnMut(&SignalArtifactRef) -> AnalysisBridgeResult<ResolvedSignalArtifact>,
) -> AnalysisBridgeResult<PlannedGenerator> {
    let generator = matching_generator(signal, admitted)?;
    let component_count = signal
        .shape()
        .realized_component_count(topology_entity_count)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "signal shape cannot be realized against the selected topology".into(),
            )
        })?;
    plan_generator_chunks(
        &generator.invocation,
        admitted.frozen_inputs.direct_captures(),
        admitted.materialization_identity.digest(),
        admitted.sample_points.canonical_values(),
        component_count,
        maximum_inline_result_bytes,
        resolve_artifact,
    )
}

pub fn plan_generator_chunks(
    invocation: &runmat_analysis_core::GeneratorInvocation,
    captures: &[ValuePayload],
    materialization_digest: Digest,
    sample_points: &[f64],
    values_per_sample: usize,
    maximum_inline_result_bytes: usize,
    mut resolve_artifact: impl FnMut(&SignalArtifactRef) -> AnalysisBridgeResult<ResolvedSignalArtifact>,
) -> AnalysisBridgeResult<PlannedGenerator> {
    let result_bytes_per_sample = values_per_sample.checked_mul(16).ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal result width exceeds the platform bound".into())
    })?;
    let payload_budget = maximum_inline_result_bytes
        .checked_sub(RESULT_ENVELOPE_BYTES)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "signal result bound leaves no room for the execution envelope".into(),
            )
        })?;
    let samples_per_chunk = payload_budget / result_bytes_per_sample;
    if samples_per_chunk == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "one signal sample exceeds the inline execution result bound".into(),
        ));
    }

    let mut input_admissions = BTreeMap::new();
    let authored_arguments = invocation
        .arguments
        .iter()
        .map(|argument| encode_argument(argument, &mut resolve_artifact, &mut input_admissions))
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let mut chunks = Vec::new();
    for (ordinal, (first_sample, points)) in sample_points
        .chunks(samples_per_chunk)
        .enumerate()
        .map(|(ordinal, points)| (ordinal, (ordinal * samples_per_chunk, points)))
    {
        let ordinal = u32::try_from(ordinal).map_err(|_| {
            AnalysisBridgeError::Invalid("signal chunk count exceeds its identity bound".into())
        })?;
        let point_bits = points
            .iter()
            .map(|point| point.to_bits())
            .collect::<Vec<_>>();
        let mut arguments = Vec::with_capacity(captures.len() + authored_arguments.len() + 1);
        arguments.extend(captures.iter().cloned());
        arguments.push(axis_argument(points)?);
        arguments.extend(authored_arguments.iter().cloned());
        for argument in &arguments {
            argument.validate(ValueLimits::default())?;
        }
        let identity = chunk_identity(
            materialization_digest,
            ordinal,
            first_sample,
            &point_bits,
            &arguments,
        )?;
        chunks.push(GeneratorChunk {
            ordinal,
            first_sample,
            sample_points: point_bits,
            arguments,
            identity,
        });
    }
    if chunks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator produced no executable chunks".into(),
        ));
    }
    Ok(PlannedGenerator {
        chunks,
        input_admissions: input_admissions.into_values().collect(),
    })
}

/// Validate ordered chunk results and seal them into the solver-facing canonical artifact.
pub fn canonicalize_signal_generator_outputs(
    signal: &PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
    topology_entity_count: Option<usize>,
    input_admissions: &[SignalInputAdmission],
    chunks: &[GeneratorChunk],
    outputs: &[GeneratorChunkOutput],
) -> AnalysisBridgeResult<CanonicalSignalRealization> {
    validate_signal_generator_chunks(signal, admitted, chunks)?;
    let generator = matching_generator(signal, admitted)?;
    if chunks.len() != outputs.len()
        || chunks
            .iter()
            .zip(outputs)
            .any(|(chunk, output)| chunk.identity != output.chunk_identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "signal chunk result inventory differs from its admitted plan",
        ));
    }
    let component_count = signal
        .shape()
        .realized_component_count(topology_entity_count)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "signal shape cannot be realized against the selected topology".into(),
            )
        })?;
    let mut values = DecodedValues::new(signal.precision());
    let mut expected_first_sample = 0;
    for (chunk, output) in chunks.iter().zip(outputs) {
        if chunk.first_sample != expected_first_sample {
            return Err(AnalysisBridgeError::Identity(
                "signal chunks are not contiguous and ordered",
            ));
        }
        let sample_count = chunk.sample_points.len();
        let decoded = decode_chunk(
            &output.value,
            sample_count,
            signal.shape(),
            topology_entity_count,
            component_count,
        )?;
        values.extend(decoded)?;
        expected_first_sample = expected_first_sample
            .checked_add(sample_count)
            .ok_or_else(|| AnalysisBridgeError::Invalid("signal sample count overflows".into()))?;
    }
    if expected_first_sample != admitted.sample_points.scalar_count() {
        return Err(AnalysisBridgeError::Identity(
            "signal chunks do not cover every admitted sample",
        ));
    }
    let topology_entity_count = topology_entity_count
        .map(u64::try_from)
        .transpose()
        .map_err(|_| AnalysisBridgeError::Invalid("topology entity count exceeds u64".into()))?;
    Ok(CanonicalSignalRealization::new(
        CanonicalSignalRealizationInput {
            materialization: admitted.materialization_identity,
            signal_id: signal.signal_id().clone(),
            quantity: signal.quantity(),
            shape: signal.shape().clone(),
            topology_entity_count,
            domain: generator.domain.clone(),
            knots: admitted
                .sample_points
                .canonical_values()
                .iter()
                .copied()
                .map(CanonicalSignalKnot::continuous)
                .collect(),
            precision: signal.precision(),
            values: values.finish(),
            interpolation: generator.interpolation,
            extrapolation: generator.extrapolation,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: admitted.sample_points.scalar_count(),
            },
            provenance: admitted.provenance.clone(),
            input_admissions: input_admissions.to_vec(),
        },
    )?)
}

/// Revalidate a transported chunk plan against its authored signal and admitted callable.
pub fn validate_signal_generator_chunks(
    signal: &PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
    chunks: &[GeneratorChunk],
) -> AnalysisBridgeResult<()> {
    let generator = matching_generator(signal, admitted)?;
    validate_generator_chunks(
        &generator.invocation,
        admitted.frozen_inputs.direct_captures(),
        admitted.materialization_identity.digest(),
        admitted.sample_points.canonical_values(),
        chunks,
    )
}

pub fn validate_generator_chunks(
    invocation: &runmat_analysis_core::GeneratorInvocation,
    captures: &[ValuePayload],
    materialization_digest: Digest,
    sample_points: &[f64],
    chunks: &[GeneratorChunk],
) -> AnalysisBridgeResult<()> {
    if chunks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator produced no executable chunks".into(),
        ));
    }
    let mut expected_first_sample = 0_usize;
    for (ordinal, chunk) in chunks.iter().enumerate() {
        let expected_ordinal = u32::try_from(ordinal).map_err(|_| {
            AnalysisBridgeError::Invalid("signal chunk count exceeds its identity bound".into())
        })?;
        let end = expected_first_sample
            .checked_add(chunk.sample_points.len())
            .ok_or_else(|| AnalysisBridgeError::Invalid("signal sample count overflows".into()))?;
        let expected_points =
            sample_points
                .get(expected_first_sample..end)
                .ok_or(AnalysisBridgeError::Identity(
                    "signal chunks exceed the admitted sample axis",
                ))?;
        let axis_index = captures.len();
        if chunk.ordinal != expected_ordinal
            || chunk.first_sample != expected_first_sample
            || chunk.sample_points
                != expected_points
                    .iter()
                    .map(|point| point.to_bits())
                    .collect::<Vec<_>>()
            || chunk.arguments.get(..axis_index) != Some(captures)
            || chunk.arguments.get(axis_index) != Some(&axis_argument(expected_points)?)
            || chunk.arguments.len() != axis_index + invocation.arguments.len() + 1
        {
            return Err(AnalysisBridgeError::Identity(
                "signal chunk differs from its admitted sample or argument layout",
            ));
        }
        for (declared, encoded) in invocation
            .arguments
            .iter()
            .zip(&chunk.arguments[axis_index + 1..])
        {
            validate_encoded_argument(declared, encoded)?;
        }
        let identity = chunk_identity(
            materialization_digest,
            chunk.ordinal,
            chunk.first_sample,
            &chunk.sample_points,
            &chunk.arguments,
        )?;
        if chunk.identity != identity {
            return Err(AnalysisBridgeError::Identity(
                "signal chunk identity differs from its canonical plan",
            ));
        }
        expected_first_sample = end;
    }
    if expected_first_sample != sample_points.len() {
        return Err(AnalysisBridgeError::Identity(
            "signal chunks do not cover every admitted sample",
        ));
    }
    Ok(())
}

fn matching_generator<'a>(
    signal: &'a PhysicalSignal,
    admitted: &AdmittedSignalGenerator,
) -> AnalysisBridgeResult<&'a runmat_analysis_core::SignalGenerator> {
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    let descriptor = generator
        .domain
        .axis_descriptor(None)
        .map_err(AnalysisBridgeError::Invalid)?;
    if admitted.schema_version != 2 || admitted.sample_points.descriptor() != descriptor {
        return Err(AnalysisBridgeError::Identity(
            "signal admission axis descriptor differs from its source",
        ));
    }
    if generator.invocation.callable.display_name() != admitted.callable.qualified_name {
        return Err(AnalysisBridgeError::Identity(
            "signal generator differs from its admitted callable",
        ));
    }
    Ok(generator)
}

fn axis_argument(points: &[f64]) -> AnalysisBridgeResult<ValuePayload> {
    let rows = u64::try_from(points.len())
        .map_err(|_| AnalysisBridgeError::Invalid("signal chunk is too large".into()))?;
    Ok(ValuePayload::Inline(Box::new(InlineValue::Dense(
        DenseValue {
            element_type: ElementType::F64,
            shape: vec![rows, 1],
            little_endian_data: points
                .iter()
                .flat_map(|point| point.to_bits().to_le_bytes())
                .collect(),
        },
    ))))
}

fn encode_argument(
    argument: &GeneratorArgument,
    resolve_artifact: &mut impl FnMut(
        &SignalArtifactRef,
    ) -> AnalysisBridgeResult<ResolvedSignalArtifact>,
    input_admissions: &mut BTreeMap<(String, Digest), SignalInputAdmission>,
) -> AnalysisBridgeResult<ValuePayload> {
    let inline = match argument {
        GeneratorArgument::Boolean(value) => InlineValue::Logical(*value),
        GeneratorArgument::SignedInteger(value) => InlineValue::I64(*value),
        GeneratorArgument::UnsignedInteger(value) => InlineValue::U64(*value),
        GeneratorArgument::Real(value) => InlineValue::F64Bits(value.to_bits()),
        GeneratorArgument::Text(value) => InlineValue::String(value.clone().into()),
        GeneratorArgument::RealVector(values) => InlineValue::Dense(DenseValue {
            element_type: ElementType::F64,
            shape: vec![
                u64::try_from(values.len()).map_err(|_| {
                    AnalysisBridgeError::Invalid("generator vector argument is too large".into())
                })?,
                1,
            ],
            little_endian_data: values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        }),
        GeneratorArgument::Artifact(artifact) => {
            let resolved = resolve_artifact(artifact)?;
            let admission = resolve_signal_artifact(artifact, &resolved)?;
            let key = (admission.artifact_id.clone(), admission.logical_digest);
            if input_admissions
                .insert(key, admission.clone())
                .is_some_and(|existing| existing != admission)
            {
                return Err(AnalysisBridgeError::Identity(
                    "immutable signal input admission is inconsistent",
                ));
            }
            return Ok(resolved.value);
        }
    };
    Ok(ValuePayload::Inline(Box::new(inline)))
}

fn validate_encoded_argument(
    declared: &GeneratorArgument,
    encoded: &ValuePayload,
) -> AnalysisBridgeResult<()> {
    match declared {
        GeneratorArgument::Artifact(artifact) => {
            let expected = artifact
                .content_digest
                .parse::<Digest>()
                .map_err(AnalysisBridgeError::Platform)?;
            let ValuePayload::Object(reference) = encoded else {
                return Err(AnalysisBridgeError::Identity(
                    "signal artifact argument is not an immutable object reference",
                ));
            };
            if reference.logical_digest != expected {
                return Err(AnalysisBridgeError::Identity(
                    "signal artifact argument differs from its declared content",
                ));
            }
        }
        _ => {
            let expected = encode_argument(
                declared,
                &mut |_| {
                    Err(AnalysisBridgeError::Invalid(
                        "unexpected signal artifact argument".into(),
                    ))
                },
                &mut BTreeMap::new(),
            )?;
            if encoded != &expected {
                return Err(AnalysisBridgeError::Identity(
                    "signal argument differs from its authored value",
                ));
            }
        }
    }
    Ok(())
}

fn chunk_identity(
    materialization_digest: Digest,
    ordinal: u32,
    first_sample: usize,
    sample_points: &[u64],
    arguments: &[ValuePayload],
) -> AnalysisBridgeResult<Digest> {
    let logical_arguments = arguments
        .iter()
        .map(ValuePayload::logical_digest)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Digest::sha256(runmat_canonical_codec::encode_contract(
        CHUNK_IDENTITY_PREFIX,
        "signal-generator-chunk",
        &(
            materialization_digest,
            ordinal,
            first_sample,
            sample_points,
            logical_arguments,
        ),
        runmat_canonical_codec::CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64),
    )?))
}

enum DecodedValues {
    Real(Vec<f64>, SignalPrecision),
    Complex(Vec<(f64, f64)>, SignalPrecision),
}

impl DecodedValues {
    fn new(precision: SignalPrecision) -> Self {
        match precision {
            SignalPrecision::Real32 | SignalPrecision::Real64 => Self::Real(Vec::new(), precision),
            SignalPrecision::Complex64 | SignalPrecision::Complex128 => {
                Self::Complex(Vec::new(), precision)
            }
        }
    }

    fn extend(&mut self, decoded: NumericValues) -> AnalysisBridgeResult<()> {
        match (self, decoded) {
            (Self::Real(target, precision), NumericValues::Real(values)) => target.extend(
                values
                    .into_iter()
                    .map(|value| quantize_real(value, *precision)),
            ),
            (Self::Complex(target, precision), NumericValues::Real(values)) => target.extend(
                values
                    .into_iter()
                    .map(|value| (quantize_real(value, *precision), 0.0)),
            ),
            (Self::Complex(target, precision), NumericValues::Complex(values)) => {
                target.extend(values.into_iter().map(|(real, imaginary)| {
                    (
                        quantize_real(real, *precision),
                        quantize_real(imaginary, *precision),
                    )
                }))
            }
            (Self::Real(_, _), NumericValues::Complex(_)) => {
                return Err(AnalysisBridgeError::Invalid(
                    "complex generator output cannot realize a real signal".into(),
                ))
            }
        }
        Ok(())
    }

    fn finish(self) -> CanonicalSignalValues {
        match self {
            Self::Real(values, _) => CanonicalSignalValues::real(values),
            Self::Complex(values, _) => CanonicalSignalValues::complex(values),
        }
    }
}

fn quantize_real(value: f64, precision: SignalPrecision) -> f64 {
    match precision {
        SignalPrecision::Real32 | SignalPrecision::Complex64 => f64::from(value as f32),
        SignalPrecision::Real64 | SignalPrecision::Complex128 => value,
    }
}

pub(crate) enum NumericValues {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
}

fn decode_chunk(
    payload: &ValuePayload,
    sample_count: usize,
    shape: &SignalShape,
    topology_entity_count: Option<usize>,
    component_count: usize,
) -> AnalysisBridgeResult<NumericValues> {
    let output = decode_numeric_output(payload)?;
    if output.dimensions.is_empty() {
        if sample_count == 1 && component_count == 1 {
            return Ok(output.values);
        }
        return Err(AnalysisBridgeError::Invalid(
            "scalar generator output cannot realize multiple signal values".into(),
        ));
    }
    validate_result_shape(
        &output.dimensions,
        sample_count,
        shape,
        topology_entity_count,
    )?;
    let field_extents = [
        topology_entity_count.unwrap_or(0),
        match shape {
            SignalShape::TopologyField {
                component_count, ..
            } => *component_count,
            _ => 0,
        },
    ];
    let extents: &[usize] = match shape {
        SignalShape::Scalar => &[],
        SignalShape::Vector3 => &[3],
        SignalShape::Tensor(extents) => extents,
        SignalShape::TopologyField { .. } => &field_extents,
        SignalShape::DeferredTopologyField { .. } => {
            return Err(AnalysisBridgeError::Invalid(
                "deferred topology cannot decode generator output".into(),
            ));
        }
    };
    crate::signal_layout::canonical_sample_layout(output.values, sample_count, extents)
}

pub(crate) struct DecodedNumericOutput {
    pub dimensions: Vec<u64>,
    pub values: NumericValues,
}

pub(crate) fn decode_numeric_output(
    payload: &ValuePayload,
) -> AnalysisBridgeResult<DecodedNumericOutput> {
    payload.validate(ValueLimits::default())?;
    let ValuePayload::Inline(value) = payload else {
        return Err(AnalysisBridgeError::Invalid(
            "generator output must be materialized before canonicalization".into(),
        ));
    };
    let (dimensions, values) = match value.as_ref() {
        InlineValue::F64Bits(bits) => {
            (Vec::new(), NumericValues::Real(vec![f64::from_bits(*bits)]))
        }
        InlineValue::ComplexF64Bits { real, imaginary } => (
            Vec::new(),
            NumericValues::Complex(vec![(f64::from_bits(*real), f64::from_bits(*imaginary))]),
        ),
        InlineValue::Dense(dense) => (dense.shape.clone(), decode_dense(dense)?),
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "generator output must be a dense numeric value".into(),
            ))
        }
    };
    Ok(DecodedNumericOutput { dimensions, values })
}

fn validate_result_shape(
    dimensions: &[u64],
    sample_count: usize,
    shape: &SignalShape,
    topology_entity_count: Option<usize>,
) -> AnalysisBridgeResult<()> {
    let sample_count = u64::try_from(sample_count)
        .map_err(|_| AnalysisBridgeError::Invalid("signal sample count exceeds u64".into()))?;
    let expected_tail = match shape {
        SignalShape::DeferredTopologyField { .. } => {
            return Err(AnalysisBridgeError::Invalid(
                "deferred topology has no generator output shape".into(),
            ));
        }
        SignalShape::Scalar => Vec::new(),
        SignalShape::Vector3 => vec![3],
        SignalShape::Tensor(extents) => extents
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| AnalysisBridgeError::Invalid("signal extent exceeds u64".into()))?,
        SignalShape::TopologyField {
            component_count, ..
        } => vec![
            u64::try_from(topology_entity_count.ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "topology-field signal requires an entity count".into(),
                )
            })?)
            .map_err(|_| {
                AnalysisBridgeError::Invalid("topology entity count exceeds u64".into())
            })?,
            u64::try_from(*component_count).map_err(|_| {
                AnalysisBridgeError::Invalid("topology component count exceeds u64".into())
            })?,
        ],
    };
    let valid_scalar_vector = expected_tail.is_empty()
        && (dimensions == [sample_count]
            || dimensions == [sample_count, 1]
            || dimensions == [1, sample_count]);
    let valid_sample_leading = dimensions.first() == Some(&sample_count)
        && dimensions.get(1..) == Some(expected_tail.as_slice());
    if !valid_scalar_vector && !valid_sample_leading {
        return Err(AnalysisBridgeError::Invalid(format!(
            "signal generator output shape {dimensions:?} differs from sample-leading contract"
        )));
    }
    Ok(())
}

pub(crate) fn decode_dense(value: &DenseValue) -> AnalysisBridgeResult<NumericValues> {
    let bytes = &value.little_endian_data;
    let real = match value.element_type {
        ElementType::F32 => decode_words::<4>(bytes, |word| {
            f64::from(f32::from_bits(u32::from_le_bytes(word)))
        })?,
        ElementType::F64 => {
            decode_words::<8>(bytes, |word| f64::from_bits(u64::from_le_bytes(word)))?
        }
        ElementType::I8 => bytes.iter().map(|value| f64::from(*value as i8)).collect(),
        ElementType::U8 => bytes.iter().map(|value| f64::from(*value)).collect(),
        ElementType::I16 => decode_words::<2>(bytes, |word| f64::from(i16::from_le_bytes(word)))?,
        ElementType::U16 => decode_words::<2>(bytes, |word| f64::from(u16::from_le_bytes(word)))?,
        ElementType::I32 => decode_words::<4>(bytes, |word| f64::from(i32::from_le_bytes(word)))?,
        ElementType::U32 => decode_words::<4>(bytes, |word| f64::from(u32::from_le_bytes(word)))?,
        ElementType::I64 => decode_words::<8>(bytes, |word| i64::from_le_bytes(word) as f64)?,
        ElementType::U64 => decode_words::<8>(bytes, |word| u64::from_le_bytes(word) as f64)?,
        ElementType::ComplexF32 => {
            return decode_complex_words::<4>(bytes, |word| {
                f64::from(f32::from_bits(u32::from_le_bytes(word)))
            })
        }
        ElementType::ComplexF64 => {
            return decode_complex_words::<8>(bytes, |word| {
                f64::from_bits(u64::from_le_bytes(word))
            })
        }
        _ => {
            return Err(AnalysisBridgeError::Invalid(
                "signal generator output uses an unsupported numeric storage type".into(),
            ))
        }
    };
    if real.iter().any(|value| !value.is_finite()) {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator output contains a non-finite value".into(),
        ));
    }
    Ok(NumericValues::Real(real))
}

fn decode_words<const WIDTH: usize>(
    bytes: &[u8],
    decode: impl Fn([u8; WIDTH]) -> f64,
) -> AnalysisBridgeResult<Vec<f64>> {
    if !bytes.len().is_multiple_of(WIDTH) {
        return Err(AnalysisBridgeError::Invalid(
            "signal output bytes are not element aligned".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(WIDTH)
        .map(|word| decode(word.try_into().expect("exact chunk width")))
        .collect())
}

fn decode_complex_words<const WIDTH: usize>(
    bytes: &[u8],
    decode: impl Fn([u8; WIDTH]) -> f64,
) -> AnalysisBridgeResult<NumericValues> {
    let pair_width = WIDTH * 2;
    if !bytes.len().is_multiple_of(pair_width) {
        return Err(AnalysisBridgeError::Invalid(
            "complex signal output bytes are not element aligned".into(),
        ));
    }
    let values = bytes
        .chunks_exact(pair_width)
        .map(|pair| {
            let real = decode(pair[..WIDTH].try_into().expect("exact real width"));
            let imaginary = decode(pair[WIDTH..].try_into().expect("exact imaginary width"));
            (real, imaginary)
        })
        .collect::<Vec<_>>();
    if values
        .iter()
        .any(|(real, imaginary)| !real.is_finite() || !imaginary.is_finite())
    {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator output contains a non-finite complex value".into(),
        ));
    }
    Ok(NumericValues::Complex(values))
}

#[cfg(test)]
#[path = "signal_materialization_tests.rs"]
mod tests;
