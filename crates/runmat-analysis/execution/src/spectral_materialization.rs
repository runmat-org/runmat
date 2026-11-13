use runmat_analysis_core::{
    ComplexSpectralDensity, CrossPowerSpectralDensity, NumericalDataSource,
    RandomExcitationSpectrum, ResponseSpectrum, SignalArtifactRef,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionObject};

use crate::signal_layout::canonical_sample_layout;
use crate::signal_materialization::{decode_numeric_output, NumericValues};
use crate::signal_program::{
    rebind_generator_inputs, validate_closed_generator_inputs, validate_prepared_generator_program,
};
use crate::{
    plan_generator_chunks, validate_generator_chunks, AdmittedNumericalGenerator,
    AnalysisBridgeError, AnalysisBridgeResult, GeneratorChunk, GeneratorChunkOutput,
    NumericalGeneratorKind, PlannedGenerator, PreparedGeneratorProgram, ResolvedSignalArtifact,
    SignalInputAdmission,
};

const CONTENT_PREFIX: &[u8] = b"runmat-analysis-spectral-content/v1\0";
const CONTENT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(4096, 32);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalGeneratorSource {
    ResponseSpectrum(Box<ResponseSpectrum>),
    RandomSpectrum(Box<RandomExcitationSpectrum>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedNumericalMaterialization {
    pub authored: NumericalGeneratorSource,
    pub admitted: AdmittedNumericalGenerator,
    pub program: PreparedGeneratorProgram,
    pub input_objects: Vec<ProgramExecutionObject>,
    pub input_admissions: Vec<SignalInputAdmission>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalMaterializationExpectation {
    pub authored: NumericalGeneratorSource,
    pub admitted: AdmittedNumericalGenerator,
}

impl NumericalMaterializationExpectation {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        validate_source_admission(&self.authored, &self.admitted).map(|_| ())
    }
}

impl PreparedNumericalMaterialization {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        let invocation = validate_source_admission(&self.authored, &self.admitted)?;
        validate_generator_chunks(
            invocation,
            self.admitted.invocation.frozen_inputs.direct_captures(),
            self.admitted.materialization_identity.digest(),
            self.admitted.sample_points.canonical_values(),
            &self
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect::<Vec<_>>(),
        )?;
        validate_prepared_generator_program(&self.admitted, &self.program)?;
        validate_input_admission_inventory(invocation, &self.input_admissions)
    }

    pub fn validate_closed(&self) -> AnalysisBridgeResult<()> {
        self.validate()?;
        validate_closed_generator_inputs(&self.program, &self.input_objects, &self.input_admissions)
    }

    pub fn rebind(mut self, access: &ArtifactAccess) -> AnalysisBridgeResult<Self> {
        rebind_generator_inputs(&mut self.program, &mut self.input_objects, access)?;
        self.validate_closed()?;
        Ok(self)
    }

    pub fn expectation(&self) -> AnalysisBridgeResult<NumericalMaterializationExpectation> {
        self.validate()?;
        Ok(NumericalMaterializationExpectation {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
        })
    }
}

pub fn prepare_numerical_materialization(
    authored: NumericalGeneratorSource,
    admitted: AdmittedNumericalGenerator,
    executable: runmat_execution::ExecutableUnitEnvelope,
    chunks: Vec<GeneratorChunk>,
    mut input_objects: Vec<ProgramExecutionObject>,
    mut input_admissions: Vec<SignalInputAdmission>,
) -> AnalysisBridgeResult<PreparedNumericalMaterialization> {
    input_objects.sort_by_key(|object| object.reference.logical_digest);
    input_admissions.sort_by(|left, right| {
        (&left.artifact_id, left.logical_digest).cmp(&(&right.artifact_id, right.logical_digest))
    });
    let program = crate::prepare_generator_program(&admitted, executable, chunks, &input_objects)?;
    let prepared = PreparedNumericalMaterialization {
        authored,
        admitted,
        program,
        input_objects,
        input_admissions,
    };
    prepared.validate_closed()?;
    Ok(prepared)
}

fn validate_source_admission<'a>(
    source: &'a NumericalGeneratorSource,
    admitted: &AdmittedNumericalGenerator,
) -> AnalysisBridgeResult<&'a runmat_analysis_core::GeneratorInvocation> {
    admitted.invocation.program_revision.validate()?;
    match source {
        NumericalGeneratorSource::ResponseSpectrum(spectrum) => {
            matching_response_generator(spectrum, admitted)
        }
        NumericalGeneratorSource::RandomSpectrum(spectrum) => {
            matching_random_generator(spectrum, admitted)
        }
    }
}

pub fn plan_response_spectrum_generator_chunks(
    spectrum: &ResponseSpectrum,
    admitted: &AdmittedNumericalGenerator,
    maximum_inline_result_bytes: usize,
    resolve_artifact: impl FnMut(&SignalArtifactRef) -> AnalysisBridgeResult<ResolvedSignalArtifact>,
) -> AnalysisBridgeResult<PlannedGenerator> {
    let invocation = matching_response_generator(spectrum, admitted)?;
    plan_generator_chunks(
        invocation,
        admitted.invocation.frozen_inputs.direct_captures(),
        admitted.materialization_identity.digest(),
        admitted.sample_points.canonical_values(),
        1,
        maximum_inline_result_bytes,
        resolve_artifact,
    )
}

pub fn plan_random_spectrum_generator_chunks(
    spectrum: &RandomExcitationSpectrum,
    admitted: &AdmittedNumericalGenerator,
    maximum_inline_result_bytes: usize,
    resolve_artifact: impl FnMut(&SignalArtifactRef) -> AnalysisBridgeResult<ResolvedSignalArtifact>,
) -> AnalysisBridgeResult<PlannedGenerator> {
    let invocation = matching_random_generator(spectrum, admitted)?;
    let values_per_sample = spectrum
        .channels
        .len()
        .checked_mul(spectrum.channels.len())
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("random-spectrum matrix width overflows".into())
        })?;
    plan_generator_chunks(
        invocation,
        admitted.invocation.frozen_inputs.direct_captures(),
        admitted.materialization_identity.digest(),
        admitted.sample_points.canonical_values(),
        values_per_sample,
        maximum_inline_result_bytes,
        resolve_artifact,
    )
}

pub fn materialize_response_spectrum(
    authored: &ResponseSpectrum,
    admitted: &AdmittedNumericalGenerator,
    input_admissions: &[SignalInputAdmission],
    chunks: &[GeneratorChunk],
    outputs: &[GeneratorChunkOutput],
) -> AnalysisBridgeResult<ResponseSpectrum> {
    let invocation = matching_response_generator(authored, admitted)?;
    validate_outputs(invocation, admitted, chunks, outputs)?;
    let mut ordinates = Vec::with_capacity(admitted.sample_points.scalar_count());
    for (chunk, output) in chunks.iter().zip(outputs) {
        let decoded = decode_numeric_output(&output.value)?;
        validate_vector_shape(&decoded.dimensions, chunk.sample_points.len())?;
        let values = canonical_sample_layout(decoded.values, chunk.sample_points.len(), &[])?;
        match values {
            NumericValues::Real(values) => ordinates.extend(values),
            NumericValues::Complex(_) => {
                return Err(AnalysisBridgeError::Invalid(
                    "response-spectrum generator output must be real".into(),
                ))
            }
        }
    }
    validate_input_admission_inventory(invocation, input_admissions)?;
    let content_digest = content_digest("response-spectrum-ordinates", &ordinates)?;
    let mut materialized = authored.clone();
    materialized.ordinates = NumericalDataSource::Materialized {
        values: ordinates,
        provenance: admitted.invocation.provenance.clone(),
        content_digest: content_digest.to_string(),
    };
    materialized
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(materialized)
}

pub fn materialize_random_spectrum(
    authored: &RandomExcitationSpectrum,
    admitted: &AdmittedNumericalGenerator,
    input_admissions: &[SignalInputAdmission],
    chunks: &[GeneratorChunk],
    outputs: &[GeneratorChunkOutput],
) -> AnalysisBridgeResult<RandomExcitationSpectrum> {
    let invocation = matching_random_generator(authored, admitted)?;
    validate_outputs(invocation, admitted, chunks, outputs)?;
    let width = authored.channels.len();
    let mut matrices = Vec::with_capacity(admitted.sample_points.scalar_count());
    for (chunk, output) in chunks.iter().zip(outputs) {
        let sample_count = chunk.sample_points.len();
        let decoded = decode_numeric_output(&output.value)?;
        validate_matrix_shape(&decoded.dimensions, sample_count, width)?;
        let values = canonical_sample_layout(decoded.values, sample_count, &[width, width])?;
        append_spectral_matrices(&mut matrices, values, sample_count, width)?;
    }
    validate_input_admission_inventory(invocation, input_admissions)?;
    let content_digest = content_digest("random-spectrum-density", &matrices)?;
    let mut materialized = authored.clone();
    materialized.spectral_density = NumericalDataSource::Materialized {
        values: matrices,
        provenance: admitted.invocation.provenance.clone(),
        content_digest: content_digest.to_string(),
    };
    materialized
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(materialized)
}

pub fn canonical_materialized_response_spectrum_bytes(
    spectrum: &ResponseSpectrum,
) -> AnalysisBridgeResult<Vec<u8>> {
    spectrum
        .resolved_ordinates()
        .map_err(AnalysisBridgeError::Invalid)?;
    spectrum
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(encode_contract(
        CONTENT_PREFIX,
        "materialized-response-spectrum",
        spectrum,
        CONTENT_LIMITS,
    )?)
}

pub fn canonical_materialized_random_spectrum_bytes(
    spectrum: &RandomExcitationSpectrum,
) -> AnalysisBridgeResult<Vec<u8>> {
    spectrum
        .resolved_spectral_density()
        .map_err(AnalysisBridgeError::Invalid)?;
    spectrum
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(encode_contract(
        CONTENT_PREFIX,
        "materialized-random-spectrum",
        spectrum,
        CONTENT_LIMITS,
    )?)
}

pub fn decode_materialized_response_spectrum(
    bytes: &[u8],
) -> AnalysisBridgeResult<ResponseSpectrum> {
    let spectrum: ResponseSpectrum = decode_contract(
        CONTENT_PREFIX,
        "materialized-response-spectrum",
        bytes,
        CONTENT_LIMITS,
    )?;
    spectrum
        .resolved_ordinates()
        .map_err(AnalysisBridgeError::Invalid)?;
    spectrum
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(spectrum)
}

pub fn decode_materialized_random_spectrum(
    bytes: &[u8],
) -> AnalysisBridgeResult<RandomExcitationSpectrum> {
    let spectrum: RandomExcitationSpectrum = decode_contract(
        CONTENT_PREFIX,
        "materialized-random-spectrum",
        bytes,
        CONTENT_LIMITS,
    )?;
    spectrum
        .resolved_spectral_density()
        .map_err(AnalysisBridgeError::Invalid)?;
    spectrum
        .validate_intrinsic()
        .map_err(AnalysisBridgeError::Invalid)?;
    Ok(spectrum)
}

fn validate_outputs(
    invocation: &runmat_analysis_core::GeneratorInvocation,
    admitted: &AdmittedNumericalGenerator,
    chunks: &[GeneratorChunk],
    outputs: &[GeneratorChunkOutput],
) -> AnalysisBridgeResult<()> {
    validate_generator_chunks(
        invocation,
        admitted.invocation.frozen_inputs.direct_captures(),
        admitted.materialization_identity.digest(),
        admitted.sample_points.canonical_values(),
        chunks,
    )?;
    if chunks.len() != outputs.len()
        || chunks
            .iter()
            .zip(outputs)
            .any(|(chunk, output)| chunk.identity != output.chunk_identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "generator result inventory differs from its admitted plan",
        ));
    }
    Ok(())
}

fn matching_response_generator<'a>(
    spectrum: &'a ResponseSpectrum,
    admitted: &AdmittedNumericalGenerator,
) -> AnalysisBridgeResult<&'a runmat_analysis_core::GeneratorInvocation> {
    let invocation = spectrum.ordinates.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("response spectrum is not callable-backed".into())
    })?;
    if admitted.schema_version != 2
        || admitted.sample_points.descriptor()
            != crate::callable::spectrum_axis_descriptor(&spectrum.abscissa)?
        || admitted.sample_points.canonical_values().as_slice()
            != crate::callable::spectrum_axis(&spectrum.abscissa)
        || !matches!(
            &admitted.kind,
            NumericalGeneratorKind::ResponseSpectrum { spectrum_id }
                if spectrum_id == &spectrum.spectrum_id
        )
        || invocation.callable.display_name() != admitted.invocation.callable.qualified_name
    {
        return Err(AnalysisBridgeError::Identity(
            "response-spectrum generator differs from its admission",
        ));
    }
    Ok(invocation)
}

fn matching_random_generator<'a>(
    spectrum: &'a RandomExcitationSpectrum,
    admitted: &AdmittedNumericalGenerator,
) -> AnalysisBridgeResult<&'a runmat_analysis_core::GeneratorInvocation> {
    let invocation = spectrum.spectral_density.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("random spectrum is not callable-backed".into())
    })?;
    if admitted.schema_version != 2
        || admitted.sample_points.descriptor()
            != runmat_analysis_core::GeneratorDomain::FrequencyHertz
                .axis_descriptor(None)
                .map_err(AnalysisBridgeError::Invalid)?
        || admitted.sample_points.canonical_values() != spectrum.frequencies_hz.canonical_values()
        || !matches!(
            &admitted.kind,
            NumericalGeneratorKind::RandomSpectrum {
                excitation_id,
                channel_count,
            } if excitation_id == &spectrum.excitation_id && *channel_count == spectrum.channels.len()
        )
        || invocation.callable.display_name() != admitted.invocation.callable.qualified_name
    {
        return Err(AnalysisBridgeError::Identity(
            "random-spectrum generator differs from its admission",
        ));
    }
    Ok(invocation)
}

fn validate_vector_shape(dimensions: &[u64], samples: usize) -> AnalysisBridgeResult<()> {
    let samples = u64::try_from(samples)
        .map_err(|_| AnalysisBridgeError::Invalid("spectrum sample count exceeds u64".into()))?;
    if dimensions.is_empty() && samples == 1
        || dimensions == [samples]
        || dimensions == [samples, 1]
        || dimensions == [1, samples]
    {
        Ok(())
    } else {
        Err(AnalysisBridgeError::Invalid(format!(
            "response-spectrum generator output shape {dimensions:?} is not an N-element vector"
        )))
    }
}

fn validate_matrix_shape(
    dimensions: &[u64],
    samples: usize,
    width: usize,
) -> AnalysisBridgeResult<()> {
    let samples = u64::try_from(samples).map_err(|_| {
        AnalysisBridgeError::Invalid("random-spectrum sample count exceeds u64".into())
    })?;
    let width = u64::try_from(width).map_err(|_| {
        AnalysisBridgeError::Invalid("random-spectrum channel count exceeds u64".into())
    })?;
    let scalar = samples == 1 && width == 1 && dimensions.is_empty();
    let one_channel = width == 1
        && (dimensions == [samples] || dimensions == [samples, 1] || dimensions == [samples, 1, 1]);
    if scalar || one_channel || dimensions == [samples, width, width] {
        Ok(())
    } else {
        Err(AnalysisBridgeError::Invalid(format!(
            "random-spectrum generator output shape {dimensions:?} is not N-by-C-by-C"
        )))
    }
}

fn append_spectral_matrices(
    target: &mut Vec<CrossPowerSpectralDensity>,
    values: NumericValues,
    sample_count: usize,
    width: usize,
) -> AnalysisBridgeResult<()> {
    let values = match values {
        NumericValues::Real(values) => values
            .into_iter()
            .map(|value| ComplexSpectralDensity::new(value, 0.0))
            .collect::<Vec<_>>(),
        NumericValues::Complex(values) => values
            .into_iter()
            .map(|(real, imaginary)| ComplexSpectralDensity::new(real, imaginary))
            .collect::<Vec<_>>(),
    };
    let matrix_size = width.checked_mul(width).ok_or_else(|| {
        AnalysisBridgeError::Invalid("random-spectrum matrix size overflows".into())
    })?;
    for sample in 0..sample_count {
        let sample_values = &values[sample * matrix_size..(sample + 1) * matrix_size];
        target.push(CrossPowerSpectralDensity {
            values: runmat_types::physical::Quantity::new(sample_values.to_vec()),
        });
    }
    Ok(())
}

fn validate_input_admission_inventory(
    invocation: &runmat_analysis_core::GeneratorInvocation,
    admissions: &[SignalInputAdmission],
) -> AnalysisBridgeResult<()> {
    let declared = invocation
        .arguments
        .iter()
        .filter_map(|argument| match argument {
            runmat_analysis_core::GeneratorArgument::Artifact(artifact) => Some((
                artifact.artifact_id.as_str(),
                artifact.content_digest.as_str(),
            )),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let admitted = admissions
        .iter()
        .map(|admission| {
            admission.validate()?;
            Ok((
                admission.artifact_id.as_str(),
                admission.logical_digest.to_string(),
            ))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let admitted = admitted
        .iter()
        .map(|(id, digest)| (*id, digest.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    if declared.len() != admissions.len() || declared != admitted {
        return Err(AnalysisBridgeError::Identity(
            "numerical generator input admissions differ from authored artifact arguments",
        ));
    }
    Ok(())
}

fn content_digest<T: serde::Serialize>(
    domain: &'static str,
    values: &T,
) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        CONTENT_PREFIX,
        domain,
        values,
        CONTENT_LIMITS,
    )?))
}

#[cfg(test)]
#[path = "spectral_materialization_tests.rs"]
mod tests;
