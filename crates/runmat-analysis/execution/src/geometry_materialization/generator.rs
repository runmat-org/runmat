use std::collections::BTreeSet;

use runmat_analysis_fea::execution::GeneratorIdentity;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::value::{DenseValue, ElementType, InlineValue, ValueLimits, ValuePayload};
use runmat_execution::{Digest, ExecutableUnitEnvelope, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_geometry_core::{GeometryAuthoringValue, GeometryScalarExpression};
use runmat_geometry_ops::ResolvedGeneratedGeometryRequest;
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{
    CapabilityRequirement, NumericClass, NumericDomain, ShapeFact, ValueFact, ValueKindFact,
};

use crate::{
    prepare_generator_program, AdmittedGeneratorInvocation, AnalysisBridgeError,
    AnalysisBridgeResult, GeneratorChunk, GeneratorProgramAdmission, GeneratorProgramAdmissionRef,
    PreparedGeneratorProgram,
};

pub const GENERATED_GEOMETRY_RUNTIME_CLASS: &str = "geometry.Generated";
use crate::named_callable::CompiledCallableBinding;

const GENERATED_ADMISSION_PREFIX: &[u8] = b"runmat-generated-geometry-admission\0";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 100_000, 100_000, 64);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedGeometryGenerator {
    pub invocation: AdmittedGeneratorInvocation,
    pub request: ResolvedGeneratedGeometryRequest,
    pub arguments: Vec<ValuePayload>,
    pub materialization_identity: Digest,
}

impl AdmittedGeometryGenerator {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        self.invocation.program_revision.validate()?;
        self.invocation.compiled_binding.validate_admission(
            &self.invocation.callable,
            self.invocation.function,
            self.invocation.executable_source_digest,
        )?;
        validate_resolved_request(&self.request)?;
        if self.invocation.callable.owner_identity.trim().is_empty()
            || self.invocation.callable.qualified_name != self.request.function
            || self.invocation.callable.source_digest.bytes() != self.request.source_digest.bytes()
            || !self.invocation.frozen_inputs.direct_captures().is_empty()
            || self.invocation.frozen_inputs.graph().is_some()
            || self.arguments.len() != self.request.arguments.len()
        {
            return Err(AnalysisBridgeError::Invalid(
                "generated geometry admission is malformed".into(),
            ));
        }
        for argument in &self.arguments {
            argument.validate(ValueLimits::default())?;
        }
        let (callable_digest, invocation_digest, generator_identity, materialization_identity) =
            geometry_generator_identities(
                &self.invocation.program_revision,
                &self.invocation.callable,
                self.invocation.executable_source_digest,
                &self.invocation.compiled_binding,
                self.invocation.function,
                &self.request,
                &self.arguments,
            )?;
        let expected_provenance = runmat_analysis_core::GeneratorProvenance {
            project_revision: self.invocation.program_revision.canonical_identity(),
            callable_id: callable_digest.to_string(),
            frozen_input_digest: invocation_digest.to_string(),
        };
        if self.invocation.generator_identity != generator_identity
            || self.materialization_identity != materialization_identity
            || self.invocation.provenance != expected_provenance
        {
            return Err(AnalysisBridgeError::Identity(
                "generated geometry admission identity does not match its frozen invocation",
            ));
        }
        Ok(())
    }
}

impl GeneratorProgramAdmission for AdmittedGeometryGenerator {
    fn invocation_admission(&self) -> GeneratorProgramAdmissionRef<'_> {
        GeneratorProgramAdmissionRef {
            program_revision: &self.invocation.program_revision,
            callable: &self.invocation.callable,
            executable_source_digest: self.invocation.executable_source_digest,
            compiled_binding: &self.invocation.compiled_binding,
            function: self.invocation.function,
            generator_identity: self.invocation.generator_identity,
            required_capabilities: &self.invocation.required_capabilities,
            frozen_inputs: &self.invocation.frozen_inputs,
            provenance: &self.invocation.provenance,
        }
    }

    fn materialization_digest(&self) -> Digest {
        self.materialization_identity
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedGeometryGenerator {
    pub admitted: AdmittedGeometryGenerator,
    pub program: PreparedGeneratorProgram,
}

pub fn resolve_geometry_generator_request(
    definition: &runmat_geometry_core::GeometryDefinition,
    bindings: &std::collections::BTreeMap<String, GeometryAuthoringValue>,
) -> AnalysisBridgeResult<ResolvedGeneratedGeometryRequest> {
    let runmat_geometry_core::GeometryDefinitionSource::Generated(generated) = &definition.source
    else {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator resolution requires a generated definition".into(),
        ));
    };
    runmat_geometry_ops::resolve_generated_geometry_request(
        generated,
        &definition.parameters,
        bindings,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}

pub fn admit_named_geometry_generator(
    request: ResolvedGeneratedGeometryRequest,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedGeometryGenerator> {
    let executable_source_digest = compiled_binding.source_digest();
    program_revision.validate()?;
    compiled_binding.validate_admission(&callable, function.function, executable_source_digest)?;
    if callable.owner_identity.trim().is_empty()
        || callable.qualified_name != request.function
        || callable.source_digest.bytes() != request.source_digest.bytes()
        || function.convergence != FunctionConvergence::Exact
        || !function.callable.parameters_complete
        || function.callable.variadic_inputs
        || function.callable.parameters.len() != request.arguments.len()
        || !function.callable.outputs_complete
        || function.callable.variadic_outputs
        || function.callable.outputs.len() != 1
        || function.outputs.len() != 1
        || !function.callable.captures_complete
        || !function.callable.captures.is_empty()
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator differs from its exact imported function contract".into(),
        ));
    }
    validate_geometry_output(&function.outputs[0])?;
    let arguments = request
        .arguments
        .iter()
        .zip(&function.callable.parameters)
        .map(|(argument, parameter)| encode_argument(&argument.value, parameter))
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    if !function.effects.is_deterministic_value_function()
        || !function.capabilities.0.is_subset(available_capabilities)
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator has forbidden effects or unavailable capabilities".into(),
        ));
    }
    let (callable_digest, invocation_digest, generator_identity, materialization_identity) =
        geometry_generator_identities(
            &program_revision,
            &callable,
            executable_source_digest,
            &compiled_binding,
            function.function,
            &request,
            &arguments,
        )?;
    let admitted = AdmittedGeometryGenerator {
        invocation: AdmittedGeneratorInvocation {
            program_revision: program_revision.clone(),
            callable,
            executable_source_digest,
            compiled_binding,
            function: function.function,
            frozen_inputs: crate::FrozenGeneratorInputs::DirectCaptures(Vec::new()),
            generator_identity,
            required_capabilities: function.capabilities.0.clone(),
            provenance: runmat_analysis_core::GeneratorProvenance {
                project_revision: program_revision.canonical_identity(),
                callable_id: callable_digest.to_string(),
                frozen_input_digest: invocation_digest.to_string(),
            },
        },
        request,
        arguments,
        materialization_identity,
    };
    admitted.validate()?;
    Ok(admitted)
}

pub fn prepare_geometry_generator(
    admitted: AdmittedGeometryGenerator,
    executable: ExecutableUnitEnvelope,
) -> AnalysisBridgeResult<PreparedGeometryGenerator> {
    let identity = contract_digest(
        "geometry-generator-call",
        &(admitted.materialization_identity, &admitted.arguments),
    )?;
    let chunk = GeneratorChunk {
        ordinal: 0,
        first_sample: 0,
        sample_points: Vec::new(),
        arguments: admitted.arguments.clone(),
        identity,
    };
    let program = prepare_generator_program(&admitted, executable, vec![chunk], &[])?;
    Ok(PreparedGeometryGenerator { admitted, program })
}

fn validate_geometry_output(output: &ValueFact) -> AnalysisBridgeResult<()> {
    let ValueKindFact::Object(object) = &output.kind else {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator output must be a typed geometry value".into(),
        ));
    };
    if object
        .runtime_class
        .as_ref()
        .map(|name| name.display_name())
        != Some(GENERATED_GEOMETRY_RUNTIME_CLASS)
        || object.handle_semantics == Some(true)
        || output.shape.element_count() != Some(1)
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator output must be one immutable geometry.Generated scalar".into(),
        ));
    }
    Ok(())
}

fn encode_argument(
    value: &GeometryAuthoringValue,
    parameter: &ValueFact,
) -> AnalysisBridgeResult<ValuePayload> {
    let payload = match value {
        GeometryAuthoringValue::Scalar(expression) => ValuePayload::Inline(Box::new(
            InlineValue::F64Bits(literal(expression)?.to_bits()),
        )),
        GeometryAuthoringValue::Vector(values) => dense_f64(
            values
                .iter()
                .map(literal)
                .collect::<AnalysisBridgeResult<Vec<_>>>()?,
            vec![3, 1],
        ),
        GeometryAuthoringValue::Matrix(values) => dense_f64(
            values
                .iter()
                .map(literal)
                .collect::<AnalysisBridgeResult<Vec<_>>>()?,
            vec![values.len() as u64, 1],
        ),
        GeometryAuthoringValue::Text(value) => {
            ValuePayload::Inline(Box::new(InlineValue::String(value.clone().into())))
        }
        GeometryAuthoringValue::Boolean(value) => {
            ValuePayload::Inline(Box::new(InlineValue::Logical(*value)))
        }
        GeometryAuthoringValue::Parameter(_)
        | GeometryAuthoringValue::Feature(_)
        | GeometryAuthoringValue::Region(_) => {
            return Err(AnalysisBridgeError::Invalid(
                "geometry generator arguments must resolve to portable scalar, vector, matrix, text, or logical values"
                    .into(),
            ));
        }
    };
    if !argument_matches(parameter, &payload) {
        return Err(AnalysisBridgeError::Invalid(
            "geometry generator argument differs from its exact static parameter".into(),
        ));
    }
    payload.validate(ValueLimits::default())?;
    Ok(payload)
}

fn dense_f64(values: Vec<f64>, shape: Vec<u64>) -> ValuePayload {
    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::F64,
        shape,
        little_endian_data: values
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
    })))
}

fn literal(expression: &GeometryScalarExpression) -> AnalysisBridgeResult<f64> {
    match expression {
        GeometryScalarExpression::Literal { value } if value.is_finite() => Ok(*value),
        // Explicit numerical-language boundary: the resolved request retains its
        // typed declarations/identity while MATLAB receives canonical numbers.
        GeometryScalarExpression::Canonical { value } if value.canonical_value().is_finite() => {
            Ok(value.canonical_value())
        }
        _ => Err(AnalysisBridgeError::Invalid(
            "geometry generator argument was not fully resolved".into(),
        )),
    }
}

fn argument_matches(parameter: &ValueFact, payload: &ValuePayload) -> bool {
    match (&parameter.kind, payload) {
        (ValueKindFact::Unknown, _) => true,
        (ValueKindFact::Logical, ValuePayload::Inline(value)) => {
            matches!(value.as_ref(), InlineValue::Logical(_)) && scalar_shape(&parameter.shape)
        }
        (ValueKindFact::String, ValuePayload::Inline(value)) => {
            matches!(value.as_ref(), InlineValue::String(_)) && scalar_shape(&parameter.shape)
        }
        (ValueKindFact::Numeric(numeric), ValuePayload::Inline(value))
            if numeric.class == NumericClass::Double && numeric.domain == NumericDomain::Real =>
        {
            match value.as_ref() {
                InlineValue::F64Bits(_) => scalar_shape(&parameter.shape),
                InlineValue::Dense(dense) if dense.element_type == ElementType::F64 => {
                    shape_matches(&dense.shape, &parameter.shape)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn scalar_shape(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Unknown | ShapeFact::Scalar) || shape.element_count() == Some(1)
}

fn shape_matches(actual: &[u64], expected: &ShapeFact) -> bool {
    match expected {
        ShapeFact::Unknown => true,
        ShapeFact::Scalar => actual.iter().all(|extent| *extent == 1),
        ShapeFact::Ranked { rank } => actual.len() == *rank,
        ShapeFact::Shaped { dims } => {
            actual.len() == dims.len()
                && actual.iter().zip(dims).all(|(actual, expected)| {
                    !matches!(expected, runmat_types::DimensionFact::Known(value) if *actual != *value as u64)
                })
        }
    }
}

fn contract_digest<T: serde::Serialize>(
    domain: &'static str,
    value: &T,
) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(&encode_contract(
        GENERATED_ADMISSION_PREFIX,
        domain,
        value,
        CONTRACT_LIMITS,
    )?))
}

fn geometry_generator_identities(
    program_revision: &ProgramRevision,
    callable: &BundleCallable,
    executable_source_digest: Digest,
    compiled_binding: &CompiledCallableBinding,
    function: runmat_types::ProgramFunctionId,
    request: &ResolvedGeneratedGeometryRequest,
    arguments: &[ValuePayload],
) -> AnalysisBridgeResult<(Digest, Digest, GeneratorIdentity, Digest)> {
    let callable_digest = contract_digest(
        "callable",
        &(
            program_revision,
            callable,
            executable_source_digest,
            compiled_binding,
            function,
        ),
    )?;
    let capture_digest = contract_digest("captures", &Vec::<ValuePayload>::new())?;
    let invocation_digest = contract_digest("geometry-invocation", &(request, arguments))?;
    let generator_identity =
        GeneratorIdentity::new(callable_digest, capture_digest, invocation_digest)?;
    let materialization_identity =
        contract_digest("geometry-materialization", &(generator_identity, request))?;
    Ok((
        callable_digest,
        invocation_digest,
        generator_identity,
        materialization_identity,
    ))
}

fn validate_resolved_request(
    request: &ResolvedGeneratedGeometryRequest,
) -> AnalysisBridgeResult<()> {
    if request.module_path.trim().is_empty()
        || request.function.trim().is_empty()
        || request.source_digest.bytes().iter().all(|byte| *byte == 0)
    {
        return Err(AnalysisBridgeError::Invalid(
            "generated geometry request is malformed".into(),
        ));
    }
    let mut parameters = BTreeSet::new();
    for argument in &request.arguments {
        if argument.parameter.trim().is_empty() || !parameters.insert(argument.parameter.as_str()) {
            return Err(AnalysisBridgeError::Invalid(
                "generated geometry request parameters must be named uniquely".into(),
            ));
        }
        match &argument.value {
            GeometryAuthoringValue::Scalar(value) => {
                literal(value)?;
            }
            GeometryAuthoringValue::Vector(values) => {
                for value in values.iter() {
                    literal(value)?;
                }
            }
            GeometryAuthoringValue::Matrix(values) => {
                for value in values {
                    literal(value)?;
                }
            }
            GeometryAuthoringValue::Text(_) | GeometryAuthoringValue::Boolean(_) => {}
            GeometryAuthoringValue::Parameter(_)
            | GeometryAuthoringValue::Feature(_)
            | GeometryAuthoringValue::Region(_) => {
                return Err(AnalysisBridgeError::Invalid(
                    "generated geometry request contains unresolved values".into(),
                ));
            }
        }
    }
    Ok(())
}
