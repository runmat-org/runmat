//! Admission and value boundaries for portable, pure engineering assessment functions.
use crate::named_callable::{
    prepare_named_callable_descriptor, AnalysisCallablePurpose, NamedCallableAdmissionRef,
    NamedCallableSourceRef,
};
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_analysis_core::{AssessmentFunctionOutcome, AssessmentRule, AuthoredAssessment};
use runmat_execution::value::{ElementType, InlineValue, ValueLimits, ValuePayload};
use runmat_execution::{ExecutableUnitEnvelope, ProgramRevision};
use runmat_execution_artifact::{BundleCallable, ProgramExecutionDescriptor};
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{EffectKind, NumericDomain, ValueFact, ValueKindFact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Portable callable frozen before a solve. Restoring it never reads the live project.
/// Admission is repeated against the exact criterion and authenticated executable on use.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenAssessmentCallable {
    assessment: AuthoredAssessment,
    revision: ProgramRevision,
    callable: BundleCallable,
    executable: ExecutableUnitEnvelope,
    compiled_binding: crate::named_callable::CompiledCallableBinding,
}

impl FrozenAssessmentCallable {
    pub fn new(
        assessment: AuthoredAssessment,
        revision: ProgramRevision,
        callable: BundleCallable,
        executable: ExecutableUnitEnvelope,
    ) -> AnalysisBridgeResult<Self> {
        executable.validate()?;
        let compiled_binding =
            crate::named_callable::CompiledCallableBinding::from_selected_identity(
                &executable.manifest.identity,
                &callable.qualified_name,
            )?;
        prepare_assessment_descriptor(
            &assessment,
            NamedCallableSourceRef {
                program_revision: &revision,
                callable: &callable,
                compiled_binding: &compiled_binding,
            },
            executable.clone(),
        )?;
        Ok(Self {
            assessment,
            revision,
            callable,
            executable,
            compiled_binding,
        })
    }

    pub fn assessment(&self) -> &AuthoredAssessment {
        &self.assessment
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        prepare_assessment_descriptor(
            &self.assessment,
            NamedCallableSourceRef {
                program_revision: &self.revision,
                callable: &self.callable,
                compiled_binding: &self.compiled_binding,
            },
            self.executable.clone(),
        )?;
        Ok(())
    }

    pub fn prepare(
        &self,
        assessment: &AuthoredAssessment,
        measured_si: f64,
        study_definition: runmat_execution::Digest,
        evidence_identity: runmat_execution::Digest,
    ) -> AnalysisBridgeResult<PreparedAssessmentProgram> {
        if assessment != &self.assessment {
            return Err(AnalysisBridgeError::Identity(
                "frozen assessment callable belongs to another criterion",
            ));
        }
        prepare_assessment_program(
            assessment,
            measured_si,
            study_definition,
            evidence_identity,
            NamedCallableSourceRef {
                program_revision: &self.revision,
                callable: &self.callable,
                compiled_binding: &self.compiled_binding,
            },
            self.executable.clone(),
        )
    }
}

/// A prepared assessment is tied to both immutable numerical evidence and the exact criterion.
/// Private fields prevent replacing the request after admission while retaining its identity.
pub struct PreparedAssessmentProgram {
    request: runmat_execution_artifact::ProgramExecutionRequest,
    callable: BundleCallable,
    identity: runmat_execution::Digest,
    binding: AssessmentProgramBinding,
}

#[derive(Clone, Debug, PartialEq)]
struct AssessmentProgramBinding {
    assessment: AuthoredAssessment,
    measured_si_bits: u64,
    study_definition: runmat_execution::Digest,
    evidence_identity: runmat_execution::Digest,
}

/// A host-admitted result cannot be detached from its immutable inputs or criterion.
pub struct AdmittedAssessmentExecution {
    binding: AssessmentProgramBinding,
    evaluation: runmat_analysis_core::AssessmentFunctionEvaluation,
    program_identity: runmat_execution::Digest,
    diagnostic: Option<String>,
}
impl AdmittedAssessmentExecution {
    pub fn validate_binding(
        &self,
        assessment: &AuthoredAssessment,
        measured_si: f64,
        study_definition: runmat_execution::Digest,
        evidence_identity: runmat_execution::Digest,
    ) -> AnalysisBridgeResult<runmat_analysis_core::AssessmentFunctionEvaluation> {
        if self.binding
            != (AssessmentProgramBinding {
                assessment: assessment.clone(),
                measured_si_bits: measured_si.to_bits(),
                study_definition,
                evidence_identity,
            })
        {
            return Err(AnalysisBridgeError::Identity(
                "assessment execution belongs to other inputs",
            ));
        }
        Ok(self.evaluation)
    }
    pub fn program_identity(&self) -> runmat_execution::Digest {
        self.program_identity
    }
    pub fn assessment_id(&self) -> &str {
        &self.binding.assessment.assessment_id
    }
    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
    }
}
impl PreparedAssessmentProgram {
    pub fn request(&self) -> &runmat_execution_artifact::ProgramExecutionRequest {
        &self.request
    }
    pub fn identity(&self) -> runmat_execution::Digest {
        self.identity
    }

    pub fn admit_response(
        &self,
        response: &runmat_execution_artifact::ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<AdmittedAssessmentExecution> {
        use runmat_analysis_core::AssessmentFunctionEvaluation;
        use runmat_execution_artifact::ProgramExecutionResponse;
        response.validate_against(&self.request)?;
        let (evaluation, diagnostic) = match response {
            ProgramExecutionResponse::Success { outputs } => match outputs.as_slice() {
                [value] => match decode_assessment_outcome(value) {
                    Ok(outcome) => (AssessmentFunctionEvaluation::Completed(outcome), None),
                    Err(error) => (AssessmentFunctionEvaluation::Failed, Some(error.to_string())),
                },
                _ => (
                    AssessmentFunctionEvaluation::Failed,
                    Some("assessment must return exactly one logical record".into()),
                ),
            },
            ProgramExecutionResponse::Failure { message } => {
                (AssessmentFunctionEvaluation::Failed, Some(message.clone()))
            }
            ProgramExecutionResponse::RuntimeFailure { failure } => (
                AssessmentFunctionEvaluation::Failed,
                Some(failure.message.clone()),
            ),
            ProgramExecutionResponse::GraphSuccess { .. }
            | ProgramExecutionResponse::SpmdSuccess { .. } => (
                AssessmentFunctionEvaluation::Failed,
                Some("assessment returned a graph or SPMD result instead of one logical record".into()),
            ),
            ProgramExecutionResponse::ExternalizedSuccess { .. } => (
                AssessmentFunctionEvaluation::Failed,
                Some("assessment returned an object-backed value instead of its bounded logical record".into()),
            ),
        };
        Ok(AdmittedAssessmentExecution {
            binding: self.binding.clone(),
            evaluation,
            program_identity: self.identity,
            diagnostic: diagnostic.map(|message| message.chars().take(1024).collect()),
        })
    }
}

pub fn prepare_assessment_program(
    assessment: &AuthoredAssessment,
    measured_si: f64,
    study_definition: runmat_execution::Digest,
    evidence_identity: runmat_execution::Digest,
    source: NamedCallableSourceRef<'_>,
    executable: ExecutableUnitEnvelope,
) -> AnalysisBridgeResult<PreparedAssessmentProgram> {
    let NamedCallableSourceRef {
        program_revision: revision,
        callable,
        compiled_binding,
    } = source;
    use runmat_execution_artifact::{
        ProgramExecutionInputs, ProgramExecutionRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };
    let descriptor = prepare_assessment_descriptor(
        assessment,
        NamedCallableSourceRef {
            program_revision: revision,
            callable,
            compiled_binding,
        },
        executable,
    )?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        context: runmat_execution::ProgramInvocationContext::Direct,
        arguments: assessment_inputs(assessment, measured_si)?,
        objects: Vec::new(),
    };
    let request = ProgramExecutionRequest::from_parts(descriptor, inputs)?;
    request.validate_for_portable_host()?;
    let encoded = runmat_canonical_codec::encode_contract(
        b"runmat-analysis-assessment/v1\0",
        "assessment-program",
        &(
            assessment,
            measured_si.to_bits(),
            study_definition,
            evidence_identity,
            &request.recipe,
            request.artifact.id,
        ),
        runmat_canonical_codec::CanonicalLimits::new(1024 * 1024, 8192, 4096, 32),
    )?;
    let identity = runmat_execution::Digest::sha256(encoded);
    Ok(PreparedAssessmentProgram {
        request,
        callable: callable.clone(),
        identity,
        binding: AssessmentProgramBinding {
            assessment: assessment.clone(),
            measured_si_bits: measured_si.to_bits(),
            study_definition,
            evidence_identity,
        },
    })
}

pub fn build_assessment_task_submission(
    prepared: &PreparedAssessmentProgram,
    dependencies: BTreeSet<runmat_execution::TaskId>,
    context: crate::named_callable::NamedCallableExecutionContext,
) -> AnalysisBridgeResult<runmat_execution_runner::TaskSubmission> {
    use runmat_execution::task::TaskRequest;
    use runmat_execution::{OutputContract, TaskId};
    context.resources.validate()?;
    prepared.request.validate_for_portable_host()?;
    crate::named_callable::validate_program_host(&prepared.request, &context.host)?;
    if !context
        .resources
        .required_capabilities
        .contains(&runmat_execution::resource::Capability::ProcessIsolation)
    {
        return Err(AnalysisBridgeError::Invalid(
            "assessment tasks require bounded process isolation".into(),
        ));
    }
    Ok(runmat_execution_runner::TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: TaskId::derive(&[prepared.identity.bytes()]),
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                runmat_execution::identity::ArtifactId::derive(&[prepared
                    .request
                    .artifact
                    .id
                    .0
                    .bytes()]),
            ),
            callable: crate::named_callable::program_task_callable(
                &prepared.callable.owner_identity,
                &prepared.request,
            ),
            host: context.host,
            invocation_context: prepared.request.context.clone(),
            inputs: prepared.request.arguments.clone(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: context.resources,
            retry: context.retry,
            deadline_unix_millis: context.deadline_unix_millis,
        },
        dependencies,
        priority: context.priority,
    })
}

/// The caller resolves the callable from the checked study's frozen module graph. No live
/// workspace closure or unknown output contract can cross this boundary.
pub fn prepare_assessment_descriptor(
    assessment: &AuthoredAssessment,
    source: NamedCallableSourceRef<'_>,
    executable: ExecutableUnitEnvelope,
) -> AnalysisBridgeResult<ProgramExecutionDescriptor> {
    let NamedCallableSourceRef {
        program_revision: revision,
        callable,
        compiled_binding,
    } = source;
    let AssessmentRule::Function {
        function,
        parameters,
    } = &assessment.rule
    else {
        return Err(AnalysisBridgeError::Invalid(
            "assessment does not name a function".into(),
        ));
    };
    if callable.source_digest.bytes() != &function.source_digest
        || callable.qualified_name != function.symbol
        || function.module_path.trim().is_empty()
        || parameters
            .iter()
            .any(|parameter| !parameter.canonical_value().is_finite())
    {
        return Err(AnalysisBridgeError::Identity(
            "assessment callable differs from its checked source",
        ));
    }
    // Authenticate against the original frozen selection before recovering
    // static facts from a later executable candidate.
    executable.validate()?;
    compiled_binding.validate_candidate(&executable.manifest.identity, revision, callable)?;
    let facts =
        crate::callable::executable::verified_named_function(&executable, revision, callable)?;
    validate_assessment_function(
        &facts,
        parameters.len().checked_add(1).ok_or_else(|| {
            AnalysisBridgeError::Invalid("assessment input count overflows".into())
        })?,
    )?;
    let executable_source_digest = compiled_binding.source_digest();
    prepare_named_callable_descriptor(
        NamedCallableAdmissionRef {
            program_revision: revision,
            callable,
            compiled_binding,
            executable_source_digest,
            function: compiled_binding.function(),
            required_capabilities: &facts.capabilities.0,
        },
        executable,
        AnalysisCallablePurpose::Assessment,
    )
}

fn validate_assessment_function(
    function: &FunctionAnalysis,
    input_count: usize,
) -> AnalysisBridgeResult<()> {
    let callable = &function.callable;
    if function.convergence != FunctionConvergence::Exact
        || !callable.parameters_complete
        || callable.variadic_inputs
        || callable.parameters.len() != input_count
        || !callable.outputs_complete
        || callable.variadic_outputs
        || callable.outputs.len() != 1
        || function.outputs.len() != 1
        || !callable.captures_complete
        || !callable.captures.is_empty()
        || !callable.parameters.iter().all(real_scalar)
        || !outcome_fact(&callable.outputs[0])
        || !outcome_fact(&function.outputs[0])
        || !function
            .effects
            .0
            .is_subset(&BTreeSet::from([EffectKind::MayThrow]))
        || !function.capabilities.0.is_empty()
    {
        return Err(AnalysisBridgeError::Invalid(
            "assessment requires exact real scalar inputs, a complete {evaluated, accepted} logical record, and no captures or ambient capabilities".into()));
    }
    Ok(())
}
fn real_scalar(fact: &ValueFact) -> bool {
    fact.is_scalar()
        && matches!(&fact.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real)
}
fn outcome_fact(fact: &ValueFact) -> bool {
    let ValueKindFact::Struct(record) = &fact.kind else {
        return false;
    };
    fact.is_scalar()
        && record.fields_complete
        && record.fields.len() == 2
        && ["evaluated", "accepted"].iter().all(|name| {
            record
                .fields
                .get(*name)
                .is_some_and(|field| field.is_scalar() && field.kind == ValueKindFact::Logical)
        })
}

pub fn assessment_inputs(
    assessment: &AuthoredAssessment,
    measured_si: f64,
) -> AnalysisBridgeResult<Vec<ValuePayload>> {
    let AssessmentRule::Function { parameters, .. } = &assessment.rule else {
        return Err(AnalysisBridgeError::Invalid(
            "assessment does not name a function".into(),
        ));
    };
    std::iter::once(measured_si)
        .chain(
            parameters
                .iter()
                .map(|parameter| parameter.canonical_value()),
        )
        .map(|value| {
            if value.is_finite() {
                Ok(ValuePayload::Inline(Box::new(InlineValue::F64Bits(
                    value.to_bits(),
                ))))
            } else {
                Err(AnalysisBridgeError::Invalid(
                    "assessment input must be finite".into(),
                ))
            }
        })
        .collect()
}

/// Decode a bounded scalar record. Missing, extra, duplicate or nonlogical fields are errors;
/// they must never be interpreted as a failing criterion or a successful assessment.
pub fn decode_assessment_outcome(
    payload: &ValuePayload,
) -> AnalysisBridgeResult<AssessmentFunctionOutcome> {
    payload.validate(ValueLimits {
        max_depth: 3,
        max_nodes: 3,
        max_inline_bytes: 1024,
        max_elements: 3,
        max_fields: 2,
        max_text_bytes: 64,
    })?;
    let ValuePayload::Inline(value) = payload else {
        return Err(AnalysisBridgeError::Invalid(
            "assessment outcome must be an inline scalar record".into(),
        ));
    };
    let InlineValue::Struct(fields) = value.as_ref() else {
        return Err(AnalysisBridgeError::Invalid(
            "assessment outcome must be a scalar record".into(),
        ));
    };
    let logical = |name: &str| -> Option<bool> {
        let mut matching = fields.iter().filter(|field| field.name == name);
        let field = matching.next()?;
        if matching.next().is_some() {
            return None;
        }
        match &field.value {
            ValuePayload::Inline(value) => match value.as_ref() {
                InlineValue::Logical(value) => Some(*value),
                InlineValue::Dense(value)
                    if value.element_type == ElementType::Logical
                        && value.shape.iter().all(|extent| *extent == 1) =>
                {
                    match value.little_endian_data.as_slice() {
                        [0] => Some(false),
                        [1] => Some(true),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    };
    match (fields.len(), logical("evaluated"), logical("accepted")) {
        (2, Some(evaluated), Some(accepted)) => Ok(AssessmentFunctionOutcome {
            evaluated,
            accepted,
        }),
        _ => Err(AnalysisBridgeError::Invalid(
            "assessment outcome requires exactly evaluated and accepted logical fields".into(),
        )),
    }
}

#[cfg(test)]
mod tests;
