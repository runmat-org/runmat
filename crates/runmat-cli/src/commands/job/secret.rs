use std::path::PathBuf;

use anyhow::{Context, Result};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

pub const SAVED_REMOTE_RUN_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMaterialization {
    Pending,
    Materialized,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum SavedOutputIntent {
    Generic,
    FeaStudy {
        study_id: String,
        expectation: Box<runmat_execution_runner_native::StructuralStudyResultExpectation>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        assessment_functions:
            Vec<runmat_analysis_execution::assessment_program::FrozenAssessmentCallable>,
    },
    FeaSweep {
        sweep_id: String,
        case_ids: Vec<String>,
        expectation: Box<runmat_execution_runner_native::StructuralSweepResultExpectation>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        assessment_functions:
            Vec<Vec<runmat_analysis_execution::assessment_program::FrozenAssessmentCallable>>,
    },
    Mesh {
        solver_mesh_path: PathBuf,
        evidence_path: PathBuf,
        force: bool,
        materialization: OutputMaterialization,
    },
    StructuralStudy {
        continuation: String,
        input_materializations: Vec<runmat_analysis_execution::InputMaterializationExpectation>,
        revision: runmat_execution::ProgramRevision,
        materialization: OutputMaterialization,
        result: Option<
            Box<
                runmat_runtime::operations::OperationEnvelope<
                    runmat_runtime::analysis::AnalysisStudyRunData,
                >,
            >,
        >,
    },
    StructuralSweep {
        spec: Box<runmat_runtime::analysis::AnalysisStudySweepSpec>,
        continuations: Vec<String>,
        input_materializations:
            Vec<Vec<runmat_analysis_execution::InputMaterializationExpectation>>,
        revision: runmat_execution::ProgramRevision,
        materialization: OutputMaterialization,
        result: Option<
            Box<
                runmat_runtime::operations::OperationEnvelope<
                    runmat_runtime::analysis::AnalysisStudySweepData,
                >,
            >,
        >,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SavedRemoteRun {
    pub schema_version: u16,
    pub run_id: String,
    pub server_url: String,
    pub project_id: String,
    pub endpoint_fingerprint: String,
    pub run_key: String,
    pub output_intent: SavedOutputIntent,
    #[serde(deserialize_with = "runmat_execution::console::deserialize_controls")]
    pub console: Option<runmat_execution::console::ConsoleLimits>,
    #[serde(deserialize_with = "deserialize_console_budget")]
    pub console_budget: Option<runmat_execution_artifact::ProgramExecutionConsoleBudget>,
}

impl SavedRemoteRun {
    pub fn new(
        run_id: String,
        server_url: String,
        project_id: String,
        endpoint_fingerprint: String,
        run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
        output_intent: SavedOutputIntent,
    ) -> Self {
        Self {
            schema_version: SAVED_REMOTE_RUN_SCHEMA_VERSION,
            run_id,
            server_url,
            project_id,
            endpoint_fingerprint,
            run_key: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(run_key.expose_for_recipient_envelope()),
            output_intent,
            console: None,
            console_budget: None,
        }
    }

    fn validate(&self, expected_run_id: &str) -> Result<()> {
        if self.schema_version != SAVED_REMOTE_RUN_SCHEMA_VERSION {
            anyhow::bail!("saved remote run uses an unsupported schema");
        }
        if self.run_id != expected_run_id {
            anyhow::bail!("saved remote run identity does not match its storage key");
        }
        if let Some(limits) = self.console {
            limits.validate()?;
        }
        if let Some(budget) = self.console_budget {
            budget.validate()?;
        }
        anyhow::ensure!(
            self.console.is_some() == self.console_budget.is_some(),
            "saved remote console controls are incomplete"
        );
        match &self.output_intent {
            SavedOutputIntent::FeaStudy {
                study_id,
                expectation,
                assessment_functions,
            } => {
                if study_id.trim().is_empty() {
                    anyhow::bail!("saved FEA study identity is empty");
                }
                expectation.validate()?;
                validate_frozen_assessments(&expectation.study.assessments, assessment_functions)?;
            }
            SavedOutputIntent::FeaSweep {
                sweep_id,
                case_ids,
                expectation,
                assessment_functions,
            } => {
                if sweep_id.trim().is_empty()
                    || case_ids.len() != expectation.studies.len()
                    || case_ids.iter().any(|case| case.trim().is_empty())
                    || case_ids
                        .iter()
                        .enumerate()
                        .any(|(index, case)| case_ids[..index].contains(case))
                {
                    anyhow::bail!("saved FEA sweep identity or case order is invalid");
                }
                expectation.validate()?;
                if assessment_functions.is_empty() {
                    for study in &expectation.studies {
                        validate_frozen_assessments(&study.study.assessments, &[])?;
                    }
                } else {
                    if assessment_functions.len() != expectation.studies.len() {
                        anyhow::bail!(
                            "saved FEA sweep callable inventory differs from its case order"
                        );
                    }
                    for (study, functions) in expectation.studies.iter().zip(assessment_functions) {
                        validate_frozen_assessments(&study.study.assessments, functions)?;
                    }
                }
            }
            _ => {}
        }
        if let SavedOutputIntent::Mesh {
            solver_mesh_path,
            evidence_path,
            ..
        } = &self.output_intent
        {
            if !solver_mesh_path.is_absolute() || !evidence_path.is_absolute() {
                anyhow::bail!("saved mesh output destinations must be absolute paths");
            }
            if solver_mesh_path == evidence_path {
                anyhow::bail!("saved mesh output destinations must be distinct");
            }
        }
        if let SavedOutputIntent::StructuralStudy {
            continuation,
            input_materializations,
            revision,
            materialization,
            result,
            ..
        } = &self.output_intent
        {
            let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(continuation)
                .context("saved structural study continuation is malformed")?;
            let intent =
                runmat_runtime::analysis::PreparedStructuralStudyIntent::from_continuation_bytes(
                    &bytes,
                )
                .map_err(anyhow::Error::msg)
                .context("saved structural study continuation is invalid")?;
            validate_input_expectations(&intent, input_materializations, revision)?;
            if (*materialization == OutputMaterialization::Materialized) != result.is_some() {
                anyhow::bail!("saved structural study materialization state is inconsistent");
            }
        }
        if let SavedOutputIntent::StructuralSweep {
            spec,
            continuations,
            input_materializations,
            revision,
            materialization,
            result,
            ..
        } = &self.output_intent
        {
            let expected_identity = runmat_runtime::analysis::analysis_study_sweep_identity(
                spec,
                &runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|error| anyhow::anyhow!(error.message))?;
            if continuations.len() != spec.studies.len()
                || input_materializations.len() != spec.studies.len()
            {
                anyhow::bail!("saved structural sweep continuation count is invalid");
            }
            for ((continuation, authored), signals) in continuations
                .iter()
                .zip(&spec.studies)
                .zip(input_materializations)
            {
                let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(continuation)
                    .context("saved structural sweep continuation is malformed")?;
                let intent = runmat_runtime::analysis::PreparedStructuralStudyIntent::from_continuation_bytes(&bytes)
                    .map_err(anyhow::Error::msg)
                    .context("saved structural sweep continuation is invalid")?;
                if intent.study_id() != authored.study_id
                    || intent.authored_study_digest()
                        != runmat_runtime::analysis::analysis_study_content_digest(authored)
                {
                    anyhow::bail!(
                        "saved structural sweep continuation differs from its authored study"
                    );
                }
                validate_input_expectations(&intent, signals, revision)?;
            }
            if (*materialization == OutputMaterialization::Materialized) != result.is_some() {
                anyhow::bail!("saved structural sweep materialization state is inconsistent");
            }
            if let Some(result) = result {
                let data = &result.data;
                if data.sweep_id != spec.sweep_id
                    || data.identity != expected_identity
                    || data.study_count != spec.studies.len()
                    || data.success_count != data.run_entries.len()
                    || data.failed_count != data.failure_entries.len()
                    || data.success_count + data.failed_count != data.study_count
                {
                    anyhow::bail!("saved structural sweep result differs from its authored sweep");
                }
                let mut represented = std::collections::BTreeSet::new();
                for entry in &data.run_entries {
                    if !spec
                        .studies
                        .iter()
                        .any(|study| study.study_id == entry.study_id)
                        || !represented.insert(entry.study_id.as_str())
                    {
                        anyhow::bail!("saved structural sweep result contains a substituted run");
                    }
                }
                for entry in &data.failure_entries {
                    if spec
                        .studies
                        .get(entry.study_index)
                        .is_none_or(|study| study.study_id != entry.study_id)
                        || !represented.insert(entry.study_id.as_str())
                    {
                        anyhow::bail!(
                            "saved structural sweep result contains a substituted failure"
                        );
                    }
                }
                if represented.len() != spec.studies.len() {
                    anyhow::bail!("saved structural sweep result omits an authored study");
                }
            }
        }
        Ok(())
    }

    pub fn key(&self) -> Result<runmat_execution_artifact::encryption::RunKeyMaterial> {
        self.validate(&self.run_id)?;
        let bytes: [u8; 32] = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(&self.run_key)
            .context("saved run secret is malformed")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("saved run secret has an invalid length"))?;
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy(bytes)
            .map_err(anyhow::Error::from)
    }
}

fn deserialize_console_budget<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<runmat_execution_artifact::ProgramExecutionConsoleBudget>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let budget: Option<runmat_execution_artifact::ProgramExecutionConsoleBudget> =
        Deserialize::deserialize(deserializer)?;
    if let Some(budget) = budget {
        budget.validate().map_err(serde::de::Error::custom)?;
    }
    Ok(budget)
}

fn validate_frozen_assessments(
    assessments: &[runmat_analysis_core::AuthoredAssessment],
    functions: &[runmat_analysis_execution::assessment_program::FrozenAssessmentCallable],
) -> Result<()> {
    let expected: Vec<_> = assessments
        .iter()
        .filter(|assessment| {
            matches!(
                assessment.rule,
                runmat_analysis_core::AssessmentRule::Function { .. }
            )
        })
        .collect();
    if functions.len() != expected.len()
        || functions
            .iter()
            .zip(expected)
            .any(|(function, assessment)| function.assessment() != assessment)
    {
        anyhow::bail!("saved FEA function inventory differs from its checked criteria");
    }
    for function in functions {
        function.validate().map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

fn validate_input_expectations(
    intent: &runmat_runtime::analysis::PreparedStructuralStudyIntent,
    expectations: &[runmat_analysis_execution::InputMaterializationExpectation],
    revision: &runmat_execution::ProgramRevision,
) -> Result<()> {
    let model = intent.model();
    let authored_signals = model
        .signals
        .iter()
        .filter(|signal| signal.generator().is_some())
        .collect::<Vec<_>>();
    let authored_responses = model
        .response_spectra
        .iter()
        .filter(|spectrum| spectrum.ordinates.generator().is_some())
        .collect::<Vec<_>>();
    let authored_random = model
        .random_excitations
        .iter()
        .filter(|spectrum| spectrum.spectral_density.generator().is_some())
        .collect::<Vec<_>>();
    if authored_signals.len() + authored_responses.len() + authored_random.len()
        != expectations.len()
    {
        anyhow::bail!("saved input expectations differ from the authored study");
    }
    for (index, expectation) in expectations.iter().enumerate() {
        expectation.validate().map_err(anyhow::Error::msg)?;
        let authored_matches = match expectation {
            runmat_analysis_execution::InputMaterializationExpectation::Signal(value) => {
                authored_signals
                    .get(index)
                    .is_some_and(|signal| *signal == &value.authored)
            }
            runmat_analysis_execution::InputMaterializationExpectation::Numerical(value) => {
                let numerical_index = index.saturating_sub(authored_signals.len());
                match &value.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        spectrum,
                    ) => authored_responses
                        .get(numerical_index)
                        .is_some_and(|authored| *authored == spectrum.as_ref()),
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                        spectrum,
                    ) => {
                        let random_index = numerical_index.saturating_sub(authored_responses.len());
                        authored_random
                            .get(random_index)
                            .is_some_and(|authored| *authored == spectrum.as_ref())
                    }
                }
            }
        };
        if !authored_matches {
            anyhow::bail!("saved input expectations differ from the authored study");
        }
        let admitted_revision = match expectation {
            runmat_analysis_execution::InputMaterializationExpectation::Signal(value) => {
                &value.admitted.program_revision
            }
            runmat_analysis_execution::InputMaterializationExpectation::Numerical(value) => {
                &value.admitted.invocation.program_revision
            }
        };
        if admitted_revision != revision {
            anyhow::bail!("saved input expectation differs from the remote program revision");
        }
    }
    Ok(())
}

pub fn mark_output_materialized(run_id: &str) -> Result<()> {
    let mut saved = load(run_id)?;
    match &mut saved.output_intent {
        SavedOutputIntent::Mesh {
            materialization, ..
        } => *materialization = OutputMaterialization::Materialized,
        SavedOutputIntent::StructuralStudy { .. }
        | SavedOutputIntent::StructuralSweep { .. }
        | SavedOutputIntent::FeaStudy { .. }
        | SavedOutputIntent::FeaSweep { .. }
        | SavedOutputIntent::Generic => {
            anyhow::bail!("remote run has no deferred output intent")
        }
    }
    save(&saved)
}

pub fn mark_structural_sweep_materialized(
    run_id: &str,
    result: runmat_runtime::operations::OperationEnvelope<
        runmat_runtime::analysis::AnalysisStudySweepData,
    >,
) -> Result<()> {
    let mut saved = load(run_id)?;
    let SavedOutputIntent::StructuralSweep {
        materialization,
        result: saved_result,
        ..
    } = &mut saved.output_intent
    else {
        anyhow::bail!("remote run has no deferred structural sweep output intent");
    };
    *materialization = OutputMaterialization::Materialized;
    *saved_result = Some(Box::new(result));
    save(&saved)
}

pub fn mark_structural_study_materialized(
    run_id: &str,
    result: runmat_runtime::operations::OperationEnvelope<
        runmat_runtime::analysis::AnalysisStudyRunData,
    >,
) -> Result<()> {
    let mut saved = load(run_id)?;
    let SavedOutputIntent::StructuralStudy {
        materialization,
        result: saved_result,
        ..
    } = &mut saved.output_intent
    else {
        anyhow::bail!("remote run has no deferred structural study output intent");
    };
    *materialization = OutputMaterialization::Materialized;
    *saved_result = Some(Box::new(result));
    save(&saved)
}

pub fn save(value: &SavedRemoteRun) -> Result<()> {
    value.validate(&value.run_id)?;
    let path = path(&value.run_id)?;
    let parent = path.parent().context("saved run path has no parent")?;
    std::fs::create_dir_all(parent).context("create saved run directory")?;
    set_private_directory(parent)?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("create staged saved remote run secret")?;
    set_private_file(temporary.path())?;
    use std::io::Write as _;
    temporary
        .write_all(&serde_json::to_vec_pretty(value)?)
        .context("write staged saved remote run secret")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync staged saved remote run secret")?;
    temporary
        .persist(&path)
        .map_err(|error| error.error)
        .context("publish saved remote run secret")?;
    Ok(())
}

pub fn load(run_id: &str) -> Result<SavedRemoteRun> {
    let path = path(run_id)?;
    let saved: SavedRemoteRun = serde_json::from_slice(
        &std::fs::read(&path)
            .with_context(|| format!("no local attach secret is available for run {run_id}"))?,
    )
    .context("read saved remote run secret")?;
    saved.validate(run_id)?;
    Ok(saved)
}

fn path(run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || run_id.len() > 256
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        anyhow::bail!("remote run id is malformed");
    }
    let root = if let Some(path) = std::env::var_os("RUNMAT_CLI_CONFIG_DIR") {
        PathBuf::from(path)
    } else {
        dirs::config_dir()
            .context("unable to locate the user configuration directory")?
            .join("runmat")
    };
    Ok(root.join("execution-runs").join(format!("{run_id}.json")))
}

#[cfg(unix)]
fn set_private_directory(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file(_path: &std::path::Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revision() -> runmat_execution::ProgramRevision {
        runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::sha256(b"saved-run-graph"),
            runmat_execution::Digest::sha256(b"saved-run-source"),
            runmat_execution::ProgramEnvironment::new(
                1,
                1,
                runmat_execution::Digest::sha256(b"saved-run-runtime"),
                runmat_execution::Digest::sha256(b"saved-run-catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn attach_secret_round_trips_without_exposing_it_in_job_identity() {
        let temp = tempfile::tempdir().unwrap();
        std::env::set_var("RUNMAT_CLI_CONFIG_DIR", temp.path());
        let key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([7; 32]).unwrap();
        let saved = SavedRemoteRun::new(
            "run_123".into(),
            "https://example.invalid".into(),
            "project_1".into(),
            "fingerprint".into(),
            &key,
            SavedOutputIntent::Mesh {
                solver_mesh_path: temp.path().join("mesh.cbor"),
                evidence_path: temp.path().join("evidence.cbor"),
                force: false,
                materialization: OutputMaterialization::Pending,
            },
        );
        save(&saved).unwrap();
        let loaded = load("run_123").unwrap();
        assert_eq!(
            loaded.key().unwrap().expose_for_recipient_envelope(),
            key.expose_for_recipient_envelope()
        );
        assert_eq!(loaded.output_intent, saved.output_intent);
        mark_output_materialized("run_123").unwrap();
        assert!(matches!(
            load("run_123").unwrap().output_intent,
            SavedOutputIntent::Mesh {
                materialization: OutputMaterialization::Materialized,
                ..
            }
        ));
        std::env::remove_var("RUNMAT_CLI_CONFIG_DIR");
    }

    #[test]
    fn saved_mesh_intent_rejects_ambiguous_or_relocated_destinations() {
        let key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([9; 32]).unwrap();
        let relative = SavedRemoteRun::new(
            "run_relative".into(),
            "https://example.invalid".into(),
            "project_1".into(),
            "fingerprint".into(),
            &key,
            SavedOutputIntent::Mesh {
                solver_mesh_path: "mesh.cbor".into(),
                evidence_path: "evidence.cbor".into(),
                force: false,
                materialization: OutputMaterialization::Pending,
            },
        );
        assert!(relative.validate("run_relative").is_err());

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("result.cbor");
        let ambiguous = SavedRemoteRun::new(
            "run_ambiguous".into(),
            "https://example.invalid".into(),
            "project_1".into(),
            "fingerprint".into(),
            &key,
            SavedOutputIntent::Mesh {
                solver_mesh_path: path.clone(),
                evidence_path: path,
                force: false,
                materialization: OutputMaterialization::Pending,
            },
        );
        assert!(ambiguous.validate("run_ambiguous").is_err());
        assert!(ambiguous.validate("different_run").is_err());
    }

    #[test]
    fn malformed_structural_study_continuation_is_rejected() {
        let key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([11; 32]).unwrap();
        let continuation = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"continuation");
        let saved = SavedRemoteRun::new(
            "run_structural".into(),
            "https://example.invalid".into(),
            "project_1".into(),
            "fingerprint".into(),
            &key,
            SavedOutputIntent::StructuralStudy {
                continuation,
                input_materializations: Vec::new(),
                revision: revision(),
                materialization: OutputMaterialization::Pending,
                result: None,
            },
        );
        assert!(saved.validate("run_structural").is_err());
    }
}
