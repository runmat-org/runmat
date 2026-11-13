use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use runmat_aot::selected_admission::SelectedHostAdmission;
use runmat_execution::host::{
    NativeClockInputs, NativeDeploymentProfile, NativeDeploymentServiceKind, NativeFilesystemEntry,
};
use runmat_types::EffectKind;

const MAX_WITNESS_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum WitnessMode {
    Pure,
    Clock {
        inputs: Option<NativeClockInputs>,
    },
    DeterministicLocal {
        environment_key: String,
        environment_value: String,
        environment_write_key: String,
        environment_write_value: String,
        language_seed: u64,
        read_path: String,
        read_utf8: String,
        write_path: String,
        write_utf8: String,
    },
    HostedExternal {
        console_line: String,
        http_port: u16,
        tcp_port: u16,
    },
    HostedCallback {
        artifact_bundle: PathBuf,
        expected_bundle_digest: String,
        foreign_policy: PathBuf,
        expected_foreign_policy_digest: String,
        interface_name: String,
        artifact_identity: String,
        library_digest: String,
        input: i32,
        offset: i32,
        expected: i32,
    },
}

impl WitnessMode {
    pub fn load(path: Option<&std::path::Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::Pure);
        };
        let bytes = runmat_aot::selected_admission::read_bounded(path, MAX_WITNESS_BYTES as u64)
            .with_context(|| format!("read bounded witness mode `{}`", path.display()))?;
        if let Ok(mode) = serde_json::from_slice::<Self>(&bytes) {
            return Ok(mode);
        }
        // Preserve the earlier fifth-argument Clock input form while keeping
        // all new witnesses explicitly tagged and closed.
        let inputs: Option<NativeClockInputs> =
            serde_json::from_slice(&bytes).context("invalid typed native paired witness JSON")?;
        Ok(Self::Clock { inputs })
    }

    pub fn clock_inputs(&self) -> Option<NativeClockInputs> {
        match self {
            Self::Clock { inputs } => inputs.clone(),
            _ => None,
        }
    }

    pub fn validate(&self, host: &SelectedHostAdmission) -> Result<()> {
        let expected = self.expected_effects();
        let selected = host.service_abis();
        for effect in &expected {
            ensure!(
                selected.contains_key(&NativeDeploymentServiceKind::Effect(*effect)),
                "witness requires an independently selected {effect:?} service"
            );
        }
        match self {
            Self::Pure => {
                ensure!(
                    host.clock_inputs.is_none() && host.local_service_inputs.is_none(),
                    "pure witness cannot carry selected service inputs"
                );
                Ok(())
            }
            Self::Clock { inputs } => {
                if let Some(inputs) = inputs {
                    inputs.validate()?;
                }
                ensure!(
                    host.clock_inputs.as_ref() == inputs.as_ref(),
                    "Clock witness inputs differ from the independent host admission"
                );
                Ok(())
            }
            Self::DeterministicLocal {
                environment_key,
                environment_value,
                language_seed,
                read_path,
                read_utf8,
                ..
            } => {
                ensure!(
                    host.profile == NativeDeploymentProfile::DeterministicDeployed,
                    "deterministic-local witness requires a deterministic selected product"
                );
                let inputs = host
                    .local_service_inputs
                    .as_ref()
                    .context("deterministic-local witness lacks selected local inputs")?;
                let environment = inputs.environment.as_ref().context(
                    "deterministic-local witness lacks environment/current-directory inputs",
                )?;
                ensure!(
                    environment
                        .variables
                        .entries()
                        .iter()
                        .any(|(key, value)| key == environment_key && value == environment_value),
                    "environment oracle differs from selected local inputs"
                );
                let filesystem = inputs
                    .filesystem
                    .as_ref()
                    .context("deterministic-local witness lacks filesystem inputs")?;
                ensure!(
                    filesystem.entries.iter().any(|entry| matches!(
                        entry,
                        NativeFilesystemEntry::File { path, bytes, .. }
                            if path == read_path && bytes == read_utf8.as_bytes()
                    )),
                    "filesystem oracle differs from selected local inputs"
                );
                ensure!(
                    inputs.random.as_ref().map(|random| random.language_seed)
                        == Some(*language_seed),
                    "RNG oracle differs from selected local inputs"
                );
                ensure!(
                    *language_seed < (1_u64 << 32),
                    "RNG oracle exceeds the canonical MATLAB direct-seed domain"
                );
                Ok(())
            }
            Self::HostedExternal {
                console_line,
                http_port,
                tcp_port,
            } => {
                ensure!(
                    host.profile == NativeDeploymentProfile::HostedNative,
                    "hosted-external witness requires a hosted selected product"
                );
                ensure!(
                    !console_line.is_empty()
                        && console_line.len() <= 256
                        && *http_port != 0
                        && *tcp_port != 0
                        && http_port != tcp_port,
                    "hosted-external witness controls are invalid"
                );
                Ok(())
            }
            Self::HostedCallback {
                interface_name,
                input,
                offset,
                expected,
                ..
            } => {
                validate_hosted_callback_inputs(host.profile, host.local_service_inputs.as_ref())?;
                matlab_text(interface_name)?;
                let actual = input
                    .checked_mul(2)
                    .and_then(|value| {
                        offset
                            .checked_mul(2)
                            .and_then(|offset| value.checked_add(offset))
                    })
                    .and_then(|value| value.checked_add(1))
                    .context("hosted-callback witness arithmetic exceeds int32")?;
                ensure!(
                    actual == *expected,
                    "hosted-callback expected result differs from fixture_apply semantics"
                );
                Ok(())
            }
        }
    }

    pub fn assert_lowered_effects(&self, lowered: &runmat_types::EffectSet) -> Result<()> {
        let service_effects = [
            EffectKind::Clock,
            EffectKind::EnvironmentRead,
            EffectKind::EnvironmentWrite,
            EffectKind::FilesystemRead,
            EffectKind::FilesystemWrite,
            EffectKind::Randomness,
            EffectKind::Network,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ]
        .into_iter()
        .filter(|effect| lowered.0.contains(effect))
        .collect::<BTreeSet<_>>();
        ensure!(
            service_effects == self.expected_effects(),
            "lowered service effects {service_effects:?} differ from the closed witness mode {:?}",
            self.expected_effects()
        );
        Ok(())
    }

    pub fn assert_lowered_services(
        &self,
        required: &BTreeSet<NativeDeploymentServiceKind>,
        host: &SelectedHostAdmission,
    ) -> Result<()> {
        let selected = host.service_abis();
        let expected = self.expected_services();
        ensure!(
            required == &expected,
            "lowered deployment-service demand {required:?} differs from witness demand {expected:?}"
        );
        ensure!(
            required.iter().all(|kind| selected.contains_key(kind)),
            "lowered deployment-service demand exceeds the independent service grant"
        );
        Ok(())
    }

    pub fn source(&self, base: &str) -> Result<String> {
        let (call, function) = match self {
            Self::Pure => return Ok(base.to_string()),
            Self::Clock { inputs } => ("clockWitness();", clock_source(inputs.as_ref())),
            Self::DeterministicLocal {
                environment_key,
                environment_value,
                environment_write_key,
                environment_write_value,
                language_seed,
                read_path,
                read_utf8,
                write_path,
                write_utf8,
            } => {
                for value in [
                    environment_key,
                    environment_value,
                    environment_write_key,
                    environment_write_value,
                    read_path,
                    read_utf8,
                    write_path,
                    write_utf8,
                ] {
                    matlab_text(value)?;
                }
                ("localWitness();", format!(
                    "\nfunction localWitness()\n assert(strcmp(getenv('{ek}'), '{ev}'));\n setenv('{wk}', '{wv}');\n assert(strcmp(getenv('{wk}'), '{wv}'));\n assert(~isempty(pwd()));\n assert(~isempty(tempdir()));\n readId = fopen('{rp}', 'r');\n assert(readId >= 0);\n incoming = fread(readId, inf, '*char')';\n fclose(readId);\n assert(strcmp(incoming, '{rv}'));\n writeId = fopen('{wp}', 'w');\n assert(writeId >= 0);\n fwrite(writeId, '{wvf}', 'char');\n fclose(writeId);\n verifyId = fopen('{wp}', 'r');\n assert(verifyId >= 0);\n written = fread(verifyId, inf, '*char')';\n fclose(verifyId);\n assert(strcmp(written, '{wvf}'));\n initialSample = rand();\n rng(uint64({seed}), 'twister');\n assert(rand() == initialSample);\nend\n",
                    ek = environment_key, ev = environment_value, wk = environment_write_key,
                    wv = environment_write_value, rp = read_path, rv = read_utf8,
                    wp = write_path, wvf = write_utf8, seed = language_seed,
                ))
            }
            Self::HostedExternal {
                console_line,
                http_port,
                tcp_port,
            } => {
                matlab_text(console_line)?;
                ("externalWitness();", format!(
                    "\nfunction externalWitness()\n disp('{line}');\n text = webread('http://127.0.0.1:{http}/witness');\n assert(strcmp(text, 'runmat-http'));\n client = tcpclient('127.0.0.1', {tcp});\n write(client, uint8(9));\n reply = read(client, 3, 'uint8');\n assert(isequal(double(reply), [1 2 3]));\n close(client);\nend\n",
                    line = console_line, http = http_port, tcp = tcp_port,
                ))
            }
            Self::HostedCallback {
                interface_name,
                input,
                offset,
                expected,
                ..
            } => {
                matlab_text(interface_name)?;
                ("callbackWitness();", format!(
                    "\nfunction callbackWitness()\n answer = calllib('{alias}', 'fixture_apply', int32({input}), @(value) value + int32({offset}));\n assert(answer == int32({expected}));\nend\nfunction testCallbackError()\n callbackErrorWitness();\nend\nfunction callbackErrorWitness()\n calllib('{alias}', 'fixture_apply', int32({input}), @callbackFailure);\nend\nfunction y = callbackFailure(value)\n error('RunMat:NativePaired:CallbackFailure', 'paired callback failure');\n y = value;\nend\n",
                    alias = interface_name,
                ))
            }
        };
        Ok(base.replace(
            "function y = sharedAnswer()\n y =",
            &format!("function y = sharedAnswer()\n {call}\n y ="),
        ) + &function)
    }

    pub fn loopback_ports(&self) -> Option<(u16, u16)> {
        match self {
            Self::HostedExternal {
                http_port,
                tcp_port,
                ..
            } => Some((*http_port, *tcp_port)),
            _ => None,
        }
    }

    pub fn callback_artifact(&self) -> Result<Option<super::callback::CallbackArtifact>> {
        let Self::HostedCallback {
            artifact_bundle,
            expected_bundle_digest,
            foreign_policy,
            expected_foreign_policy_digest,
            interface_name,
            artifact_identity,
            library_digest,
            ..
        } = self
        else {
            return Ok(None);
        };
        super::callback::load(
            artifact_bundle,
            expected_bundle_digest,
            foreign_policy,
            expected_foreign_policy_digest,
            interface_name,
            artifact_identity,
            library_digest,
        )
        .map(Some)
    }

    pub fn control_count(&self) -> usize {
        if self.has_callback_error_control() {
            4
        } else {
            3
        }
    }

    pub fn has_callback_error_control(&self) -> bool {
        matches!(self, Self::HostedCallback { .. })
    }

    pub fn assert_test_result(
        &self,
        display_name: &str,
        result: &runmat_execution_artifact::target_test::NativeTargetTestResult,
    ) -> Result<()> {
        if self.has_callback_error_control() && display_name.ends_with("/testCallbackError") {
            ensure!(
                result.diagnostic.contains("paired callback failure"),
                "foreign callback failure lost its originating diagnostic: {}",
                result.diagnostic
            );
        }
        Ok(())
    }

    pub fn assert_application_output(&self, stdout: &[u8]) -> Result<()> {
        if let Self::HostedExternal { console_line, .. } = self {
            ensure!(
                stdout
                    .windows(console_line.len())
                    .any(|window| window == console_line.as_bytes()),
                "selected console did not publish the hosted witness line"
            );
        }
        Ok(())
    }

    fn expected_effects(&self) -> BTreeSet<EffectKind> {
        match self {
            Self::Pure => BTreeSet::new(),
            Self::Clock { .. } => BTreeSet::from([EffectKind::Clock]),
            Self::DeterministicLocal { .. } => BTreeSet::from([
                EffectKind::EnvironmentRead,
                EffectKind::EnvironmentWrite,
                EffectKind::FilesystemRead,
                EffectKind::FilesystemWrite,
                EffectKind::Randomness,
            ]),
            Self::HostedExternal { .. } => {
                BTreeSet::from([EffectKind::Network, EffectKind::UserInterface])
            }
            Self::HostedCallback { .. } => BTreeSet::from([
                EffectKind::EnvironmentRead,
                EffectKind::EnvironmentWrite,
                EffectKind::HostCallback,
            ]),
        }
    }

    fn expected_services(&self) -> BTreeSet<NativeDeploymentServiceKind> {
        self.expected_effects()
            .into_iter()
            .filter_map(NativeDeploymentServiceKind::for_effect)
            .chain(
                matches!(self, Self::HostedCallback { .. })
                    .then_some(NativeDeploymentServiceKind::HostedDynamicDispatch),
            )
            .collect()
    }
}

fn matlab_text(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 4096 || value.contains(['\0', '\n', '\r', '\'']) {
        bail!("witness text is empty, unbounded or not safely representable as a literal");
    }
    Ok(())
}

fn validate_hosted_callback_inputs(
    profile: NativeDeploymentProfile,
    inputs: Option<&runmat_execution::host::NativeLocalServiceInputs>,
) -> Result<()> {
    ensure!(
        profile == NativeDeploymentProfile::HostedNative,
        "hosted-callback witness requires a hosted selected product"
    );
    // Hosted environment ports are ambient services with independently
    // admitted issuers. Virtual local payloads belong only to deterministic
    // deployment and are rejected by the canonical linked local owner.
    ensure!(
        inputs.is_none(),
        "hosted-callback witness cannot carry deterministic local inputs"
    );
    Ok(())
}

fn clock_source(inputs: Option<&NativeClockInputs>) -> String {
    let mut body = String::from("\nfunction clockWitness()\n a = now();\n stamp = clock();\n");
    if let Some(inputs) = inputs {
        body.push_str(&format!(
            " expected = datenum(1970,1,1) + ({seconds} + {nanos}/1e9 + {offset})/86400;\n assert(abs(a - expected)*86400 < 0.0001);\n assert(abs(datenum(stamp) - expected - {step}/1e9/86400)*86400 < 0.0001);\n",
            seconds = inputs.initial_wall_unix_seconds,
            nanos = inputs.initial_wall_nanosecond,
            offset = inputs.local_offset_seconds,
            step = inputs.read_step_nanos,
        ));
    } else {
        body.push_str(" assert(a > datenum(2000,1,1));\n assert(datenum(stamp) >= a);\n");
    }
    body.push_str(" tic;\n pause(0.002);\n elapsed = toc;\n assert(elapsed >= 0.002);\n b = now();\n assert(b > a);\n");
    match inputs.map(|inputs| &inputs.process_cpu) {
        Some(None) => body.push_str(" unavailable = false;\n try\n  cputime();\n catch problem\n  assert(strcmp(problem.identifier, 'RunMat:cputime:Unavailable'));\n  unavailable = true;\n end\n assert(unavailable);\n"),
        Some(Some(cpu)) => body.push_str(&format!(" cpu1 = cputime();\n cpu2 = cputime();\n assert(abs(cpu1 - {initial}/1e9) < 1e-5);\n assert(abs(cpu2 - cpu1 - {step}/1e9) < 1e-5);\n", initial = cpu.initial_nanos, step = cpu.read_step_nanos)),
        None => body.push_str(" cpu1 = cputime();\n cpu2 = cputime();\n assert(cpu1 >= 0);\n assert(cpu2 >= cpu1);\n"),
    }
    body.push_str("end\n");
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_callback_uses_ambient_services_without_virtual_inputs() {
        let inputs = runmat_execution::host::NativeLocalServiceInputs {
            schema_version: 1,
            environment: None,
            random: None,
            filesystem: None,
        };
        assert!(
            validate_hosted_callback_inputs(NativeDeploymentProfile::HostedNative, None).is_ok()
        );
        assert!(validate_hosted_callback_inputs(
            NativeDeploymentProfile::HostedNative,
            Some(&inputs)
        )
        .is_err());
        assert!(validate_hosted_callback_inputs(
            NativeDeploymentProfile::DeterministicDeployed,
            None
        )
        .is_err());
        assert!(validate_hosted_callback_inputs(
            NativeDeploymentProfile::DeterministicDeployed,
            Some(&inputs)
        )
        .is_err());
    }
}
