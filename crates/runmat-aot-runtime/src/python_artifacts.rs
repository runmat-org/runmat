use runmat_python::PythonArtifactBundle;

pub(crate) fn install(
    bytes: &[u8],
    adapter: &runmat_runtime::foreign::PythonAdapter,
) -> Result<(), String> {
    let bundle = PythonArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone Python artifact bundle is invalid: {error}"))?;
    if bundle.environment.is_none() && bundle.artifacts.is_empty() {
        return Ok(());
    }
    adapter
        .install_portable_artifact_bundle(&bundle)
        .map_err(|error| format!("standalone Python artifacts failed to install: {error}"))
}

#[cfg(test)]
mod tests {
    use runmat_runtime::foreign::ForeignAdapter as _;

    use super::*;

    #[test]
    fn installs_the_exact_embedded_python_environment() {
        let Ok(installation) =
            runmat_python::discover_python(&runmat_python::PythonDiscoveryRequest::default())
        else {
            return;
        };
        let environment = runmat_python::PythonEnvironmentIdentity::from_installation(
            &installation,
            runmat_python::PythonExecutionMode::InProcess,
        );
        let expected = environment.artifact_identity();
        let bundle = PythonArtifactBundle::new(environment, Vec::new()).unwrap();
        let adapter = runmat_runtime::foreign::PythonAdapter::new(
            runmat_runtime::foreign::ForeignHandleRegistry::default(),
        )
        .unwrap();
        install(&bundle.canonical_bytes().unwrap(), &adapter).unwrap();
        assert!(adapter.descriptor().artifact_identities.contains(&expected));
    }
}
