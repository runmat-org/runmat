use std::path::PathBuf;

use runmat_execution_transport_native::control::DriverAuthority;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) struct RemoteDriverConfig {
    pub authority: DriverAuthority,
    pub endpoint_identity_file: PathBuf,
}

impl RemoteDriverConfig {
    pub fn from_env() -> NativeExecutionResult<Self> {
        let authority = DriverAuthority {
            server_url: required("RUNMAT_EXECUTION_SERVER_URL")?,
            run_id: required("RUNMAT_EXECUTION_RUN_ID")?,
            org_id: required("RUNMAT_EXECUTION_ORG_ID")?,
            project_id: required("RUNMAT_EXECUTION_PROJECT_ID")?,
            allocation_lease_id: required("RUNMAT_EXECUTION_ALLOCATION_ID")?,
            driver_lease_id: required("RUNMAT_EXECUTION_DRIVER_LEASE_ID")?,
            fencing_token: required("RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN")?
                .parse()
                .map_err(|_| invalid("driver fencing token is malformed"))?,
            credential: required("RUNMAT_EXECUTION_DRIVER_CREDENTIAL")?,
        };
        authority
            .validate()
            .map_err(|error| invalid(&error.to_string()))?;
        let endpoint_identity_file =
            PathBuf::from(required("RUNMAT_EXECUTION_ENDPOINT_IDENTITY_FILE")?);
        if !endpoint_identity_file.is_absolute() || !endpoint_identity_file.is_file() {
            return Err(invalid(
                "endpoint identity file must be an existing absolute file",
            ));
        }
        Ok(Self {
            authority,
            endpoint_identity_file,
        })
    }
}

fn required(name: &str) -> NativeExecutionResult<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(&format!("remote driver is missing {name}")))
}

fn invalid(message: &str) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.to_string())
}
