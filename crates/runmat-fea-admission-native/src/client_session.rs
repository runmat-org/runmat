//! Run-scoped lease transport. Local personal credentials can rotate while
//! the server, organization, lease and driver authority stay fixed.

use super::{auth_denial, denial};
use runmat_runtime::analysis::fea_run_admission::FeaRunDenied;
use runmat_server_client::{auth, simulation::SimulationLeaseClient};

pub(super) struct LeaseClientSession {
    pub(super) client: SimulationLeaseClient,
    pub(super) personal: Option<PersonalLeaseAuth>,
}

pub(super) struct PersonalLeaseAuth {
    pub(super) server: String,
    pub(super) config: auth::RemoteConfig,
    pub(super) token: String,
}

impl LeaseClientSession {
    pub(super) async fn current(&mut self) -> Result<&SimulationLeaseClient, FeaRunDenied> {
        if let Some(personal) = &mut self.personal {
            // Reload before renewal so this long-lived permit does not save
            // its initial config over a later CLI login or organization edit.
            let mut config =
                auth::RemoteConfig::load().map_err(|_| FeaRunDenied::LicenseInvalid)?;
            let server = auth::resolve_server_url(&config, None)
                .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            if server != personal.server
                || config.credential_store.unwrap_or_default()
                    != personal.config.credential_store.unwrap_or_default()
                || config.token_endpoint != personal.config.token_endpoint
                || config.token_client_id != personal.config.token_client_id
            {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            // The auth resolver's own load/save is not a cross-process CAS;
            // simultaneous login and refresh still need coordination there.
            let token = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                auth::resolve_auth_token(&mut config, &personal.server),
            )
            .await
            .map_err(|_| FeaRunDenied::ServerUnreachable)?
            .map_err(auth_denial)?;
            personal.config = config;
            self.install_personal_token(token)?;
        }
        Ok(&self.client)
    }

    // The returned bool distinguishes HTTP-pool reuse from bearer rotation.
    pub(super) fn install_personal_token(&mut self, token: String) -> Result<bool, FeaRunDenied> {
        let Some(personal) = &mut self.personal else {
            return Ok(false);
        };
        if token == personal.token {
            return Ok(false);
        }
        let replacement = SimulationLeaseClient::new(&personal.server, &token).map_err(denial)?;
        self.client = replacement;
        personal.token = token;
        Ok(true)
    }
}

#[cfg(test)]
mod tests;
