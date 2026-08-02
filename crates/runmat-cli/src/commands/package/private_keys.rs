use super::registry_transport::registry_client;
use crate::cli::{PackageKeyCommand, PackageKeyTarget};
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_package_cache_native::registry::RecipientKeyPair;
use runmat_server_client::auth::{resolve_server_url, RemoteConfig};
use zeroize::Zeroizing;

pub(super) async fn execute(command: PackageKeyCommand) -> Result<()> {
    match command {
        PackageKeyCommand::Register(target) => register(target).await,
        PackageKeyCommand::List(target) => list(target).await,
        PackageKeyCommand::Revoke { target, key_id } => revoke(target, &key_id).await,
    }
}

async fn register(target: PackageKeyTarget) -> Result<()> {
    let origin = registry_origin(&target)?;
    let client = registry_client(&origin).await.map_err(anyhow::Error::msg)?;
    let generated = RecipientKeyPair::generate("pending").map_err(anyhow::Error::msg)?;
    let public = generated.public_key().map_err(anyhow::Error::msg)?;
    let registered = client
        .register_recipient_key(&target.namespace, &target.name, &public.public_key)
        .await
        .context("failed to register private package recipient key")?;
    if registered.algorithm != "p256"
        || registered.public_key != public.public_key
        || registered.fingerprint != public.fingerprint.to_string()
    {
        let _ = client
            .revoke_recipient_key(&target.namespace, &target.name, &registered.id)
            .await;
        bail!("registry returned recipient key material that does not match the generated key");
    }
    let secret = generated.secret_bytes();
    let key = RecipientKeyPair::from_secret_bytes(registered.id.clone(), *secret)
        .map_err(anyhow::Error::msg)?;
    if let Err(error) = store_key(&origin, &key) {
        let _ = client
            .revoke_recipient_key(&target.namespace, &target.name, &registered.id)
            .await;
        return Err(error.context(
            "recipient key registration was rolled back because secure local storage failed",
        ));
    }
    println!(
        "{} {} {}",
        registered.id, registered.fingerprint, registered.created_at
    );
    Ok(())
}

async fn list(target: PackageKeyTarget) -> Result<()> {
    let origin = registry_origin(&target)?;
    let keys = registry_client(&origin)
        .await
        .map_err(anyhow::Error::msg)?
        .recipient_keys(&target.namespace, &target.name)
        .await
        .context("failed to list private package recipient keys")?;
    for key in keys {
        let state = if key.revoked_at.is_some() {
            "revoked"
        } else if load_key(&origin, &key.id)
            .map_err(anyhow::Error::msg)?
            .is_some()
        {
            "available"
        } else {
            "private-key-missing"
        };
        println!("{} {} {state}", key.id, key.fingerprint);
    }
    Ok(())
}

async fn revoke(target: PackageKeyTarget, key_id: &str) -> Result<()> {
    let origin = registry_origin(&target)?;
    let key = registry_client(&origin)
        .await
        .map_err(anyhow::Error::msg)?
        .revoke_recipient_key(&target.namespace, &target.name, key_id)
        .await
        .context("failed to revoke private package recipient key")?;
    delete_key(&origin, key_id)?;
    println!("{} revoked", key.id);
    Ok(())
}

pub(super) fn configured_registry_origin(explicit: Option<&str>) -> Result<String> {
    if let Some(origin) = explicit {
        return Ok(origin.trim_end_matches('/').to_string());
    }
    let config = RemoteConfig::load().context("failed to load remote configuration")?;
    resolve_server_url(&config, None).context("failed to resolve the configured registry origin")
}

fn registry_origin(target: &PackageKeyTarget) -> Result<String> {
    configured_registry_origin(target.registry.as_deref())
}

fn store_key(origin: &str, key: &RecipientKeyPair) -> Result<()> {
    let encoded = Zeroizing::new(URL_SAFE_NO_PAD.encode(key.secret_bytes().as_ref()));
    keyring_entry(origin, key.id())?
        .set_password(encoded.as_str())
        .context("failed to store the private package key in the OS credential store")
}

fn load_key(origin: &str, key_id: &str) -> Result<Option<RecipientKeyPair>, String> {
    let entry = keyring_entry(origin, key_id).map_err(|error| error.to_string())?;
    let encoded = match entry.get_password() {
        Ok(value) => Zeroizing::new(value),
        Err(keyring::Error::NoEntry) => return Ok(None),
        Err(error) => return Err(format!("failed to access the OS credential store: {error}")),
    };
    let decoded = Zeroizing::new(
        URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .map_err(|_| "stored private package key is corrupt".to_string())?,
    );
    let secret: [u8; 32] = decoded
        .as_slice()
        .try_into()
        .map_err(|_| "stored private package key has an invalid length".to_string())?;
    RecipientKeyPair::from_secret_bytes(key_id, secret).map(Some)
}

fn delete_key(origin: &str, key_id: &str) -> Result<()> {
    match keyring_entry(origin, key_id)?.delete_password() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error).context("failed to remove the private package key"),
    }
}

fn keyring_entry(origin: &str, key_id: &str) -> Result<keyring::Entry> {
    let account = keyring_account(origin, key_id);
    keyring::Entry::new("runmat", &account).context("failed to open the OS credential store")
}

fn keyring_account(origin: &str, key_id: &str) -> String {
    let origin_digest = runmat_package::ContentDigest::sha256(origin);
    format!("package-key:{origin_digest}:{key_id}")
}

#[cfg(test)]
mod tests {
    use super::keyring_account;

    #[test]
    fn credential_accounts_do_not_alias_similarly_spelled_registry_origins() {
        assert_ne!(
            keyring_account("https://packages-a.runmat.test", "pkr_1"),
            keyring_account("https://packages_a.runmat.test", "pkr_1")
        );
        assert_ne!(
            keyring_account("https://packages-a.runmat.test", "pkr_1"),
            keyring_account("https://packages-a.runmat.test", "pkr_2")
        );
    }
}
