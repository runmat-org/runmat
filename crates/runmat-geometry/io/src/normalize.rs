use runmat_geometry_core::GeometryAsset;
use sha2::{Digest, Sha256};

pub fn deterministic_import_fingerprint(
    asset: &GeometryAsset,
) -> Result<String, serde_json::Error> {
    let json = serde_json::to_vec(asset)?;
    let mut hasher = Sha256::new();
    hasher.update(json);
    let digest = hasher.finalize();
    Ok(format!("{:x}", digest))
}
