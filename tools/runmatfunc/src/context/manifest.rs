use anyhow::Result;

use crate::builtin::inventory;
use crate::builtin::metadata::BuiltinManifest;

pub fn build_manifest() -> Result<BuiltinManifest> {
    inventory::collect_manifest()
}
