//! Admit original journal bytes and decoded demand before any recovered source mutation.
use super::*;

#[cfg(test)]
mod tests;

fn read_plan(
    root: &Path,
    directory: &Path,
    record: &Record,
    maximum_resident_bytes: u64,
) -> io::Result<Vec<FinalEdit>> {
    let maximum_wire_bytes = maximum_resident_bytes.min(512 * 1024 * 1024);
    let bytes =
        crate::workspace_access::file::read(&directory.join("plan.json"), maximum_wire_bytes)?;
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != record.plan_digest {
        return Err(invalid("workspace recovery content digest mismatch"));
    }
    runmat_canonical_codec::json_materialization_resident_bytes_estimate(
        &bytes,
        limits(usize::try_from(maximum_wire_bytes).map_err(io::Error::other)?),
        maximum_resident_bytes,
    )
    .map_err(io::Error::other)?;
    let plan: Vec<FinalEdit> = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if plan.len() > MAX_WORKSPACE_PUBLICATION_EDITS
        || plan
            .iter()
            .try_fold(0usize, |total, edit| {
                total.checked_add(match edit {
                    FinalEdit::Write { bytes, .. } => bytes.len(),
                    FinalEdit::Remove { .. } => 0,
                })
            })
            .is_none_or(|size| size > MAX_WORKSPACE_PUBLICATION_BYTES)
    {
        return Err(invalid("workspace recovery exceeds publication limits"));
    }
    // Validate the entire original path inventory before the first physical edit.
    for edit in &plan {
        let path = match edit {
            FinalEdit::Write { path, .. } | FinalEdit::Remove { path } => path,
        };
        protected(root, &confined(root, path)?)?;
    }
    Ok(plan)
}

pub(super) fn recover(root: &Path, maximum_resident_bytes: u64) -> io::Result<()> {
    let pending = root.join(JOURNAL).join("pending.json");
    ensure_no_symlink(root, &pending)?;
    let Some(id) = read_metadata::<WorkspacePublicationId>(&pending, 1024, maximum_resident_bytes)?
    else {
        return Ok(());
    };
    let directory = record_directory(root, id);
    ensure_no_symlink(root, &directory)?;
    for name in ["commit.json", "plan.json", "complete"] {
        ensure_no_symlink(root, &directory.join(name))?;
    }
    let record_budget = maximum_resident_bytes
        .checked_sub(std::mem::size_of::<WorkspacePublicationId>() as u64)
        .ok_or_else(|| {
            crate::workspace_access::resource_error(
                std::mem::size_of::<WorkspacePublicationId>() as u64,
                maximum_resident_bytes,
            )
        })?;
    let Some(record) = read_record_admitted(&directory, record_budget)? else {
        if directory.exists() {
            fs::remove_dir_all(directory)?;
        }
        return clear_pending(root);
    };
    if record.receipt.id != id {
        return Err(invalid("workspace receipt identity differs from journal"));
    }
    confirm(&directory, &record)?;
    if directory.join("complete").exists() {
        if directory.join("plan.json").exists() {
            fs::remove_file(directory.join("plan.json"))?;
            sync_directory(&directory)?;
        }
        return clear_pending(root);
    }
    let remaining = maximum_resident_bytes
        .checked_sub(retained_recovery_metadata_bytes())
        .ok_or_else(|| {
            crate::workspace_access::resource_error(
                retained_recovery_metadata_bytes(),
                maximum_resident_bytes,
            )
        })?;
    let plan = read_plan(root, &directory, &record, remaining)?;
    materialize(root, &directory, &plan)
}

pub(in crate::workspace_native) fn limits(
    maximum_encoded_bytes: usize,
) -> runmat_canonical_codec::CanonicalLimits {
    runmat_canonical_codec::CanonicalLimits::new(
        maximum_encoded_bytes,
        MAX_WORKSPACE_PUBLICATION_BYTES,
        65_536,
        64,
    )
}

/// Small journal metadata uses the same exact read/profile/decode path as plans.
pub(super) fn read_metadata<T: serde::de::DeserializeOwned>(
    path: &Path,
    wire_limit: u64,
    maximum: u64,
) -> io::Result<Option<T>> {
    let bytes = match crate::workspace_access::file::read(path, wire_limit.min(maximum)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    runmat_canonical_codec::json_materialization_resident_bytes_estimate(
        &bytes,
        limits(usize::try_from(wire_limit.min(maximum)).map_err(io::Error::other)?),
        maximum,
    )
    .map_err(io::Error::other)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(io::Error::other)
}
