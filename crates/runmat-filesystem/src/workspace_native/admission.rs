//! Native requested phase residency, rederived before source content allocation.
//! Raw file bytes stay bytes; metadata uses the shared structural codec estimate.
//! Allocator rounding, OS buffers and measured RSS are not established here.
use super::{capture::inspection::CapturePlan, invalid, *};
use crate::workspace::{WorkspaceEdit, WorkspacePublicationRequest};
use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};
use serde::{ser::SerializeSeq, Serialize, Serializer};

fn add(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("workspace phase byte count overflow"))
}
fn multiply(a: u64, b: u64) -> io::Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("workspace phase byte count overflow"))
}
fn metadata(value: &(impl Serialize + ?Sized), maximum: u64) -> io::Result<u64> {
    serialized_materialization_resident_bytes_estimate(
        value,
        CanonicalLimits::content_bounded(
            usize::try_from(maximum).map_err(io::Error::other)?,
            4096,
            32,
        ),
    )
    .map_err(io::Error::other)
}
fn require(bytes: u64, maximum: u64) -> io::Result<u64> {
    if bytes > maximum {
        return Err(crate::workspace_access::resource_error(bytes, maximum));
    }
    Ok(bytes)
}
fn content(plan: &CapturePlan) -> io::Result<u64> {
    plan.files
        .iter()
        .try_fold(0, |sum, file| add(sum, file.length))
}
fn writes(request: &WorkspacePublicationRequest) -> io::Result<u64> {
    request.edits.iter().try_fold(0, |sum, edit| {
        add(
            sum,
            match edit {
                WorkspaceEdit::Write { bytes, .. } => bytes.len() as u64,
                _ => 0,
            },
        )
    })
}

// Only borrowed metadata: never serialize numeric file contents merely to plan storage.
#[derive(Serialize)]
enum EditShape<'a> {
    Write { path: &'a Path, length: u64 },
    Remove { path: &'a Path },
    Rename { from: &'a Path, to: &'a Path },
}
struct Edits<'a>(&'a [WorkspaceEdit]);
impl Serialize for Edits<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for edit in self.0 {
            let shape = match edit {
                WorkspaceEdit::Write { path, bytes } => EditShape::Write {
                    path,
                    length: bytes.len() as u64,
                },
                WorkspaceEdit::Remove { path } => EditShape::Remove { path },
                WorkspaceEdit::Rename { from, to } => EditShape::Rename { from, to },
            };
            sequence.serialize_element(&shape)?;
        }
        sequence.end()
    }
}

pub(super) fn request(request: &WorkspacePublicationRequest, maximum: u64) -> io::Result<()> {
    let shape = (
        &request.root,
        &request.excluded_subtrees,
        Edits(&request.edits),
    );
    require(add(metadata(&shape, maximum)?, writes(request)?)?, maximum)?;
    Ok(())
}
pub(super) fn capture(plan: &CapturePlan, maximum: u64) -> io::Result<u64> {
    // The metadata allowance includes the plan, snapshot and revision transport.
    require(add(metadata(plan, maximum)?, content(plan)?)?, maximum)
}
pub(super) fn publication(
    plan: &CapturePlan,
    request: &WorkspacePublicationRequest,
    maximum: u64,
) -> io::Result<u64> {
    let shape = (plan, &request.excluded_subtrees, Edits(&request.edits));
    // Capture->private tree and private tree->post-edit snapshot are the two
    // simultaneous raw content views. Edits may append all their bytes. Four
    // conservative metadata views cover inspection, indexed validation, snapshots
    // and journal preparation, including shared fixed transport allowances.
    let payload = multiply(add(content(plan)?, writes(request)?)?, 2)?;
    require(
        add(payload, multiply(metadata(&shape, maximum)?, 4)?)?,
        maximum,
    )
}

/// A scoped successful commit must admit its future cold plan decode, not only
/// the cheaper preparation writer. The shared canonical profile is conservative
/// relative to the existing JSON plan's actual encoded/value inventory.
pub(super) fn journal(plan: &[super::journal::FinalEdit], maximum: u64) -> io::Result<()> {
    let estimate = serialized_materialization_resident_bytes_estimate(
        plan,
        super::journal::recovery_limits(usize::try_from(maximum).map_err(io::Error::other)?),
    )
    .map_err(io::Error::other)?;
    require(
        add(estimate, super::journal::retained_recovery_metadata_bytes())?,
        maximum,
    )?;
    Ok(())
}
