use std::{io, sync::Arc};

use runmat_execution::ForeignPolicyRef;
use runmat_execution_artifact::archive::{read_bundle, write_bundle, ArchiveLimits};
use runmat_execution_artifact::{ExecutableForm, ExecutionBundle};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_types::InteropManifest;

/// One immutable foreign-artifact closure for one already-issued V3 unit.
///
/// This owns archive bytes, not paths, a RuntimeContext, or a foreign adapter.
/// The issuing body or declaration product retains it across publication and
/// definition retirement; a graph snapshot only clones the session-local Rc.
#[derive(Clone)]
pub struct SelectedUnitInteropReceipt {
    archive: Arc<[u8]>,
    canonical_v3: Arc<[u8]>,
    source_owner: SelectedProgramSourceOwner,
    foreign_policy: ForeignPolicyRef,
}

struct BoundedArchiveWriter {
    bytes: Vec<u8>,
    max_total_bytes: u64,
}

impl io::Write for BoundedArchiveWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let incoming = u64::try_from(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "archive size exceeds u64"))?;
        let next = u64::try_from(self.bytes.len())
            .ok()
            .and_then(|current| current.checked_add(incoming))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "archive size overflow"))?;
        if next > self.max_total_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "selected foreign archive exceeds its total byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl std::fmt::Debug for SelectedUnitInteropReceipt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedUnitInteropReceipt")
            .field("archive_bytes", &self.archive.len())
            .finish_non_exhaustive()
    }
}

impl SelectedUnitInteropReceipt {
    /// Validate and archive one exact selected unit at publication. A truly
    /// pure unit without captured policy has no archive or second code owner;
    /// an empty-interop unit with a policy keeps its exact archive receipt.
    pub fn from_bundle(
        bundle: Option<ExecutionBundle>,
        canonical_v3: &Arc<[u8]>,
        source_owner: &SelectedProgramSourceOwner,
        limits: ArchiveLimits,
    ) -> Result<Option<std::rc::Rc<Self>>, String> {
        let admitted =
            super::admit_executable_unit(canonical_v3).map_err(|error| error.to_string())?;
        if !source_owner.is_live() || admitted.admitted_source_map() != source_owner.portable_map()
        {
            return Err("selected foreign closure differs from its issuing source owner".into());
        }
        let expected = &admitted.envelope().manifest;
        if expected.interop == InteropManifest::empty() && bundle.is_none() {
            return Ok(None);
        }
        let bundle = bundle.ok_or("selected foreign interop has no exact bundle")?;
        bundle.validate().map_err(|error| error.to_string())?;
        if bundle.manifest.program_revision != expected.identity.program {
            return Err("selected foreign bundle revision differs from its V3 unit".into());
        }
        if bundle.manifest.recipes.len() != 1 || bundle.manifest.artifacts.len() != 1 {
            return Err("selected foreign bundle must contain one exact V3 recipe".into());
        }
        let matching = bundle
            .manifest
            .artifacts
            .iter()
            .filter(|artifact| artifact.form == ExecutableForm::ExecutableUnitV4)
            .collect::<Vec<_>>();
        if matching.len() != 1 || matching[0].executable_bytes.as_slice() != canonical_v3.as_ref() {
            return Err("selected foreign bundle has no unique exact V3 artifact".into());
        }
        let recipe = bundle
            .manifest
            .recipes
            .iter()
            .find(|recipe| recipe.id().ok() == Some(matching[0].recipe_id))
            .ok_or("selected foreign bundle has no exact V3 recipe")?;
        if recipe.interop != expected.interop {
            return Err("selected foreign bundle interop differs from its V3 unit".into());
        }
        let foreign_policy = recipe
            .foreign_policy
            .as_ref()
            .map(runmat_execution_artifact::foreign_policy_ref)
            .transpose()
            .map_err(|error| error.to_string())?
            .ok_or("selected foreign archive has no captured policy")?;
        let mut archive = BoundedArchiveWriter {
            bytes: Vec::new(),
            max_total_bytes: limits.max_total_bytes,
        };
        write_bundle(&bundle, &mut archive, limits).map_err(|error| error.to_string())?;
        let decoded =
            read_bundle(archive.bytes.as_slice(), limits).map_err(|error| error.to_string())?;
        if decoded.identity().map_err(|error| error.to_string())?
            != bundle.identity().map_err(|error| error.to_string())?
        {
            return Err("selected foreign archive differs from its checked bundle".into());
        }
        Ok(Some(std::rc::Rc::new(Self {
            archive: Arc::from(archive.bytes),
            canonical_v3: Arc::clone(canonical_v3),
            source_owner: source_owner.clone(),
            foreign_policy,
        })))
    }

    pub fn archive(&self) -> &Arc<[u8]> {
        &self.archive
    }

    pub fn foreign_policy(&self) -> ForeignPolicyRef {
        self.foreign_policy
    }

    /// Exact immutable product/issuance agreement; an equal definition
    /// fingerprint alone cannot substitute a newer foreign closure.
    pub fn matches_selected_unit(
        &self,
        canonical_v3: &Arc<[u8]>,
        source_owner: &SelectedProgramSourceOwner,
    ) -> bool {
        self.canonical_v3.as_ref() == canonical_v3.as_ref()
            && self.source_owner.same_issuance(source_owner)
    }

    pub fn same_selected_admission(&self, other: &Self) -> bool {
        self.matches_selected_unit(&other.canonical_v3, &other.source_owner)
            && self.archive.as_ref() == other.archive.as_ref()
            && self.foreign_policy == other.foreign_policy
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::BoundedArchiveWriter;

    #[test]
    fn archive_limit_refuses_the_next_member_before_extending_bytes() {
        let mut writer = BoundedArchiveWriter {
            bytes: Vec::new(),
            max_total_bytes: 4,
        };
        writer.write_all(&[1, 2, 3]).expect("first member");
        assert!(writer.write_all(&[4, 5]).is_err());
        assert_eq!(writer.bytes, [1, 2, 3]);
    }
}
