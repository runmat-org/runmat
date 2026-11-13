//! Maintenance inventories retain their reservation through validation and deletion.
use super::*;

pub(super) struct Digests<'a, 'store> {
    store: &'a access::FilesystemAccess<'store>,
    values: HashSet<Digest>,
    reservation: access::Reservation<'a>,
}
impl<'a, 'store> Digests<'a, 'store> {
    pub(super) fn new(
        store: &'a access::FilesystemAccess<'store>,
    ) -> Result<Self, AnalysisPublicationError> {
        Ok(Self {
            store,
            values: HashSet::new(),
            reservation: store.reserve_resident(
                store.retained_estimate(&Vec::<Digest>::new())?
                    + std::mem::size_of::<HashSet<Digest>>() as u64,
            )?,
        })
    }
    pub(super) fn insert(&mut self, digest: Digest) -> Result<(), AnalysisPublicationError> {
        if !self.values.contains(&digest) {
            // Shared borrowed metadata profiling includes conservative entry and
            // collection overhead before hash-table growth; duplicate references cost nothing.
            self.reservation
                .grow(self.store.retained_increment_estimate(&[digest])?)?;
            self.values.insert(digest);
        }
        Ok(())
    }
    pub(super) fn contains(&self, digest: &Digest) -> bool {
        self.values.contains(digest)
    }
    pub(super) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &Digest> {
        self.values.iter()
    }
}

pub(super) struct Paths<'a, 'store> {
    store: &'a access::FilesystemAccess<'store>,
    values: Vec<PathBuf>,
    reservation: access::Reservation<'a>,
}
impl<'a, 'store> Paths<'a, 'store> {
    pub(super) fn new(
        store: &'a access::FilesystemAccess<'store>,
    ) -> Result<Self, AnalysisPublicationError> {
        Ok(Self {
            store,
            values: Vec::new(),
            reservation: store.reserve_resident(
                store.retained_estimate(&Vec::<PathBuf>::new())?
                    + std::mem::size_of::<Vec<PathBuf>>() as u64,
            )?,
        })
    }
    pub(super) fn push(&mut self, path: &Path) -> Result<(), AnalysisPublicationError> {
        self.reservation
            .grow(self.store.retained_increment_estimate(&[path])?)?;
        self.values.push(path.to_path_buf());
        Ok(())
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &PathBuf> {
        self.values.iter()
    }
    /// The caller retains the guard while a provider consumes the owned request.
    pub(super) fn into_parts(self) -> (Vec<PathBuf>, access::Reservation<'a>) {
        (self.values, self.reservation)
    }
}
