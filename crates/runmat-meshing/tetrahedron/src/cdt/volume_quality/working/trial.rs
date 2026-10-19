//! First-write quality undo follows the same nested lifetime as cavity undo.
use super::*;
use std::ops::{Deref, DerefMut};
pub(in crate::cdt) struct QualityTrial<'a> {
    owner: &'a mut WorkingVolumeQuality,
    committed: bool,
}

impl WorkingVolumeQuality {
    pub(in crate::cdt) fn trial(&mut self) -> QualityTrial<'_> {
        self.journals.push(BTreeMap::new());
        QualityTrial {
            owner: self,
            committed: false,
        }
    }
}

impl QualityTrial<'_> {
    pub(in crate::cdt) fn commit(mut self) {
        self.owner.journals.pop().expect("live quality trial");
        self.committed = true;
    }
}
impl Deref for QualityTrial<'_> {
    type Target = WorkingVolumeQuality;
    fn deref(&self) -> &Self::Target {
        self.owner
    }
}
impl DerefMut for QualityTrial<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.owner
    }
}
impl Drop for QualityTrial<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let journal = self.owner.journals.pop().expect("live quality trial");
            // Every surviving parent's first-write entry was already recorded
            // before the nested edit; restoration leaves that entry unchanged.
            for (key, entry) in journal {
                self.owner.replace(key, entry.map(Arc::unwrap_or_clone));
            }
        }
    }
}
