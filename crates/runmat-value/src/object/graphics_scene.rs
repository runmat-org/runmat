//! Nonportable candidate identity for one selected Figure scene.
//!
//! Issuing a candidate does not grant access to a scene. The Figure registry
//! admits it only while it is the exact current lease for that Figure. A
//! pending host event retains a strong lease, so a successor cannot acquire
//! the same identity through pointer reuse.

use std::sync::Arc;

#[derive(Debug)]
struct SceneIdentity {
    figure: u32,
}

#[derive(Clone, Debug)]
pub struct GraphicsSceneLease(Arc<SceneIdentity>);

impl GraphicsSceneLease {
    pub fn candidate(figure: u32) -> Self {
        Self(Arc::new(SceneIdentity { figure }))
    }

    pub fn figure(&self) -> u32 {
        self.0.figure
    }

    pub fn shares_lease(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_scene_keeps_old_identity_alive_across_successor() {
        let selected = GraphicsSceneLease::candidate(7);
        let pending = selected.clone();
        let successor = GraphicsSceneLease::candidate(7);
        assert!(selected.shares_lease(&pending));
        assert!(!pending.shares_lease(&successor));
        assert_eq!(successor.figure(), 7);
        drop(selected);
        assert!(!pending.shares_lease(&successor));
    }
}
