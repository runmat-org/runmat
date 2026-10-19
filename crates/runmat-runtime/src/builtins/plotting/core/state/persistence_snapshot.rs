//! Checked immutable snapshots of the shared plot registry.

use super::{registry, shared_figure::FigureOwner, FigureHandle, GraphicsSceneLease};
use runmat_plot::plots::Figure;
use std::collections::HashMap;
use thiserror::Error;

fn figure_owner_allocation_upper_bound(payload_bytes: usize) -> Option<usize> {
    // Both platform owners retain two counters before the payload. Include
    // worst-case alignment padding without relying on their private layouts.
    let header_and_padding = (2 * std::mem::size_of::<usize>())
        .checked_add(std::mem::align_of::<Figure>().saturating_sub(1))?;
    header_and_padding
        .checked_add(std::mem::size_of::<Figure>())?
        .checked_add(payload_bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FigureSnapshotScope {
    SharedGlobalRegistry,
}

#[derive(Clone, Debug)]
pub struct FigureSnapshotEntry {
    pub handle: FigureHandle,
    pub revision: u64,
    /// Exact runtime-only scene authority captured under the same registry
    /// guard as `figure`; `revision` remains only a presentation cache hint.
    pub scene_lease: GraphicsSceneLease,
    pub figure: FigureOwner,
}

impl FigureSnapshotEntry {
    #[cfg(any(all(feature = "gui", not(target_arch = "wasm32")), test))]
    pub(crate) fn retained_byte_upper_bound(&self) -> Option<usize> {
        Self::retained_byte_upper_bound_for_payload(self.figure.retained_snapshot_bytes()?)
    }

    pub(crate) fn retained_byte_upper_bound_for_payload(payload_bytes: usize) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(figure_owner_allocation_upper_bound(payload_bytes)?)
    }
}

#[derive(Clone, Debug)]
pub struct FigurePersistenceSnapshot {
    pub scope: FigureSnapshotScope,
    pub entries: Vec<FigureSnapshotEntry>,
    pub retained_bytes: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FigurePersistenceSnapshotError {
    #[error("figure handle {0} does not exist")]
    MissingFigure(u32),
    #[error("figure snapshot requested {requested} roots, exceeding limit {maximum}")]
    TooManyRoots { requested: usize, maximum: usize },
    #[error("figure snapshot size overflowed")]
    SizeOverflow,
    #[error("figure snapshot retains {retained_bytes} bytes, exceeding limit {maximum_bytes}")]
    BudgetExceeded {
        retained_bytes: usize,
        maximum_bytes: usize,
    },
}

pub fn capture_figure_persistence_snapshot(
    handles: &[FigureHandle],
    maximum_roots: usize,
    maximum_bytes: usize,
) -> Result<FigurePersistenceSnapshot, FigurePersistenceSnapshotError> {
    if handles.len() > maximum_roots {
        return Err(FigurePersistenceSnapshotError::TooManyRoots {
            requested: handles.len(),
            maximum: maximum_roots,
        });
    }
    let mut entries = Vec::with_capacity(handles.len());
    let registry = registry();
    let mut retained_bytes = std::mem::size_of::<FigureSnapshotEntry>()
        .checked_mul(entries.capacity())
        .ok_or(FigurePersistenceSnapshotError::SizeOverflow)?;
    if retained_bytes > maximum_bytes {
        return Err(FigurePersistenceSnapshotError::BudgetExceeded {
            retained_bytes,
            maximum_bytes,
        });
    }
    let mut selected = HashMap::<FigureHandle, (u64, GraphicsSceneLease)>::new();

    for handle in handles {
        let state = registry
            .figures
            .get(handle)
            .ok_or_else(|| FigurePersistenceSnapshotError::MissingFigure(handle.as_u32()))?;
        if let std::collections::hash_map::Entry::Vacant(entry) = selected.entry(*handle) {
            let figure_bytes = figure_owner_allocation_upper_bound(
                state
                    .figure
                    .retained_snapshot_bytes()
                    .ok_or(FigurePersistenceSnapshotError::SizeOverflow)?,
            )
            .ok_or(FigurePersistenceSnapshotError::SizeOverflow)?;
            retained_bytes = retained_bytes
                .checked_add(figure_bytes)
                .ok_or(FigurePersistenceSnapshotError::SizeOverflow)?;
            if retained_bytes > maximum_bytes {
                return Err(FigurePersistenceSnapshotError::BudgetExceeded {
                    retained_bytes,
                    maximum_bytes,
                });
            }
            entry.insert((state.revision, state.scene_lease.clone()));
        }
    }

    let mut snapshots = HashMap::<FigureHandle, FigureOwner>::with_capacity(selected.len());
    for handle in selected.keys() {
        let state = registry
            .figures
            .get(handle)
            .expect("all figure handles were validated during snapshot admission");
        snapshots.insert(*handle, state.figure.snapshot());
    }

    for handle in handles {
        let (revision, scene_lease) = &selected[handle];
        entries.push(FigureSnapshotEntry {
            handle: *handle,
            revision: *revision,
            scene_lease: scene_lease.clone(),
            figure: snapshots[handle].clone(),
        });
    }

    Ok(FigurePersistenceSnapshot {
        scope: FigureSnapshotScope::SharedGlobalRegistry,
        entries,
        retained_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::state::{
        clone_figure, import_figure, lock_plot_test_registry, reset_plot_state, set_figure_name,
    };
    use runmat_plot::plots::surface::ColorMap;
    use runmat_plot::plots::LinePlot;
    use std::sync::Arc;

    #[test]
    fn snapshot_is_revisioned_immutable_and_charged_once_per_figure() {
        let _guard = lock_plot_test_registry();
        reset_plot_state();
        let mut figure = Figure::new();
        figure.set_name("before");
        let handle = import_figure(figure);
        let expected_bytes = figure_owner_allocation_upper_bound(
            clone_figure(handle)
                .expect("figure")
                .retained_snapshot_bytes()
                .expect("bounded figure"),
        )
        .expect("bounded shared allocation");

        let expected_bytes = expected_bytes + 2 * std::mem::size_of::<FigureSnapshotEntry>();
        let snapshot = capture_figure_persistence_snapshot(&[handle, handle], 2, expected_bytes)
            .expect("duplicate roots share retained payload");
        assert_eq!(snapshot.scope, FigureSnapshotScope::SharedGlobalRegistry);
        assert_eq!(snapshot.entries.len(), 2);
        assert_eq!(snapshot.retained_bytes, expected_bytes);
        assert!(FigureOwner::ptr_eq(
            &snapshot.entries[0].figure,
            &snapshot.entries[1].figure
        ));
        assert!(snapshot.entries[0]
            .scene_lease
            .shares_lease(&snapshot.entries[1].scene_lease));
        assert_eq!(snapshot.entries[0].scene_lease.figure(), handle.as_u32());
        assert_eq!(
            capture_figure_persistence_snapshot(
                &[handle, handle],
                2,
                expected_bytes - std::mem::size_of::<FigureSnapshotEntry>(),
            )
            .unwrap_err(),
            FigurePersistenceSnapshotError::BudgetExceeded {
                retained_bytes: expected_bytes,
                maximum_bytes: expected_bytes - std::mem::size_of::<FigureSnapshotEntry>(),
            }
        );

        set_figure_name(handle, "after".into()).expect("mutate live figure");
        assert_eq!(snapshot.entries[0].figure.name.as_deref(), Some("before"));
        let successor = capture_figure_persistence_snapshot(&[handle], 1, usize::MAX)
            .expect("capture selected successor");
        assert!(!snapshot.entries[0]
            .scene_lease
            .shares_lease(&successor.entries[0].scene_lease));
        assert_eq!(
            clone_figure(handle).expect("live figure").name.as_deref(),
            Some("after")
        );
    }

    #[test]
    fn snapshot_rejects_retained_payload_before_creating_a_snapshot() {
        let _guard = lock_plot_test_registry();
        reset_plot_state();
        let mut figure = Figure::new();
        figure.set_name("x".repeat(4096));
        let handle = import_figure(figure);
        let required = std::mem::size_of::<FigureSnapshotEntry>()
            + figure_owner_allocation_upper_bound(
                clone_figure(handle)
                    .expect("figure")
                    .retained_snapshot_bytes()
                    .expect("bounded figure"),
            )
            .expect("bounded shared allocation");

        let error = capture_figure_persistence_snapshot(&[handle], 1, required - 1)
            .expect_err("one byte below exact retained charge must reject");
        assert_eq!(
            error,
            FigurePersistenceSnapshotError::BudgetExceeded {
                retained_bytes: required,
                maximum_bytes: required - 1,
            }
        );
    }

    #[test]
    fn snapshot_budget_includes_reference_counted_host_colormap_tables() {
        let _guard = lock_plot_test_registry();
        reset_plot_state();
        let baseline = Figure::new()
            .retained_snapshot_bytes()
            .expect("bounded baseline");
        let mut figure = Figure::new();
        let rows: Arc<[[f32; 3]]> = vec![[0.1, 0.2, 0.3]; 37].into();
        figure.set_figure_colormap(ColorMap::Listed(rows), 37);
        let retained = figure
            .retained_snapshot_bytes()
            .expect("bounded listed colormap");
        let table_allocation = 2 * std::mem::size_of::<usize>()
            + std::mem::align_of::<[f32; 3]>()
                .max(std::mem::align_of::<usize>())
                .saturating_sub(1)
            + 37 * std::mem::size_of::<[f32; 3]>();

        assert!(
            retained >= baseline + table_allocation,
            "an immutable snapshot pins reference-counted host tables"
        );
    }

    #[test]
    fn snapshot_rejects_duplicate_root_count_before_entry_allocation() {
        let _guard = lock_plot_test_registry();
        reset_plot_state();
        let handle = import_figure(Figure::new());
        let roots = vec![handle; 33];

        assert_eq!(
            capture_figure_persistence_snapshot(&roots, 32, usize::MAX).unwrap_err(),
            FigurePersistenceSnapshotError::TooManyRoots {
                requested: 33,
                maximum: 32,
            }
        );
    }

    #[test]
    fn snapshot_budget_counts_spare_vector_and_string_capacity() {
        let _guard = lock_plot_test_registry();
        reset_plot_state();
        let baseline = Figure::new()
            .retained_snapshot_bytes()
            .expect("bounded baseline");
        let mut figure = Figure::new();
        for _ in 0..64 {
            figure.add_line_plot(LinePlot::new(vec![0.0], vec![1.0]).expect("line"));
        }
        figure.clear();
        figure.set_name(String::with_capacity(4096));
        let retained = figure
            .retained_snapshot_bytes()
            .expect("bounded spare capacity");
        assert!(retained > baseline + 4096);

        let handle = import_figure(figure);
        let entry_bytes = std::mem::size_of::<FigureSnapshotEntry>();
        let baseline_budget = entry_bytes
            + figure_owner_allocation_upper_bound(baseline).expect("bounded baseline allocation");
        assert!(matches!(
            capture_figure_persistence_snapshot(&[handle], 1, baseline_budget),
            Err(FigurePersistenceSnapshotError::BudgetExceeded { .. })
        ));
    }
}
