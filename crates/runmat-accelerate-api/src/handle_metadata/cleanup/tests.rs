use super::*;

#[derive(Default)]
struct Annotations {
    logical: RwLock<HashSet<GpuHandleIdentity>>,
    hits: RwLock<HashMap<GpuHandleIdentity, u64>>,
    classes: RwLock<HashMap<GpuHandleIdentity, ClassIdentity>>,
    transpose: RwLock<HashMap<GpuHandleIdentity, TransposeInfo>>,
}

impl Annotations {
    fn locks(&self) -> AnnotationLocks<'_> {
        AnnotationLocks {
            logical: &self.logical,
            hits: &self.hits,
            classes: &self.classes,
            transpose: &self.transpose,
        }
    }

    fn populate(&self, identity: GpuHandleIdentity) {
        self.logical.write().unwrap().insert(identity);
        self.hits.write().unwrap().insert(identity, 7);
        self.classes
            .write()
            .unwrap()
            .insert(identity, standard::LOGICAL.owned());
        self.transpose.write().unwrap().insert(
            identity,
            TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            },
        );
    }
}

#[test]
fn checked_annotations_remove_only_exact_identity() {
    let annotations = Annotations::default();
    annotations.populate((1, 9));
    annotations.populate((2, 9));
    annotations.locks().clear((1, 9)).unwrap();
    assert_eq!(annotations.logical.read().unwrap().len(), 1);
    assert!(annotations.logical.read().unwrap().contains(&(2, 9)));
    assert_eq!(
        *annotations.hits.read().unwrap(),
        HashMap::from([((2, 9), 7)])
    );
    assert_eq!(annotations.classes.read().unwrap().len(), 1);
    assert_eq!(annotations.transpose.read().unwrap().len(), 1);
    annotations.locks().clear((1, 9)).unwrap();
}

#[cfg(panic = "unwind")]
#[test]
fn checked_annotation_poison_preserves_all_entries_before_mutation() {
    fn poison<T>(lock: &RwLock<T>) {
        assert!(catch_allocation_cleanup_unwind(|| {
            let _guard = lock.write().unwrap();
            panic!("injected annotation lock failure");
        })
        .is_err());
    }

    for target in [
        HandleAnnotationLock::Logical,
        HandleAnnotationLock::LogicalHits,
        HandleAnnotationLock::Class,
        HandleAnnotationLock::Transpose,
    ] {
        let annotations = Annotations::default();
        annotations.populate((1, 9));
        match target {
            HandleAnnotationLock::Logical => poison(&annotations.logical),
            HandleAnnotationLock::LogicalHits => poison(&annotations.hits),
            HandleAnnotationLock::Class => poison(&annotations.classes),
            HandleAnnotationLock::Transpose => poison(&annotations.transpose),
        }
        assert_eq!(
            annotations.locks().clear((1, 9)),
            Err(HandleAnnotationLockPoisoned(target))
        );
        // Inspection alone recovers poisoned guards; production cleanup never
        // resumes mutation through a poisoned annotation lock.
        assert!(annotations
            .logical
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(1, 9)));
        assert_eq!(
            annotations
                .hits
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get(&(1, 9)),
            Some(&7)
        );
        assert!(annotations
            .classes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&(1, 9)));
        assert!(annotations
            .transpose
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&(1, 9)));
    }
}

#[test]
fn checked_annotation_cleanup_preserves_physical_descriptor() {
    let handle = GpuTensorHandle::new(vec![2, 3], u32::MAX - 17, u64::MAX - 19)
        .with_numeric_descriptor(
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        )
        .with_provenance(GpuHandleProvenance::Explicit);
    let original = handle.descriptor;
    set_handle_logical(&handle, true);
    record_handle_transpose(&handle, 3, 2);
    clear_handle_annotations_checked(&handle).unwrap();
    assert_eq!(handle.descriptor, original);
    assert!(!handle_is_logical(&handle));
    assert_eq!(handle_logical_hits_for_handle(&handle), None);
    assert_eq!(handle_transpose_info(&handle), None);
    assert_eq!(
        handle_class_identity(&handle),
        Some(standard::SINGLE.owned())
    );
}

#[cfg(panic = "unwind")]
#[test]
fn checked_residency_callback_is_separate_and_can_reenter_annotations() {
    fn callback(handle: &GpuTensorHandle) {
        set_handle_logical(handle, true);
        assert!(handle_is_logical(handle));
        panic!("callback failed after mutation");
    }
    let handle = GpuTensorHandle::new(vec![1], u32::MAX - 17, u64::MAX - 20);
    assert_eq!(clear_residency_with(&handle, None), Ok(()));
    assert_eq!(
        clear_residency_with(&handle, Some(callback)),
        Err(AllocationCleanupPanicked)
    );
    // Failure is not a claim that callback effects were rolled back. The owner
    // must remember this phase rather than invoking the callback again.
    assert!(handle_is_logical(&handle));
    clear_handle_annotations_checked(&handle).unwrap();
    assert!(!handle_is_logical(&handle));
}
