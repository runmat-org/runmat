use super::*;
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::copy_verified_bounded_with_admission, LogicalObject, ObjectNamespace,
};
use std::{cell::RefCell, collections::BTreeMap};

struct BoundedOnly {
    objects: BTreeMap<Digest, Vec<u8>>,
    reads: RefCell<Vec<(Digest, u64)>>,
}

impl CacheImport for BoundedOnly {
    fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        panic!("study replay must not request an unbounded provider read")
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded_with_admission(digest, maximum, &mut |_| Ok(()))
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.reads.borrow_mut().push((digest, maximum));
        copy_verified_bounded_with_admission(
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            maximum,
            admit,
        )
    }
}

#[test]
fn caller_limits_and_closed_membership_bound_verified_reads_and_admission() {
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "state",
        "application/octet-stream",
        b"data".to_vec(),
    )
    .unwrap();
    let access = ArtifactAccess {
        authorization_scope: "study-import".into(),
        encryption_context: Digest::sha256("key"),
    };
    let reference = access
        .reference(&object, ValueRefKind::ResultObject, "test/v1")
        .unwrap();
    let source = BoundedOnly {
        objects: BTreeMap::from([(object.descriptor.digest, object.bytes.clone())]),
        reads: RefCell::new(Vec::new()),
    };
    let limits = ObjectInventoryLimits {
        max_objects: 2,
        max_object_bytes: 4,
        max_total_bytes: 8,
    };
    let closed = CallerBoundedImport::new(&source, limits)
        .unwrap()
        .close(std::slice::from_ref(&reference), &access)
        .unwrap();
    assert_eq!(
        closed
            .read_verified(reference.logical_digest)
            .unwrap()
            .unwrap(),
        object.bytes
    );
    assert_eq!(
        source.reads.borrow().last().unwrap().1,
        reference.encoded_length
    );
    assert!(closed
        .read_verified_bounded(reference.logical_digest, 3)
        .is_err());
    assert_eq!(source.reads.borrow().last().unwrap().1, 3);
    let reads = source.reads.borrow().len();
    assert!(closed.read_verified(Digest::sha256("unlisted")).is_err());
    assert_eq!(source.reads.borrow().len(), reads);
    for smaller in [
        ObjectInventoryLimits {
            max_total_bytes: 3,
            ..limits
        },
        ObjectInventoryLimits {
            max_object_bytes: 3,
            ..limits
        },
    ] {
        assert!(CallerBoundedImport::new(&source, smaller)
            .unwrap()
            .close(std::slice::from_ref(&reference), &access)
            .is_err());
    }
    let mut foreign = access.clone();
    foreign.authorization_scope.push_str("-foreign");
    assert!(CallerBoundedImport::new(&source, limits)
        .unwrap()
        .close(std::slice::from_ref(&reference), &foreign)
        .is_err());
    assert_eq!(source.reads.borrow().len(), reads);
    let mut admitted = Vec::new();
    assert!(closed
        .read_verified_bounded_with_admission(reference.logical_digest, 4, &mut |bytes| {
            admitted.push(bytes);
            Err(ArtifactError::Limit("declined before buffer copy".into()))
        })
        .is_err());
    assert_eq!(admitted, vec![4]);
    struct Unbounded;
    impl CacheImport for Unbounded {
        fn read_verified(&self, _: Digest) -> ArtifactResult<Option<Vec<u8>>> {
            panic!("bounded import must not fall back to an unbounded provider")
        }
    }
    assert!(CallerBoundedImport::new(&Unbounded, limits)
        .unwrap()
        .read_verified(reference.logical_digest)
        .is_err());
}
