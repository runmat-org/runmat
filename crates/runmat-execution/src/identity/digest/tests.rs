use super::Digest;

#[test]
fn cooperative_hash_preserves_the_original_byte_stream() {
    let large = vec![17; 3 * 1024 * 1024 + 11];
    let parts = [b"domain\0".as_slice(), &[], large.as_slice(), b"tail"];
    let mut probes = 0;
    let actual = Digest::sha256_parts_with_checkpoint(parts, || {
        probes += 1;
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(actual, Digest::sha256_parts(parts));
    assert_eq!(actual, Digest::sha256(parts.concat()));
    assert_eq!(probes, 11);
    assert_eq!(
        Digest::sha256_parts_with_checkpoint([&[] as &[u8]], || Ok::<_, ()>(())).unwrap(),
        Digest::sha256([])
    );
}

#[test]
fn cooperative_hash_stops_before_visiting_later_parts() {
    let large = vec![1; 4 * 1024 * 1024];
    let mut parts_visited = 0;
    let mut probes = 0;
    let parts = [large.as_slice(), b"not visited"]
        .into_iter()
        .inspect(|_| parts_visited += 1);
    let result = Digest::sha256_parts_with_checkpoint(parts, || {
        probes += 1;
        if probes == 3 {
            Err("cancelled")
        } else {
            Ok(())
        }
    });
    assert_eq!(result, Err("cancelled"));
    assert_eq!(parts_visited, 1);
    assert_eq!(probes, 3);
}

#[test]
fn segmented_hash_is_boundary_independent_and_order_sensitive() {
    let bytes = (0_u8..=255).collect::<Vec<_>>();
    for width in [1, 3, 64, 255, 256, 257] {
        assert_eq!(
            Digest::sha256_parts(bytes.chunks(width)),
            Digest::sha256(&bytes)
        );
    }
    assert_eq!(
        Digest::sha256_parts([b"".as_slice(), b"abc", b""]),
        Digest::sha256(b"abc")
    );
    assert_eq!(
        Digest::sha256_parts(std::iter::empty::<&[u8]>()),
        Digest::sha256(b"")
    );
    assert_ne!(
        Digest::sha256_parts([b"abc", b"def"]),
        Digest::sha256_parts([b"def", b"abc"])
    );
}
