use super::*;
use conforming_split::ProtectedFacetSplit;

fn triangle(indices: [u8; 3]) -> DelaunayRecoveredFacetTriangle {
    DelaunayRecoveredFacetTriangle {
        node_identities: indices.map(|index| StableDigest::from_bytes([index; 32])),
    }
}
fn check(protection: &FacetProtection, expected: &[DelaunayRecoveredFacetTriangle]) {
    assert_eq!(protection.triangles, expected);
    let faces = expected
        .iter()
        .map(|triangle| {
            let mut face = triangle.node_identities;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(protection.faces, faces);
    assert_eq!(
        protection.ordered_faces(),
        faces.into_iter().collect::<Vec<_>>()
    );
}

#[test]
fn oriented_lineage_and_duplicate_split_children_keep_exact_protection() {
    let [forward, reverse, first, second, final_child] =
        [[1, 2, 3], [1, 3, 2], [1, 2, 4], [1, 3, 4], [1, 2, 5]].map(triangle);
    let mut protection = FacetProtection::default();
    for value in [forward, reverse, forward] {
        protection.insert(value);
    }
    check(&protection, &[forward, reverse]);
    protection.apply_splits(
        &mut [],
        vec![ProtectedFacetSplit {
            source: forward,
            children: vec![first, second],
        }],
    );
    check(&protection, &[first, second, reverse]);
    protection.apply_splits(
        &mut [],
        vec![ProtectedFacetSplit {
            source: reverse,
            children: vec![first],
        }],
    );
    check(&protection, &[first, second, first]);
    protection.apply_splits(
        &mut [],
        vec![ProtectedFacetSplit {
            source: first,
            children: vec![final_child],
        }],
    );
    check(&protection, &[final_child, second, first]);
    protection.insert(first);
    check(&protection, &[final_child, second, first]);
}
