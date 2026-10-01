use super::FillLines;

#[test]
fn trailing_singleton_axes_do_not_allocate_rank_sized_metadata() {
    let lines = FillLines::new(&[2, 3], Some(usize::MAX)).unwrap();
    let offsets: Vec<_> = lines
        .iter()
        .map(|line| (line.start, line.length, line.stride))
        .collect();
    assert_eq!(
        offsets,
        (0..6).map(|start| (start, 1, 6)).collect::<Vec<_>>()
    );
    assert_eq!(FillLines::new(&[2, 0, 3], None).unwrap().iter().count(), 0);
    assert!(FillLines::new(&[usize::MAX, 2], None).is_err());
    assert!(FillLines::new(&[2, 3], Some(0)).is_err());
}
