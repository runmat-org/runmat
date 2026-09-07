pub(super) fn vector(input: &[usize], len: usize) -> Vec<usize> {
    if input.contains(&0) {
        if input.first() == Some(&1) {
            return vec![1, 0];
        }
        if input.get(1) == Some(&1) {
            return vec![0, 1];
        }
        return vec![0, 0];
    }
    if input.first() == Some(&1) || input.iter().all(|dimension| *dimension == 1) {
        vec![1, len]
    } else {
        vec![len, 1]
    }
}
