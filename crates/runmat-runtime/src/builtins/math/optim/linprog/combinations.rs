//! Pull traversal preserves the solver's active-count/lexicographic order.

pub(super) struct Combinations {
    len: usize,
    current: Vec<usize>,
    ready: bool,
    finished: bool,
}

impl Combinations {
    pub(super) fn new(len: usize, choose: usize) -> Self {
        Self {
            len,
            current: if choose <= len {
                (0..choose).collect()
            } else {
                Vec::new()
            },
            ready: true,
            finished: choose > len,
        }
    }

    pub(super) fn next(&mut self) -> Option<&[usize]> {
        if self.finished {
            return None;
        }
        if self.ready {
            self.ready = false;
            return Some(&self.current);
        }
        let choose = self.current.len();
        let Some(index) = (0..choose)
            .rev()
            .find(|&index| self.current[index] < self.len - (choose - index))
        else {
            self.finished = true;
            return None;
        };
        // Each increment is bounded strictly below len by the search above.
        self.current[index] += 1;
        for next in index + 1..choose {
            self.current[next] = self.current[next - 1] + 1;
        }
        Some(&self.current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn pull_order_matches_original_recursion_and_checked_boundaries() {
        fn original(
            len: usize,
            choose: usize,
            start: usize,
            current: &mut Vec<usize>,
            output: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == choose {
                output.push(current.clone());
                return;
            }
            let remaining = choose - current.len();
            for index in start..=len - remaining {
                current.push(index);
                original(len, choose, index + 1, current, output);
                current.pop();
            }
        }
        for len in 0..=6 {
            for choose in 0..=len + 1 {
                let mut expected = Vec::new();
                if choose <= len {
                    original(len, choose, 0, &mut Vec::new(), &mut expected);
                }
                let mut source = Combinations::new(len, choose);
                let mut actual = Vec::new();
                while let Some(indices) = source.next() {
                    actual.push(indices.to_vec());
                }
                assert_eq!(actual, expected);
                assert!(source.next().is_none());
            }
        }
        let mut large = Combinations::new(usize::MAX, 1);
        assert_eq!(large.next(), Some(&[0][..]));
        assert_eq!(large.next(), Some(&[1][..]));
        let mut zero = Combinations::new(usize::MAX, 0);
        assert_eq!(zero.next(), Some(&[][..]));
        assert!(zero.next().is_none());
    }
}
