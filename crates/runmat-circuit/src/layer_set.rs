//! Copper layer indices and sets of them.
//!
//! A board's stack-up sets its copper layer count. Indices are [`CopperIndex`] (0 is the top
//! layer), wide enough for any stack-up that fits in memory. [`LayerSet`] holds any subset of a
//! stack's layers: the first 64 layers sit in one inline word, so a set on a typical board
//! never allocates, and deeper layers go in further words.

/// A copper layer's position in the stack-up, top first (0 is the top layer).
pub type CopperIndex = u32;

const WORD: CopperIndex = 64;

/// A set of copper layers of any stack depth.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerSet {
    /// Layers 0–63.
    low: u64,
    /// Layers 64 and up, 64 per word; no trailing zero words.
    high: Vec<u64>,
}

impl LayerSet {
    pub fn new() -> LayerSet {
        LayerSet::default()
    }

    /// The set holding `layer` alone.
    pub fn single(layer: CopperIndex) -> LayerSet {
        let mut s = LayerSet::new();
        s.insert(layer);
        s
    }

    /// The layers `from..=to`.
    pub fn span(from: CopperIndex, to: CopperIndex) -> LayerSet {
        (from..=to).collect()
    }

    pub fn insert(&mut self, layer: CopperIndex) {
        let (w, bit) = (layer / WORD, layer % WORD);
        if w == 0 {
            self.low |= 1 << bit;
            return;
        }
        let k = (w - 1) as usize;
        if self.high.len() <= k {
            self.high.resize(k + 1, 0);
        }
        self.high[k] |= 1 << bit;
    }

    pub fn contains(&self, layer: CopperIndex) -> bool {
        let (w, bit) = (layer / WORD, layer % WORD);
        let word = if w == 0 {
            self.low
        } else {
            self.high.get((w - 1) as usize).copied().unwrap_or(0)
        };
        word & (1 << bit) != 0
    }

    /// Whether the two sets share a layer.
    pub fn intersects(&self, other: &LayerSet) -> bool {
        self.low & other.low != 0 || self.high.iter().zip(&other.high).any(|(a, b)| a & b != 0)
    }

    /// The top-most layer the two sets share.
    pub fn first_common(&self, other: &LayerSet) -> Option<CopperIndex> {
        if self.low & other.low != 0 {
            return Some((self.low & other.low).trailing_zeros());
        }
        self.high
            .iter()
            .zip(&other.high)
            .enumerate()
            .find_map(|(k, (a, b))| {
                (a & b != 0).then(|| (k as CopperIndex + 1) * WORD + (a & b).trailing_zeros())
            })
    }

    /// Adds every layer of `other`.
    pub fn union_with(&mut self, other: &LayerSet) {
        self.low |= other.low;
        if self.high.len() < other.high.len() {
            self.high.resize(other.high.len(), 0);
        }
        for (a, b) in self.high.iter_mut().zip(&other.high) {
            *a |= b;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.low == 0 && self.high.is_empty()
    }

    /// The top-most layer of the set.
    pub fn first(&self) -> Option<CopperIndex> {
        self.iter().next()
    }

    /// The layers, top first.
    pub fn iter(&self) -> impl Iterator<Item = CopperIndex> + '_ {
        std::iter::once(self.low)
            .chain(self.high.iter().copied())
            .enumerate()
            .flat_map(|(w, word)| {
                (0..WORD)
                    .filter(move |b| word & (1 << b) != 0)
                    .map(move |b| w as CopperIndex * WORD + b)
            })
    }
}

impl FromIterator<CopperIndex> for LayerSet {
    fn from_iter<I: IntoIterator<Item = CopperIndex>>(iter: I) -> LayerSet {
        let mut s = LayerSet::new();
        for l in iter {
            s.insert(l);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_hold_layers_past_the_first_word() {
        let deep: LayerSet = [0, 39, 63, 64, 130].into_iter().collect();
        assert_eq!(deep.iter().collect::<Vec<_>>(), vec![0, 39, 63, 64, 130]);
        assert!(deep.contains(130) && !deep.contains(129) && !deep.contains(1000));
        assert!(deep.intersects(&LayerSet::single(64)));
        assert!(!deep.intersects(&LayerSet::span(65, 129)));
        let mut u = LayerSet::single(5);
        u.union_with(&deep);
        assert_eq!(u.first(), Some(0));
        assert_eq!(u.iter().count(), 6);
        assert!(LayerSet::new().is_empty() && LayerSet::single(70).first() == Some(70));
    }
}
