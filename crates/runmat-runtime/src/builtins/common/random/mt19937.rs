//! The standard MT19937 engine behind RunMat's host random stream.
//!
//! This is an independent Rust implementation of the published MT19937ar
//! recurrence, 2002 initialization, and 53-bit real conversion. The reference
//! is Matsumoto and Nishimura, <https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/emt19937ar.html>.
//! RunMat's seed-to-stream policy is owned by `random::state`.
//!
//! The authors' reference is BSD-3-Clause licensed. Its notice follows so
//! this module retains attribution if the implementation is treated as derived:
//!
//! Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura.
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//! 1. Redistributions of source code must retain the above copyright notice,
//!    this list of conditions and the following disclaimer.
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//! 3. The names of its contributors may not be used to endorse or promote
//!    products derived from this software without specific prior written
//!    permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
//! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//! CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//! SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//! INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//! CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//! POSSIBILITY OF SUCH DAMAGE.

use std::fmt;

pub const STATE_WORDS: usize = 624;
const OFFSET: usize = 397;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;
const TWIST: u32 = 0x9908_b0df;

/// One complete engine state. `index == 624` means that the next word twists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mt19937State {
    words: [u32; STATE_WORDS],
    index: usize,
}

impl Mt19937State {
    pub fn words(&self) -> &[u32; STATE_WORDS] {
        &self.words
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn validated(words: [u32; STATE_WORDS], index: usize) -> Result<Self, StateError> {
        if index > STATE_WORDS {
            return Err(StateError::InvalidIndex);
        }
        if words.iter().all(|word| *word == 0) {
            return Err(StateError::ZeroState);
        }
        Ok(Self { words, index })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateError {
    InvalidIndex,
    ZeroState,
    EmptySeedWords,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIndex => write!(f, "MT19937 index must be in 0..=624"),
            Self::ZeroState => write!(f, "MT19937 state cannot be all zero"),
            Self::EmptySeedWords => write!(f, "MT19937 seed words cannot be empty"),
        }
    }
}

impl std::error::Error for StateError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mt19937 {
    state: Mt19937State,
}

impl Mt19937 {
    /// The reference single-word initialization.
    pub fn seed(seed: u32) -> Self {
        let mut words = [0_u32; STATE_WORDS];
        words[0] = seed;
        for index in 1..STATE_WORDS {
            let previous = words[index - 1];
            words[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state: Mt19937State {
                words,
                index: STATE_WORDS,
            },
        }
    }

    /// The reference array initialization, useful for explicit word seeds.
    /// RunMat's seed selection policy is separate from this recurrence.
    pub fn seed_words(seed: &[u32]) -> Result<Self, StateError> {
        if seed.is_empty() {
            return Err(StateError::EmptySeedWords);
        }
        let mut engine = Self::seed(19_650_218);
        let words = &mut engine.state.words;
        let mut position = 1;
        let mut seed_position = 0;
        for _ in 0..STATE_WORDS.max(seed.len()) {
            let previous = words[position - 1];
            words[position] = (words[position]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(seed[seed_position])
            .wrapping_add(seed_position as u32);
            position += 1;
            seed_position += 1;
            if position == STATE_WORDS {
                words[0] = words[STATE_WORDS - 1];
                position = 1;
            }
            if seed_position == seed.len() {
                seed_position = 0;
            }
        }
        for _ in 1..STATE_WORDS {
            let previous = words[position - 1];
            words[position] = (words[position]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(position as u32);
            position += 1;
            if position == STATE_WORDS {
                words[0] = words[STATE_WORDS - 1];
                position = 1;
            }
        }
        words[0] = UPPER;
        Ok(engine)
    }

    pub fn from_state(state: Mt19937State) -> Self {
        Self { state }
    }

    pub fn snapshot(&self) -> Mt19937State {
        self.state.clone()
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.state.index == STATE_WORDS {
            self.twist();
        }
        let mut word = self.state.words[self.state.index];
        self.state.index += 1;
        word ^= word >> 11;
        word ^= (word << 7) & 0x9d2c_5680;
        word ^= (word << 15) & 0xefc6_0000;
        word ^= word >> 18;
        word
    }

    /// Consume unobserved source words without tempering them. A reservation
    /// uses this to advance the shared stream before cooperative work starts.
    pub fn advance_words(&mut self, mut words: u64) {
        while words > 0 {
            if self.state.index == STATE_WORDS {
                self.twist();
            }
            let available = (STATE_WORDS - self.state.index) as u64;
            let step = available.min(words);
            self.state.index += step as usize;
            words -= step;
        }
    }

    /// The authors' 53-bit [0,1) conversion consumes exactly two output words.
    pub fn next_f64(&mut self) -> f64 {
        let high = self.next_u32() >> 5;
        let low = self.next_u32() >> 6;
        ((high as f64) * 67_108_864.0 + (low as f64)) / 9_007_199_254_740_992.0
    }

    fn twist(&mut self) {
        for index in 0..STATE_WORDS {
            let next = (index + 1) % STATE_WORDS;
            let distant = (index + OFFSET) % STATE_WORDS;
            let joined = (self.state.words[index] & UPPER) | (self.state.words[next] & LOWER);
            let parity = if (joined & 1) == 0 { 0 } else { TWIST };
            self.state.words[index] = self.state.words[distant] ^ (joined >> 1) ^ parity;
        }
        self.state.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_single_seed_words_and_53_bit_conversion() {
        let mut engine = Mt19937::seed(5489);
        for expected in [
            3_499_211_612,
            581_869_302,
            3_890_346_734,
            3_586_334_585,
            545_404_204,
            4_161_255_391,
        ] {
            assert_eq!(engine.next_u32(), expected);
        }
        let mut real = Mt19937::seed(5489);
        assert_eq!(
            real.next_f64().to_bits(),
            0.814_723_686_393_178_9_f64.to_bits()
        );
        assert_eq!(real.snapshot().index(), 2);
        assert_eq!(real.next_u32(), 3_890_346_734);
    }

    #[test]
    fn published_array_seed_words_and_state_roundtrip_across_twist() {
        let mut engine = Mt19937::seed_words(&[0x123, 0x234, 0x345, 0x456]).unwrap();
        for expected in [
            1_067_595_299,
            955_945_823,
            477_289_528,
            4_107_218_783,
            4_228_976_476,
        ] {
            assert_eq!(engine.next_u32(), expected);
        }
        for _ in 5..STATE_WORDS - 1 {
            engine.next_u32();
        }
        let state = engine.snapshot();
        assert_eq!(state.index(), STATE_WORDS - 1);
        let mut restored = Mt19937::from_state(state);
        for _ in 0..4 {
            assert_eq!(engine.next_u32(), restored.next_u32());
        }
    }

    #[test]
    fn imported_state_validates_index_and_nonzero_words() {
        assert_eq!(
            Mt19937State::validated([1; STATE_WORDS], STATE_WORDS + 1),
            Err(StateError::InvalidIndex)
        );
        assert_eq!(
            Mt19937State::validated([0; STATE_WORDS], 0),
            Err(StateError::ZeroState)
        );
        assert_eq!(Mt19937::seed_words(&[]), Err(StateError::EmptySeedWords));
        let seed = Mt19937::seed(5489);
        assert_eq!(
            Mt19937State::validated(*seed.snapshot().words(), STATE_WORDS).unwrap(),
            seed.snapshot()
        );
    }
}
