pub trait Entropy {
    fn fill(&mut self, output: &mut [u8]);
}

#[derive(Clone, Debug)]
pub struct DeterministicEntropy {
    state: u64,
}

impl DeterministicEntropy {
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Entropy for DeterministicEntropy {
    fn fill(&mut self, output: &mut [u8]) {
        for byte in output {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            *byte = self.state as u8;
        }
    }
}
