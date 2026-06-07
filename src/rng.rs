//! Simple LCG PRNG.

#[derive(Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn gen_range(&mut self, r: std::ops::Range<f64>) -> f64 {
        r.start + self.gen_f64() * (r.end - r.start)
    }
    pub fn gen_range_usize(&mut self, r: std::ops::Range<usize>) -> usize {
        r.start + (self.gen_f64() * (r.end - r.start) as f64) as usize
    }
}
