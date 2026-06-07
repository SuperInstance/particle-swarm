//! Swarm topologies: ring, star, von Neumann.

use crate::particle::Particle;

/// Trait for swarm topology that determines neighborhood.
pub trait Topology {
    /// Get the indices of neighbors for particle `i`.
    fn neighbors(&self, i: usize, swarm_size: usize) -> Vec<usize>;
    /// Get the best particle in the neighborhood.
    fn neighborhood_best(&self, swarm: &[Particle], i: usize) -> Vec<f64> {
        let neighbors = self.neighbors(i, swarm.len());
        let best_idx = *neighbors.iter().min_by(|&&a, &&b| {
            swarm[a].best_fitness.partial_cmp(&swarm[b].best_fitness).unwrap()
        }).unwrap();
        swarm[best_idx].best_position.clone()
    }
}

/// Star (global) topology: every particle is neighbor to every other.
#[derive(Clone)]
pub struct StarTopology;

impl Topology for StarTopology {
    fn neighbors(&self, _i: usize, swarm_size: usize) -> Vec<usize> {
        (0..swarm_size).collect()
    }
}

/// Ring topology: each particle has K nearest neighbors.
#[derive(Clone)]
pub struct RingTopology {
    pub k: usize,
}

impl Topology for RingTopology {
    fn neighbors(&self, i: usize, swarm_size: usize) -> Vec<usize> {
        let half = self.k / 2;
        let mut neighbors = Vec::new();
        for d in 1..=half {
            neighbors.push((i + d) % swarm_size);
            neighbors.push((i + swarm_size - d) % swarm_size);
        }
        neighbors.push(i);
        neighbors
    }
}

/// Von Neumann topology: 2D grid neighborhood.
#[derive(Clone)]
pub struct VonNeumannTopology {
    pub rows: usize,
    pub cols: usize,
}

impl VonNeumannTopology {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    fn grid_pos(&self, i: usize) -> (usize, usize) {
        (i / self.cols, i % self.cols)
    }

    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }
}

impl Topology for VonNeumannTopology {
    fn neighbors(&self, i: usize, swarm_size: usize) -> Vec<usize> {
        let (r, c) = self.grid_pos(i);
        let mut neighbors = vec![i];
        // Up
        if r > 0 { neighbors.push(self.idx(r - 1, c)); }
        // Down
        let next_r = r + 1;
        if next_r < self.rows && self.idx(next_r, c) < swarm_size { neighbors.push(self.idx(next_r, c)); }
        // Left
        if c > 0 { neighbors.push(self.idx(r, c - 1)); }
        // Right
        let next_c = c + 1;
        if next_c < self.cols && self.idx(r, next_c) < swarm_size { neighbors.push(self.idx(r, next_c)); }
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SimpleRng;

    #[test]
    fn star_all_neighbors() {
        let star = StarTopology;
        let n = star.neighbors(2, 5);
        assert_eq!(n.len(), 5);
    }

    #[test]
    fn ring_correct_count() {
        let ring = RingTopology { k: 2 };
        let n = ring.neighbors(0, 10);
        assert_eq!(n.len(), 3); // self + 2
    }

    #[test]
    fn ring_wraps_around() {
        let ring = RingTopology { k: 2 };
        let n = ring.neighbors(0, 10);
        assert!(n.contains(&9)); // wraps
        assert!(n.contains(&1));
    }

    #[test]
    fn von_neumann_correct() {
        let vn = VonNeumannTopology::new(3, 3);
        let n = vn.neighbors(4, 9); // center
        assert_eq!(n.len(), 5); // self + 4
    }

    #[test]
    fn von_neumann_corner() {
        let vn = VonNeumannTopology::new(3, 3);
        let n = vn.neighbors(0, 9); // top-left
        assert_eq!(n.len(), 3); // self + right + down
    }

    #[test]
    fn neighborhood_best_finds_best() {
        let mut rng = SimpleRng::new(42);
        let mut swarm: Vec<Particle> = (0..5).map(|_| Particle::random(2, 0.0, 1.0, &mut rng)).collect();
        swarm[0].best_fitness = 10.0;
        swarm[0].best_position = vec![0.0, 0.0];
        swarm[1].best_fitness = 5.0;
        swarm[1].best_position = vec![1.0, 1.0];
        swarm[2].best_fitness = 8.0;
        swarm[3].best_fitness = 3.0;
        swarm[3].best_position = vec![0.5, 0.5]; // best fitness
        swarm[4].best_fitness = 7.0;

        let star = StarTopology;
        let nb = star.neighborhood_best(&swarm, 0);
        assert_eq!(nb, vec![0.5, 0.5]); // index 3 has best (lowest) fitness
    }
}
