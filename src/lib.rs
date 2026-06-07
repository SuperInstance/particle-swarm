//! # Particle Swarm Optimization
//!
//! A PSO library with velocity update rules, inertia weight strategies,
//! topology options (ring, star, von Neumann), and multi-objective PSO
//! with Pareto front tracking. Zero external dependencies.
//!
//! # Example
//! ```
//! use particle_swarm::{SimpleRng, PSO};
//!
//! let mut rng = SimpleRng::new(42);
//! let result = PSO::new(30, 2, -10.0, 10.0)
//!     .run(&mut rng, |x| x[0]*x[0] + x[1]*x[1], 200, 0.7, 1.5, 1.5);
//! println!("Best: {:?}, Fitness: {}", result.0, result.1);
//! ```

pub mod particle;
pub mod velocity;
pub mod topology;
pub mod multi_objective;
pub mod inertia;
mod rng;

pub use rng::SimpleRng;
pub use particle::Particle;
pub use topology::{Topology, StarTopology, RingTopology, VonNeumannTopology};
pub use velocity::{update_velocity, update_position, InertiaWeight, ConstantInertia, LinearInertia};
pub use multi_objective::{MultiObjectivePSO, ParetoFront, ObjectiveVec};

/// Standard PSO runner.
pub struct PSO {
    swarm_size: usize,
    dim: usize,
    min: f64,
    max: f64,
}

impl PSO {
    pub fn new(swarm_size: usize, dim: usize, min: f64, max: f64) -> Self {
        Self { swarm_size, dim, min, max }
    }

    /// Run PSO optimization. Returns (best_position, best_fitness).
    pub fn run(
        &self,
        rng: &mut SimpleRng,
        fitness_fn: impl Fn(&[f64]) -> f64,
        iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
    ) -> (Vec<f64>, f64) {
        let mut swarm: Vec<Particle> = (0..self.swarm_size)
            .map(|_| Particle::random(self.dim, self.min, self.max, rng))
            .collect();

        // Evaluate initial
        for p in &mut swarm {
            p.fitness = fitness_fn(&p.position);
            p.update_best();
        }

        let mut global_best_pos = swarm[0].best_position.clone();
        let mut global_best_fitness = swarm[0].best_fitness;

        for p in &swarm {
            if p.best_fitness < global_best_fitness {
                global_best_fitness = p.best_fitness;
                global_best_pos = p.best_position.clone();
            }
        }

        for _ in 0..iterations {
            for p in &mut swarm {
                update_velocity(p, &global_best_pos, w, c1, c2, rng);
                update_position(p);
                p.clamp_position(self.min, self.max);
                let max_vel = (self.max - self.min) * 0.5;
                p.clamp_velocity(max_vel);

                p.fitness = fitness_fn(&p.position);
                p.update_best();

                if p.best_fitness < global_best_fitness {
                    global_best_fitness = p.best_fitness;
                    global_best_pos = p.best_position.clone();
                }
            }
        }

        (global_best_pos, global_best_fitness)
    }

    /// Run PSO with a custom topology.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_topology<T: Topology>(
        &self,
        rng: &mut SimpleRng,
        fitness_fn: impl Fn(&[f64]) -> f64,
        iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
        topology: &T,
    ) -> (Vec<f64>, f64) {
        let mut swarm: Vec<Particle> = (0..self.swarm_size)
            .map(|_| Particle::random(self.dim, self.min, self.max, rng))
            .collect();

        for p in &mut swarm {
            p.fitness = fitness_fn(&p.position);
            p.update_best();
        }

        let mut global_best_fitness = swarm.iter().map(|p| p.best_fitness).fold(f64::MAX, f64::min);
        let mut global_best_pos = swarm.iter()
            .find(|p| p.best_fitness == global_best_fitness)
            .unwrap()
            .best_position
            .clone();

        for _ in 0..iterations {
            for i in 0..swarm.len() {
                let local_best = topology.neighborhood_best(&swarm, i);
                update_velocity(&mut swarm[i], &local_best, w, c1, c2, rng);
                update_position(&mut swarm[i]);
                swarm[i].clamp_position(self.min, self.max);

                swarm[i].fitness = fitness_fn(&swarm[i].position);
                swarm[i].update_best();

                if swarm[i].best_fitness < global_best_fitness {
                    global_best_fitness = swarm[i].best_fitness;
                    global_best_pos = swarm[i].best_position.clone();
                }
            }
        }

        (global_best_pos, global_best_fitness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pso_finds_minimum() {
        let mut rng = SimpleRng::new(42);
        let pso = PSO::new(30, 2, -10.0, 10.0);
        let (pos, fitness) = pso.run(&mut rng, |x| x[0]*x[0] + x[1]*x[1], 200, 0.7, 1.5, 1.5);
        assert!(fitness < 5.0, "should find near zero, got {}", fitness);
    }

    #[test]
    fn pso_with_ring_topology() {
        let mut rng = SimpleRng::new(42);
        let pso = PSO::new(20, 2, -5.0, 5.0);
        let ring = RingTopology { k: 2 };
        let (_pos, fitness) = pso.run_with_topology(
            &mut rng, |x| (x[0]-3.0).powi(2) + (x[1]-2.0).powi(2),
            200, 0.7, 1.5, 1.5, &ring,
        );
        assert!(fitness < 5.0, "got {}", fitness);
    }

    #[test]
    fn pso_with_von_neumann() {
        let mut rng = SimpleRng::new(42);
        let pso = PSO::new(9, 2, -5.0, 5.0);
        let vn = VonNeumannTopology::new(3, 3);
        let (_pos, fitness) = pso.run_with_topology(
            &mut rng, |x| x[0]*x[0] + x[1]*x[1],
            100, 0.7, 1.5, 1.5, &vn,
        );
        assert!(fitness < 10.0);
    }
}
