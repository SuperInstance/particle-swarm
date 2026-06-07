//! Particle representation and management.

use crate::rng::SimpleRng;

/// A single particle in the swarm.
#[derive(Clone, Debug)]
pub struct Particle {
    /// Current position in search space.
    pub position: Vec<f64>,
    /// Current velocity.
    pub velocity: Vec<f64>,
    /// Best position found by this particle.
    pub best_position: Vec<f64>,
    /// Fitness at best position.
    pub best_fitness: f64,
    /// Current fitness.
    pub fitness: f64,
}

impl Particle {
    /// Create a new particle with random position and zero velocity.
    pub fn random(dim: usize, min: f64, max: f64, rng: &mut SimpleRng) -> Self {
        let position: Vec<f64> = (0..dim).map(|_| rng.gen_range(min..max)).collect();
        let velocity = vec![0.0; dim];
        Self {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: f64::MAX,
            fitness: f64::MAX,
        }
    }

    /// Update the particle's personal best if current position is better.
    pub fn update_best(&mut self) {
        if self.fitness < self.best_fitness {
            self.best_position = self.position.clone();
            self.best_fitness = self.fitness;
        }
    }

    /// Clamp velocity to range.
    pub fn clamp_velocity(&mut self, max_vel: f64) {
        for v in &mut self.velocity {
            *v = v.clamp(-max_vel, max_vel);
        }
    }

    /// Clamp position to bounds.
    pub fn clamp_position(&mut self, min: f64, max: f64) {
        for p in &mut self.position {
            *p = p.clamp(min, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_particle_has_correct_dim() {
        let mut rng = SimpleRng::new(42);
        let p = Particle::random(5, -1.0, 1.0, &mut rng);
        assert_eq!(p.position.len(), 5);
        assert_eq!(p.velocity.len(), 5);
    }

    #[test]
    fn random_particle_position_in_bounds() {
        let mut rng = SimpleRng::new(42);
        let p = Particle::random(100, -5.0, 5.0, &mut rng);
        for v in &p.position {
            assert!(*v >= -5.0 && *v <= 5.0);
        }
    }

    #[test]
    fn update_best_sets_on_improvement() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        p.fitness = 5.0;
        p.update_best();
        assert_eq!(p.best_fitness, 5.0);
        p.fitness = 3.0;
        p.update_best();
        assert_eq!(p.best_fitness, 3.0);
    }

    #[test]
    fn update_best_does_not_set_on_worse() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        p.fitness = 3.0;
        p.update_best();
        p.fitness = 10.0;
        p.update_best();
        assert_eq!(p.best_fitness, 3.0);
    }

    #[test]
    fn clamp_velocity_works() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        p.velocity = vec![10.0, -10.0, 0.5];
        p.clamp_velocity(5.0);
        assert_eq!(p.velocity, vec![5.0, -5.0, 0.5]);
    }

    #[test]
    fn clamp_position_works() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        p.position = vec![10.0, -10.0, 3.0];
        p.clamp_position(-5.0, 5.0);
        assert_eq!(p.position, vec![5.0, -5.0, 3.0]);
    }
}
