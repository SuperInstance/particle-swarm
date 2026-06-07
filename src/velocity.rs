//! Velocity update rules and inertia weight strategies.

use crate::particle::Particle;
use crate::rng::SimpleRng;

/// Inertia weight strategy.
pub trait InertiaWeight {
    fn weight(&self, iteration: usize, max_iterations: usize) -> f64;
}

/// Constant inertia weight.
#[derive(Clone)]
pub struct ConstantInertia {
    pub w: f64,
}

impl InertiaWeight for ConstantInertia {
    fn weight(&self, _iteration: usize, _max_iterations: usize) -> f64 {
        self.w
    }
}

/// Linearly decreasing inertia weight.
#[derive(Clone)]
pub struct LinearInertia {
    pub w_start: f64,
    pub w_end: f64,
}

impl InertiaWeight for LinearInertia {
    fn weight(&self, iteration: usize, max_iterations: usize) -> f64 {
        if max_iterations == 0 {
            return self.w_start;
        }
        let t = iteration as f64 / max_iterations as f64;
        self.w_start + (self.w_end - self.w_start) * t
    }
}

/// Update particle velocity using standard PSO rule.
pub fn update_velocity(
    particle: &mut Particle,
    global_best: &[f64],
    w: f64,
    c1: f64,
    c2: f64,
    rng: &mut SimpleRng,
) {
    let dim = particle.position.len();
    for (i, gb_val) in global_best.iter().enumerate().take(dim) {
        let r1 = rng.gen_f64();
        let r2 = rng.gen_f64();
        let cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i]);
        let social = c2 * r2 * (gb_val - particle.position[i]);
        particle.velocity[i] = w * particle.velocity[i] + cognitive + social;
    }
}

/// Update particle position based on velocity.
pub fn update_position(particle: &mut Particle) {
    for i in 0..particle.position.len() {
        particle.position[i] += particle.velocity[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_inertia_always_same() {
        let ci = ConstantInertia { w: 0.7 };
        assert_eq!(ci.weight(0, 100), 0.7);
        assert_eq!(ci.weight(50, 100), 0.7);
    }

    #[test]
    fn linear_inertia_decreases() {
        let li = LinearInertia { w_start: 0.9, w_end: 0.4 };
        assert!((li.weight(0, 100) - 0.9).abs() < 1e-10);
        assert!((li.weight(100, 100) - 0.4).abs() < 1e-10);
        assert!(li.weight(50, 100) < 0.9);
        assert!(li.weight(50, 100) > 0.4);
    }

    #[test]
    fn velocity_update_changes_velocity() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        let old_vel = p.velocity.clone();
        let gb = vec![0.5; 3];
        update_velocity(&mut p, &gb, 0.7, 1.5, 1.5, &mut rng);
        // velocity should have changed (unless extremely unlucky)
        let changed = p.velocity.iter().zip(&old_vel).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn position_update_moves_particle() {
        let mut rng = SimpleRng::new(42);
        let mut p = Particle::random(3, 0.0, 1.0, &mut rng);
        p.velocity = vec![0.1, 0.1, 0.1];
        let old_pos = p.position.clone();
        update_position(&mut p);
        for i in 0..3 {
            assert!((p.position[i] - old_pos[i] - 0.1).abs() < 1e-10);
        }
    }
}
