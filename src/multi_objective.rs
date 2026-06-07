//! Multi-objective PSO with Pareto front tracking.

/// An objective vector for multi-objective optimization.
#[derive(Clone, Debug)]
pub struct ObjectiveVec {
    pub values: Vec<f64>,
}

impl ObjectiveVec {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Check if this solution dominates `other` (all objectives <=, at least one <).
    pub fn dominates(&self, other: &ObjectiveVec) -> bool {
        let mut any_less = false;
        for (a, b) in self.values.iter().zip(&other.values) {
            if a > b { return false; }
            if a < b { any_less = true; }
        }
        any_less
    }
}

/// A Pareto front (non-dominated solutions).
#[derive(Clone, Debug)]
pub struct ParetoFront {
    pub solutions: Vec<(Vec<f64>, ObjectiveVec)>, // (position, objectives)
}

impl Default for ParetoFront {
    fn default() -> Self {
        Self::new()
    }
}

impl ParetoFront {
    pub fn new() -> Self {
        Self { solutions: Vec::new() }
    }

    /// Add a solution, removing any dominated solutions.
    pub fn add(&mut self, position: Vec<f64>, objectives: ObjectiveVec) {
        // Remove dominated solutions
        self.solutions.retain(|(_, obj)| !objectives.dominates(obj));
        // Check if new solution is dominated
        for (_, obj) in &self.solutions {
            if obj.dominates(&objectives) {
                return; // dominated, don't add
            }
        }
        self.solutions.push((position, objectives));
    }

    /// Number of solutions on the Pareto front.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Is the front empty?
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
}

/// Multi-objective PSO runner.
pub struct MultiObjectivePSO {
    pub swarm_size: usize,
    pub dim: usize,
    pub min: f64,
    pub max: f64,
}

impl MultiObjectivePSO {
    pub fn new(swarm_size: usize, dim: usize, min: f64, max: f64) -> Self {
        Self { swarm_size, dim, min, max }
    }

    /// Run multi-objective PSO.
    pub fn run(
        &self,
        rng: &mut crate::rng::SimpleRng,
        objectives: &[fn(&[f64]) -> f64],
        iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
    ) -> ParetoFront {
        use crate::particle::Particle;
        use crate::velocity;

        let mut swarm: Vec<Particle> = (0..self.swarm_size)
            .map(|_| Particle::random(self.dim, self.min, self.max, rng))
            .collect();

        // Evaluate initial
        for p in &mut swarm {
            let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&p.position)).collect();
            // Use first objective as primary fitness for PSO updates
            p.fitness = obj_vals[0];
            // Use sum for personal best comparison (scalarization)
            p.best_fitness = obj_vals.iter().sum();
            p.best_position = p.position.clone();
        }

        let mut front = ParetoFront::new();
        for p in &swarm {
            let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&p.position)).collect();
            front.add(p.position.clone(), ObjectiveVec::new(obj_vals));
        }

        // Find global best for PSO movement (use first objective)
        let mut global_best = swarm[0].best_position.clone();
        let mut global_best_fitness = swarm[0].best_fitness;
        for p in &swarm {
            if p.best_fitness < global_best_fitness {
                global_best_fitness = p.best_fitness;
                global_best = p.best_position.clone();
            }
        }

        for _ in 0..iterations {
            for p in &mut swarm {
                velocity::update_velocity(p, &global_best, w, c1, c2, rng);
                velocity::update_position(p);
                p.clamp_position(self.min, self.max);

                let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&p.position)).collect();
                let scalar = obj_vals.iter().sum();
                p.fitness = obj_vals[0];
                if scalar < p.best_fitness {
                    p.best_fitness = scalar;
                    p.best_position = p.position.clone();
                }
                if scalar < global_best_fitness {
                    global_best_fitness = scalar;
                    global_best = p.best_position.clone();
                }

                front.add(p.position.clone(), ObjectiveVec::new(obj_vals));
            }
        }

        front
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SimpleRng;

    #[test]
    fn dominance() {
        let a = ObjectiveVec::new(vec![1.0, 2.0]);
        let b = ObjectiveVec::new(vec![2.0, 3.0]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn non_dominating() {
        let a = ObjectiveVec::new(vec![1.0, 3.0]);
        let b = ObjectiveVec::new(vec![3.0, 1.0]);
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn pareto_front_add_non_dominated() {
        let mut pf = ParetoFront::new();
        pf.add(vec![1.0], ObjectiveVec::new(vec![1.0, 3.0]));
        pf.add(vec![2.0], ObjectiveVec::new(vec![3.0, 1.0]));
        assert_eq!(pf.len(), 2);
    }

    #[test]
    fn pareto_front_removes_dominated() {
        let mut pf = ParetoFront::new();
        pf.add(vec![1.0], ObjectiveVec::new(vec![3.0, 3.0]));
        pf.add(vec![2.0], ObjectiveVec::new(vec![1.0, 1.0]));
        assert_eq!(pf.len(), 1);
    }

    #[test]
    fn mopso_produces_front() {
        let mut rng = SimpleRng::new(42);
        let pso = MultiObjectivePSO::new(20, 2, -5.0, 5.0);
        let front = pso.run(
            &mut rng,
            &[|x| x[0] * x[0], |x| (x[0] - 2.0).powi(2)],
            50,
            0.7,
            1.5,
            1.5,
        );
        assert!(!front.is_empty());
    }
}
