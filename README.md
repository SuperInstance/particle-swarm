# particle-swarm

Particle swarm optimization library in Rust with zero external dependencies.

## Features

- **Velocity Updates**: Standard PSO velocity rule with configurable coefficients
- **Inertia Strategies**: Constant and linearly decreasing inertia weights
- **Topologies**: Star (global), ring, and von Neumann grid neighborhoods
- **Multi-Objective PSO**: Pareto front tracking with dominance detection
- **Bounds Handling**: Position and velocity clamping

## Usage

```rust
use particle_swarm::{SimpleRng, PSO};

let mut rng = SimpleRng::new(42);
let result = PSO::new(30, 2, -10.0, 10.0)
    .run(&mut rng, |x| x[0]*x[0] + x[1]*x[1], 200, 0.7, 1.5, 1.5);
println!("Best: {:?}, Fitness: {}", result.0, result.1);
```

## License

MIT
