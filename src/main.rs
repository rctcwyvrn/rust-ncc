//! The entry point.
use rust_ncc::{experiments, world};
use std::time::Instant;

fn main() {
    let exp = experiments::model_compare::generate(Some(3), 2, false);
    let mut w = world::World::new(exp);

    let now = Instant::now();
    w.simulate(0.01 * 3600.0);

    println!("Simulation complete. {} s.", now.elapsed().as_secs());
}
