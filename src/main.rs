#![allow(clippy::too_many_arguments)]
//! The entry point.
use rust_ncc::{experiments, world, EULER_OUT_FILE};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let exp = experiments::model_compare::generate(Some(3), 1, false);

    let mut w = world::World::new(exp, PathBuf::from(EULER_OUT_FILE));

    let now = Instant::now();
    w.simulate(3.0 * 3600.0);

    println!("Simulation complete. {} s.", now.elapsed().as_secs());
}
