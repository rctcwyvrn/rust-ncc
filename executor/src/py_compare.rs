use rust_ncc::{experiment_setups, world};
use std::time::Instant;

fn py_compare() {
    let exp = experiment_setups::py_compare::generate(
        Some(3),
        vec![1, 1],
        false,
        None,
        0.0,
    );
    let mut w = world::py_compare::World::new(exp);

    let now = Instant::now();
    w.simulate(0.01 * 3600.0);

    println!("Simulation complete. {} s.", now.elapsed().as_secs());
}
