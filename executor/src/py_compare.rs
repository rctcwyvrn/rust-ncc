use rust_ncc::{exp_setups, world};
use std::time::Instant;

#[allow(unused)]
fn py_compare(
    exp_ix: usize,
    cil_mag: f64,
    cal_mag: Option<f64>,
    coa_mag: Option<f64>,
    adh_scale: Option<f64>,
    seed: Option<u64>,
) {
    let exp = exp_setups::py_compare::generate(
        exp_ix, cil_mag, coa_mag, cal_mag, adh_scale, seed,
    );
    let mut w = world::py_compare::World::new(exp);

    let now = Instant::now();
    w.simulate(0.01 * 3600.0);

    println!("Simulation complete. {} s.", now.elapsed().as_secs());
}
