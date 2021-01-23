pub mod cell;
pub mod experiments;
pub mod interactions;
pub mod math;
pub mod parameters;
pub mod utils;
pub mod world;

/// Number of vertices per model cell.
pub const NVERTS: usize = 16;
/// Default directory where simulation output will be placed.
pub const EULER_OUT_FILE: &str =
    "B:\\rust-ncc\\model-comparison\\rust-out\\out_euler.dat";
