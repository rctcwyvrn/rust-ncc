// Copyright © 2020 Brian Merchant.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::math::v2d::V2D;
use crate::parameters::{
    CharQuantities, Parameters, WorldParameters,
};
use crate::utils::pcg32::Pcg32;

pub mod defaults;
pub mod markers;
//pub mod n_cells;
pub mod py_compare;
pub mod sep_pair;

#[derive(Clone, Copy)]
pub enum ExperimentType {
    NCells,
    SeparatedPair { sep_in_cell_diams: usize },
    PythonComparison,
}

/// Generate the experiment, so that it can be run.
pub fn generate(
    exp_ix: usize,
    cil: f64,
    coa: Option<f64>,
    cal: Option<f64>,
    adh: Option<f64>,
    seed: Option<u64>,
    exp_type: ExperimentType,
) -> Experiment {
    match exp_type {
        ExperimentType::SeparatedPair { sep_in_cell_diams } => {
            sep_pair::generate(
                exp_ix,
                cil,
                coa,
                cal,
                adh,
                seed,
                sep_in_cell_diams,
            )
        }
        ExperimentType::PythonComparison => unimplemented!(),
        ExperimentType::NCells => unimplemented!(),
    }
}

/// Specifies initial placement of the group.
pub struct GroupBBox {
    /// Width of group in terms of cell diameter.
    pub width: usize,
    /// Height of group in terms of cell diameter.
    pub height: usize,
    /// Bottom left of the group in normalized space units.
    pub bottom_left: V2D,
}

/// Information required for a cell group to be created.
pub struct CellGroup {
    /// The number of cells in the group.
    pub num_cells: usize,
    /// Initial layout of the cell group.
    pub layout: GroupBBox,
    /// Parameters shared by all cells in this group.
    pub parameters: Parameters,
}

/// Information required to create an experiment.
pub struct Experiment {
    pub file_name: String,
    /// Characteristic quantities.
    pub char_quants: CharQuantities,
    pub world_parameters: WorldParameters,
    /// List of cell groups involved in this experiment.
    pub cell_groups: Vec<CellGroup>,
    /// Random number generator to be used for various purposes.
    /// Initialized from a seed, otherwise from "entropy".
    pub rng: Pcg32,
    /// Seed that was used to initialize rng, if it generated from a
    /// seed.
    pub seed: Option<u64>,
}
