use crate::cell::chemistry::{distrib_gens, RgtpDistribution};
use crate::exp_setups::{CellGroup, Experiment, GroupBBox};
use crate::math::v2d::V2D;
use crate::parameters::quantity::{Length, Quantity};
use crate::parameters::{
    CharQuantities, RawCloseBounds, RawInteractionParams,
    RawParameters, RawPhysicalContactParams,
};
use crate::utils::pcg32::Pcg32;
use crate::NVERTS;
use rand::SeedableRng;

use crate::exp_setups::defaults::{
    ADH_MAG, CHAR_QUANTS, CIL_MAG, PHYS_CLOSE_DIST,
    RAW_COA_PARAMS_WITH_ZERO_MAG, RAW_PARAMS, RAW_WORLD_PARAMS,
};

/// Generate the group layout to use for this experiment.
fn group_bbox(
    group_ix: usize,
    char_quants: &CharQuantities,
    raw_params: &RawParameters,
) -> Result<GroupBBox, String> {
    // specify initial location of group centroid
    let bottom_left = V2D {
        x: char_quants
            .normalize(&raw_params.cell_diam.scale(group_ix as f64)),
        y: char_quants.normalize(&Length(0.0)),
    };
    let r = GroupBBox {
        width: 1,
        height: 1,
        bottom_left,
    };
    if r.width * r.height < 1 {
        Err(String::from(
            "Group layout area is too small to contain required number of cells.",
        ))
    } else {
        Ok(r)
    }
}

fn raw_params(group_ix: usize, randomization: bool) -> RawParameters {
    #![allow(unused_attributes)]
    #[macro_use]
    use crate::mark_verts;

    let right = mark_verts!(0, 1, 2, 3);
    let left = mark_verts!(8, 9, 10, 11);

    let (specific_rac, specific_rho) = match group_ix {
        0 => (right, left),
        1 => (left, right),
        _ => panic!("received group ix > 1"),
    };

    let rac_distrib =
        distrib_gens::specific_uniform(0.3, specific_rac);
    let init_rac = RgtpDistribution::new(rac_distrib, rac_distrib);

    let rho_distrib =
        distrib_gens::specific_uniform(0.3, specific_rho);
    let init_rho = RgtpDistribution::new(rho_distrib, rho_distrib);

    RAW_PARAMS
        .modify_randomization(randomization)
        .modify_init_rac(init_rac)
        .modify_init_rho(init_rho)
}

#[allow(clippy::too_many_arguments)]
fn make_cell_group(
    group_ix: usize,
    char_quants: &CharQuantities,
    randomization: bool,
    num_cells: usize,
) -> CellGroup {
    let raw_params = raw_params(group_ix, randomization);
    let parameters = raw_params.refine(char_quants);
    CellGroup {
        num_cells,
        layout: group_bbox(group_ix, char_quants, &raw_params)
            .unwrap(),
        parameters,
    }
}

/// Define the cell groups that will exist in this experiment.
fn make_cell_groups(
    char_quants: &CharQuantities,
    randomization: bool,
) -> Vec<CellGroup> {
    (0..2)
        .map(|group_ix| {
            make_cell_group(group_ix, char_quants, randomization, 1)
        })
        .collect::<Vec<CellGroup>>()
}

#[allow(unused)]
pub fn generate(
    exp_ix: usize,
    cil_mag: f64,
    coa_mag: Option<f64>,
    cal_mag: Option<f64>,
    adh_scale: Option<f64>,
    seed: Option<u64>,
) -> Experiment {
    let (rng, randomization) = match seed {
        Some(s) => (Pcg32::seed_from_u64(s), true),
        None => (Pcg32::from_entropy(), false),
    };

    let char_quants = *CHAR_QUANTS;
    let raw_world_params =
        RAW_WORLD_PARAMS.modify_interactions(RawInteractionParams {
            coa: coa_mag.map(|mag| {
                RAW_COA_PARAMS_WITH_ZERO_MAG.modify_mag(mag)
            }),
            chem_attr: None,
            bdry: None,
            phys_contact: RawPhysicalContactParams {
                range: RawCloseBounds {
                    zero_at: PHYS_CLOSE_DIST.scale(2.0),
                    one_at: *PHYS_CLOSE_DIST,
                },
                adh_mag: adh_scale.map(|x| ADH_MAG.scale(x)),
                cal_mag,
                cil_mag: CIL_MAG,
            },
        });
    let world_params = raw_world_params.refine(&char_quants);
    let cgs = make_cell_groups(&char_quants, randomization);

    //convert the option into string
    let cal = if let Some(i) = cal_mag {
        i.to_string()
    } else {
        "None".to_string()
    };

    let adh = if let Some(i) = adh_scale {
        i.to_string()
    } else {
        "None".to_string()
    };

    let coa = if let Some(i) = coa_mag {
        i.to_string()
    } else {
        "None".to_string()
    };

    let seed_string = if let Some(i) = seed {
        i.to_string()
    } else {
        "OFF".to_string()
    };

    Experiment {
        file_name: format!(
            "{}_cil={}_cal={}_coa={}_adh={}_seed={}_{}",
            "py_compare", cil_mag, cal, adh, coa, seed_string, exp_ix
        ),
        char_quants,
        world_parameters: world_params,
        cell_groups: cgs,
        rng,
        seed,
    }
}
