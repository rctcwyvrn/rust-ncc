use crate::cell::chemistry::{distrib_gens, RgtpDistribution};
use crate::experiment_setups::markers::{BOT_HALF, TOP_HALF};
use crate::experiment_setups::{
    defaults, CellGroup, Experiment, GroupBBox,
};
use crate::math::v2d::V2D;
use crate::parameters::quantity::{Length, Quantity};
use crate::parameters::{
    CharQuantities, RawCloseBounds, RawCoaParams,
    RawInteractionParams, RawParameters, RawPhysicalContactParams,
};
use crate::utils::pcg32::Pcg32;
use crate::NVERTS;
use rand::SeedableRng;

/// Generate the group layout to use for this experiment.
fn group_bbox(
    num_cells: usize,
    char_quants: &CharQuantities,
    bottom_left: (Length, Length),
    width: usize,
    height: usize,
) -> Result<GroupBBox, String> {
    // specify initial location of group bottom left
    let bottom_left = V2D {
        x: char_quants.normalize(&bottom_left.0),
        y: char_quants.normalize(&bottom_left.1),
    };
    let r = GroupBBox {
        width,
        height,
        bottom_left,
    };

    if r.width * r.height > num_cells {
        Err(String::from(
            "Group layout area is too small to contain required number of cells.",
        ))
    } else {
        Ok(r)
    }
}

fn raw_params(
    rng: &mut Pcg32,
    randomization: bool,
    rac_act_bounds: [bool; NVERTS],
    rho_act_bounds: [bool; NVERTS],
) -> RawParameters {
    let init_rac = RgtpDistribution::new(
        distrib_gens::specific_random(rng, 0.1, rac_act_bounds),
        distrib_gens::random(rng, 0.1),
    );

    let init_rho = RgtpDistribution::new(
        distrib_gens::specific_random(rng, 0.1, rho_act_bounds),
        distrib_gens::random(rng, 0.1),
    );

    let mut raw_params = *defaults::RAW_PARAMS;
    raw_params.modify_randomization(randomization);
    raw_params.modify_init_rac(init_rac);
    raw_params.modify_init_rho(init_rho);

    raw_params
}

#[allow(clippy::too_many_arguments)]
fn make_cell_group(
    rng: &mut Pcg32,
    char_quants: &CharQuantities,
    randomization: bool,
    rac_act_bounds: [bool; NVERTS],
    rho_act_bounds: [bool; NVERTS],
    bot_left: (Length, Length),
    num_cells: usize,
    box_width: usize,
    box_height: usize,
) -> CellGroup {
    let raw_params = raw_params(
        rng,
        randomization,
        rac_act_bounds,
        rho_act_bounds,
    );
    let parameters = raw_params.refine(char_quants);
    CellGroup {
        num_cells,
        layout: group_bbox(
            num_cells,
            char_quants,
            bot_left,
            box_width,
            box_height,
        )
        .unwrap(),
        parameters,
    }
}

/// Define the cell groups that will exist in this experiment.
fn make_cell_groups(
    rng: &mut Pcg32,
    char_quants: &CharQuantities,
    randomization: bool,
    sep_in_cell_diams: usize,
) -> Vec<CellGroup> {
    let group_zero = make_cell_group(
        rng,
        char_quants,
        randomization,
        *TOP_HALF,
        *BOT_HALF,
        (Length(0.0), Length(0.0)),
        1,
        1,
        1,
    );
    let group_one = make_cell_group(
        rng,
        char_quants,
        randomization,
        *TOP_HALF,
        *BOT_HALF,
        (
            Length(0.0),
            defaults::CELL_DIAMETER.scale(sep_in_cell_diams as f64),
        ),
        1,
        1,
        1,
    );

    vec![group_zero, group_one]
}

pub fn generate(
    exp_ix: usize,
    cil_mag: f64,
    coa_mag: Option<f64>,
    cal_mag: Option<f64>,
    adh_scale: Option<f64>,
    seed: Option<u64>,
    sep_in_cell_diams: usize,
) -> Experiment {
    let (mut rng, randomization) = match seed {
        Some(s) => (Pcg32::seed_from_u64(s), true),
        None => (Pcg32::from_entropy(), false),
    };

    let char_quants = *defaults::CHAR_QUANTS;
    let mut raw_world_params = *defaults::RAW_WORLD_PARAMS;
    raw_world_params.modify_interactions(RawInteractionParams {
        coa: RawCoaParams::default_with_mag(coa_mag),
        chem_attr: None,
        bdry: None,
        phys_contact: RawPhysicalContactParams {
            range: RawCloseBounds {
                zero_at: defaults::PHYS_CLOSE_DIST.scale(2.0),
                one_at: *defaults::PHYS_CLOSE_DIST,
            },
            adh_mag: adh_scale.map(|x| defaults::ADH_MAG.scale(x)),
            cal_mag,
            cil_mag: defaults::CIL_MAG,
        },
    });
    (RawCoaParams::default_with_mag(coa_mag));
    let world_params = raw_world_params.refine(&char_quants);
    let cgs = make_cell_groups(
        &mut rng,
        &char_quants,
        randomization,
        sep_in_cell_diams,
    );

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
            "sep_pair", cil_mag, cal, adh, coa, seed_string, exp_ix
        ),
        char_quants,
        world_parameters: world_params,
        cell_groups: cgs,
        rng,
        seed,
    }
}
