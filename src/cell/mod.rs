// Copyright © 2020 Brian Merchant.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
pub mod chemistry;
pub mod states;
// pub mod geometry;
pub mod mechanics;
pub mod rkdp5;

use crate::cell::chemistry::RacRandState;
use crate::cell::rkdp5::AuxArgs;
use crate::cell::states::{ChemState, CoreState, GeomState, MechState};
use crate::interactions::{ContactData, Interactions};
use crate::math::geometry::{calc_poly_area, check_strong_intersection};
use crate::math::v2d::V2D;
use crate::parameters::{Parameters, WorldParameters};
use crate::utils::pcg32::Pcg32;
use crate::utils::{circ_ix_minus, circ_ix_plus};
use crate::NVERTS;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

/// Cell state structure.
#[derive(Copy, Clone, Deserialize, Serialize, PartialEq, Default, Debug)]
pub struct Cell {
    /// Index of cell within world.
    pub ix: usize,
    /// Index of group that cell belongs to.
    pub group_ix: usize,
    /// Core state of the cell (position, Rho GTPase).
    pub core: CoreState,
    /// Random Rac1 activity.
    pub rac_rand: RacRandState,
    /// Mechanical activity (forces).
    pub mech: MechState,
    /// Geometry (unit inward vecs, etc.).
    pub geom: GeomState,
    /// Chemical state (various reaction rates).
    pub chem: ChemState,
}

pub fn violates_volume_exclusion(
    test_v: &V2D,
    test_u: &V2D,
    test_w: &V2D,
    contacts: &[ContactData],
) -> bool {
    let point_pairs = [(test_u, test_v), (test_v, test_w)];
    for (p0, p1) in point_pairs.iter() {
        for cd in contacts.iter() {
            for edge in cd.poly.edges.iter() {
                if check_strong_intersection(p0, p1, edge) {
                    return true;
                }
            }
        }
    }
    false
}

fn move_point_out(
    new_u: &V2D,
    mut new_v: V2D,
    new_w: &V2D,
    mut good_v: V2D,
    contacts: &[ContactData],
    num_iters: u32,
) -> V2D {
    let mut n = 0;
    while n < num_iters {
        let test_v = 0.5 * (good_v + new_v);
        if violates_volume_exclusion(&test_v, new_u, new_w, contacts) {
            new_v = test_v;
        } else {
            good_v = test_v;
        }
        n += 1;
    }
    good_v
}

#[cfg(feature = "validate")]
pub fn confirm_volume_exclusion(
    vs: &[V2D; NVERTS],
    contacts: &[ContactData],
    msg: &str,
) -> Result<(), String> {
    use crate::math::v2d::poly_to_string;
    for (vi, v) in vs.iter().enumerate() {
        let u = &vs[circ_ix_minus(vi, NVERTS)];
        let w = &vs[circ_ix_plus(vi, NVERTS)];
        for ContactData { poly, .. } in contacts {
            if violates_volume_exclusion(v, u, w, contacts) {
                return Err(format!(
                    "{} violates volume exclusion.\n\
                    vs[{}] = {}, \n\
                    other poly = {}",
                    msg,
                    vi,
                    v,
                    &poly_to_string(&poly.verts)
                ));
            }
        }
    }
    Ok(())
}

pub fn enforce_volume_exclusion(
    old_vs: &[V2D; NVERTS],
    mut new_vs: [V2D; NVERTS],
    contacts: Vec<ContactData>,
) -> Result<[V2D; NVERTS], String> {
    #[cfg(feature = "validate")]
    confirm_volume_exclusion(&old_vs, &contacts, "old_vs")?;

    for vi in 0..NVERTS {
        let old_v = old_vs[vi];
        let new_v = new_vs[vi];
        let new_u = new_vs[circ_ix_minus(vi, NVERTS)];
        let new_w = new_vs[circ_ix_plus(vi, NVERTS)];
        new_vs[vi] =
            move_point_out(&new_u, new_v, &new_w, old_v, &contacts, 20);
    }

    #[cfg(feature = "validate")]
    confirm_volume_exclusion(&new_vs, &contacts, "new_vs")?;

    Ok(new_vs)
}

impl Cell {
    pub fn new(
        ix: usize,
        group_ix: usize,
        core: CoreState,
        interactions: &Interactions,
        parameters: &Parameters,
        rng: &mut Pcg32,
    ) -> Cell {
        let geom = core.calc_geom_state();
        let mech = core.calc_mech_state(&geom, parameters);
        let rac_rand = if parameters.randomization {
            RacRandState::new(rng, parameters)
        } else {
            RacRandState::default()
        };
        let chem = core.calc_chem_state(
            &geom,
            &mech,
            &rac_rand,
            &interactions,
            parameters,
        );
        Cell {
            ix,
            group_ix,
            core,
            rac_rand,
            geom,
            chem,
            mech,
        }
    }

    #[allow(unused)]
    /// Suppose our current state is `state`. We want to determine
    /// the next state after a time period `dt` has elapsed. We
    /// assume `(next_state - state)/delta(t) = delta(state)`.
    pub fn simulate_euler(
        &self,
        tstep: u32,
        n_int_steps: u32,
        interactions: &Interactions,
        contact_data: Vec<ContactData>,
        world_parameters: &WorldParameters,
        parameters: &Parameters,
        rng: &mut Pcg32,
        out_file: &mut File,
    ) -> Result<Cell, String> {
        let mut state = self.core;
        // Assumed normalized time by time provided in CharQuant.
        // Therefore, we can take the time period to integrate over
        // as 1.0.
        let dt = 1.0 / (n_int_steps as f32);
        for _ in 0..n_int_steps {
            let geom_state = state.calc_geom_state();
            let mech_state = state.calc_mech_state(&geom_state, parameters);
            let chem_state = state.calc_chem_state(
                &geom_state,
                &mech_state,
                &self.rac_rand,
                &interactions,
                parameters,
            );
            writeln!(out_file, "------------------------------").unwrap();
            writeln!(
                out_file,
                "poly: [{}]",
                state
                    .poly
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rac_acts: [{}]",
                state
                    .rac_acts
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rac_inacts: [{}]",
                state
                    .rac_inacts
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rho_acts: [{}]",
                state
                    .rho_acts
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rho_inacts: [{}]",
                state
                    .rho_inacts
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "tot_forces: [{}]",
                mech_state
                    .sum_forces
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "uivs: [{}]",
                geom_state
                    .unit_inward_vecs
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rgtp_forces: [{}]",
                mech_state
                    .rgtp_forces
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "edge_forces: [{}]",
                mech_state
                    .edge_forces
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "cyto_forces: [{}]",
                mech_state
                    .cyto_forces
                    .iter()
                    .map(|p| format!("[{}, {}]", p.x, p.y))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "kgtps_rac: [{}]",
                chem_state
                    .kgtps_rac
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "kdgtps_rac: [{}]",
                chem_state
                    .kdgtps_rac
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "kgtps_rho: [{}]",
                chem_state
                    .kgtps_rho
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "kdgtps_rho: [{}]",
                chem_state
                    .kdgtps_rho
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "conc_rac_acts: [{}]",
                chem_state
                    .conc_rac_acts
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "x_cils: [{}]",
                interactions
                    .x_cils
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "x_coas: [{}]",
                interactions
                    .x_coas
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "rac_act_net_fluxes: [{}]",
                chem_state
                    .rac_act_net_fluxes
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "edge_strains: [{}]",
                mech_state
                    .edge_strains
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                out_file,
                "poly_area: [{}]",
                vec![calc_poly_area(&state.poly).to_string(); NVERTS]
                    .join(", ")
            )
            .unwrap();
            writeln!(out_file, "------------------------------").unwrap();
            // d(state)/dt = dynamics_f(state) <- calculate RHS of ODE
            let delta = CoreState::dynamics_f(
                &state,
                &self.rac_rand,
                &interactions,
                world_parameters,
                parameters,
            );
            state = state + dt * delta;
        }
        let geom_state = state.calc_geom_state();
        let mech_state = state.calc_mech_state(&geom_state, parameters);
        let chem_state = state.calc_chem_state(
            &geom_state,
            &mech_state,
            &self.rac_rand,
            &interactions,
            parameters,
        );
        // Enforcing volume exclusion! Tricky!
        state.poly = enforce_volume_exclusion(
            &self.core.poly,
            state.poly,
            contact_data,
        )?;
        let geom_state = state.calc_geom_state();

        #[cfg(feature = "validate")]
        state.validate("euler", &parameters)?;

        Ok(Cell {
            ix: self.ix,
            group_ix: self.group_ix,
            core: state,
            rac_rand: self.rac_rand.update(tstep, rng, parameters),
            geom: geom_state,
            chem: chem_state,
            mech: mech_state,
        })
    }

    pub fn simulate_rkdp5(
        &self,
        tstep: u32,
        interactions: &Interactions,
        contact_data: Vec<ContactData>,
        world_parameters: &WorldParameters,
        parameters: &Parameters,
        rng: &mut Pcg32,
    ) -> Result<Cell, String> {
        let aux_args = AuxArgs {
            max_iters: 100,
            atol: 1e-8,
            rtol: 1e-3,
            init_h_factor: Some(0.1),
        };
        let result = rkdp5::integrator(
            1.0,
            CoreState::dynamics_f,
            &self.core,
            &self.rac_rand,
            interactions,
            world_parameters,
            parameters,
            aux_args,
        );

        let mut state =
            result.y.expect("rkdp5 integrator: too many iterations!");
        let geom_state = state.calc_geom_state();
        let mech_state = state.calc_mech_state(&geom_state, parameters);
        let chem_state = state.calc_chem_state(
            &geom_state,
            &mech_state,
            &self.rac_rand,
            &interactions,
            parameters,
        );
        state.poly =
            enforce_volume_exclusion(&self.core.poly, state.poly, contact_data)
                .map_err(|e| format!("ci={}\n{}", self.ix, e))?;
        let geom_state = state.calc_geom_state();

        #[cfg(feature = "validate")]
        state.validate("rkdp5", &parameters)?;

        Ok(Cell {
            ix: self.ix,
            group_ix: self.group_ix,
            core: state,
            rac_rand: self.rac_rand.update(tstep, rng, parameters),
            geom: geom_state,
            chem: chem_state,
            mech: mech_state,
        })
    }
}

/// Calculate the area of an "ideal" initial cell of radius R, if it has n vertices.
pub fn calc_init_cell_area(r: f32) -> f32 {
    let poly_coords = (0..NVERTS)
        .map(|vix| {
            let theta = (vix as f32) / (NVERTS as f32) * 2.0 * PI;
            V2D {
                x: r * theta.cos(),
                y: r * theta.sin(),
            }
        })
        .collect::<Vec<V2D>>();
    calc_poly_area(&poly_coords)
}
