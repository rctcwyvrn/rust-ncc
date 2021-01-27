// Copyright © 2020 Brian Merchant.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::cell::states::fmt_var_arr;
use crate::math::hill_function3;
use crate::parameters::Parameters;
use crate::utils::normal::NormalDistrib;
use crate::utils::pcg32::Pcg32;
use crate::utils::{circ_ix_minus, circ_ix_plus};
use crate::NVERTS;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::{Debug, Display};

pub enum DistributionType {
    /// Distribute Rho GTPase randomly over all vertices.
    Random,
    /// Mark vertices `true` if Rho GTPase is to be placed there.
    Specific([bool; NVERTS]),
}

pub struct DistributionScheme {
    pub frac: f32,
    pub ty: DistributionType,
}

impl DistributionScheme {
    /// [1.0, 10.0, 4.0, 5.0] (consider the relative fraction of rho gtpase), sum = 20.0,,
    /// go through each and divide by 20 = [0.05, 0.5, 0.2, 0.25]
    /// if we now sum up everything in the new array, sum = 1.0
    fn scaled_unitize(
        frac: f32,
        mut distrib: [f32; NVERTS],
    ) -> [f32; NVERTS] {
        let sum: f32 = distrib.iter().sum();
        distrib.iter_mut().for_each(|e| *e = *e * frac / sum);
        distrib
    }

    fn gen_random(rng: &mut Pcg32, frac: f32) -> [f32; NVERTS] {
        let mut r = [0.0; NVERTS];
        let prob_distrib: Uniform<f32> =
            Uniform::new_inclusive(0.0, 1.0);
        r.iter_mut().for_each(|e| {
            *e = rng.sample(prob_distrib);
        });
        Self::scaled_unitize(frac, r)
    }

    fn gen_specific(
        frac: f32,
        marked_verts: &[bool; NVERTS],
    ) -> [f32; NVERTS] {
        //println!("marking in gen_specific: {:?}", &marked_verts);
        let mut r = [0.0; NVERTS];
        marked_verts.iter().zip(r.iter_mut()).for_each(
            |(&marked, e)| {
                if marked {
                    *e = 1.0;
                }
            },
        );
        Self::scaled_unitize(frac, r)
    }

    pub fn generate(&self, rng: &mut Pcg32) -> [f32; NVERTS] {
        match &self.ty {
            DistributionType::Random => {
                Self::gen_random(rng, self.frac)
            }
            DistributionType::Specific(marks) => {
                Self::gen_specific(self.frac, marks)
            }
        }
    }
}

#[derive(
    Clone, Copy, Deserialize, Serialize, Default, Debug, PartialEq,
)]
pub struct RgtpDistribution {
    pub active: [f32; NVERTS],
    pub inactive: [f32; NVERTS],
}

impl RgtpDistribution {
    pub fn generate(
        act_distrib: DistributionScheme,
        inact_distrib: DistributionScheme,
        rng: &mut Pcg32,
    ) -> Result<RgtpDistribution, String> {
        if act_distrib.frac + inact_distrib.frac > 1.0 {
            Err("active + inactive > 1.0".to_string())
        } else {
            Ok(RgtpDistribution {
                active: act_distrib.generate(rng),
                inactive: inact_distrib.generate(rng),
            })
        }
    }
}

fn calc_directed_fluxes(
    edge_lens: &[f32; NVERTS],
    rgtp_d: f32,
    conc_rgtps: &[f32; NVERTS],
) -> [f32; NVERTS] {
    let mut r = [0.0_f32; NVERTS];
    for i in 0..NVERTS {
        let plus_i = circ_ix_plus(i, NVERTS);
        r[i] = -1.0 * rgtp_d * (conc_rgtps[plus_i] - conc_rgtps[i])
            / edge_lens[i];
    }
    r
}

pub fn calc_net_fluxes(
    edge_lens: &[f32; NVERTS],
    rgtp_d: f32,
    conc_rgtps: &[f32; NVERTS],
) -> [f32; NVERTS] {
    let directed_fluxes =
        calc_directed_fluxes(edge_lens, rgtp_d, conc_rgtps);
    let mut r = [0.0_f32; NVERTS];
    (0..NVERTS).for_each(|i| {
        let min_i = circ_ix_minus(i, NVERTS);
        r[i] = directed_fluxes[min_i] - directed_fluxes[i];
    });
    r
}

/// Calculate approximate concentration of a Rho GTPase at a vertex.
pub fn calc_conc_rgtps(
    avg_edge_lens: &[f32; NVERTS],
    rgtps: &[f32; NVERTS],
) -> [f32; NVERTS] {
    let mut r = [0.0_f32; NVERTS];
    (0..NVERTS).for_each(|i| r[i] = rgtps[i] / avg_edge_lens[i]);
    r
}

//TODO(ES): make the documentation for this better.
/// Calculates Rac1 activation rates, as discussed in SI.
pub fn calc_kgtps_rac(
    rac_acts: &[f32; NVERTS],
    conc_rac_acts: &[f32; NVERTS],
    x_rands: &[f32; NVERTS],
    x_coas: &[f32; NVERTS],
    x_chemoas: &[f32; NVERTS],
    x_cals: &[f32; NVERTS],
    kgtp_rac_base: f32,
    kgtp_rac_auto: f32,
    halfmax_rac_conc: f32,
) -> [f32; NVERTS] {
    let nvs = rac_acts.len();
    let mut kgtps_rac = [0.0_f32; NVERTS];

    for i in 0..nvs {
        // Base activation rate of Rac1 is increased (not multiplied!)
        // by: CAL, randomization, and co-attraction.
        let base = (x_cals[i] + x_rands[i] + x_coas[i] + 1.0)
            * kgtp_rac_base;
        // Auto activation rate of Rac1 is increased by
        // chemoattraction only. This is because we assume that Sdf1
        // only stabilizes existing Rac1 activity. It does not
        // generate new Rac1 activity. See SI in Merchant(2020)?
        let auto_factor = {
            let af =
                hill_function3(halfmax_rac_conc, conc_rac_acts[i])
                    * (1.0 + x_chemoas[i]);
            // This comes from the Python code. It's necessary for
            // auto-activation to not blow up, but it also kind of
            // changes the shape of the sigmoid. Is this recorded
            // SI?
            if af > 1.25 {
                1.25
            } else {
                af
            }
        };
        let auto = auto_factor * kgtp_rac_auto;
        kgtps_rac[i] = base + auto;
    }
    kgtps_rac
}

/// Calculates Rac1 inactivation rates, as discussed in SI.
pub fn calc_kdgtps_rac(
    rac_acts: &[f32; NVERTS],
    conc_rho_acts: &[f32; NVERTS],
    x_cils: &[f32; NVERTS],
    x_tens: f32,
    kdgtp_rac_base: f32,
    kdgtp_rho_on_rac: f32,
    halfmax_conc_rho: f32,
) -> [f32; NVERTS] {
    let nvs = rac_acts.len();
    let mut kdgtps_rac = [0.0_f32; NVERTS];

    for i in 0..nvs {
        // Baseline is affected by tension inhibition, and CIL.
        let cil = (x_cils[i]
            + x_cils[circ_ix_plus(i, NVERTS)]
            + x_cils[circ_ix_minus(i, NVERTS)])
            / 3.0;
        let base = (1.0 + x_tens + cil) * kdgtp_rac_base;
        // Effect of RhoA on Rac1, related to activity of RhoA at a
        // vertex.
        let mutual =
            hill_function3(halfmax_conc_rho, conc_rho_acts[i])
                * kdgtp_rho_on_rac;
        kdgtps_rac[i] = base + mutual;
    }

    kdgtps_rac
}

/// Calculates RhoA activation rates, as discussed in SI.
pub fn calc_kgtps_rho(
    rho_acts: &[f32; NVERTS],
    conc_rho_acts: &[f32; NVERTS],
    x_cils: &[f32; NVERTS],
    kgtp_rho_base: f32,
    halfmax_rho_thresh: f32,
    kgtp_rho_auto: f32,
) -> [f32; NVERTS] {
    let nvs = rho_acts.len();
    let mut kgtps_rho = [0.0_f32; NVERTS];

    for i in 0..nvs {
        let base = (1.0 + x_cils[i]) * kgtp_rho_base;
        let auto =
            hill_function3(halfmax_rho_thresh, conc_rho_acts[i])
                * kgtp_rho_auto;
        kgtps_rho[i] = base + auto;
    }

    kgtps_rho
}

/// Calculates RhoA inactivation rates, as discussed in SI.
pub fn calc_kdgtps_rho(
    rho_acts: &[f32; NVERTS],
    conc_rac_acts: &[f32; NVERTS],
    kdgtp_rho_base: f32,
    kdgtp_rac_on_rho: f32,
    halfmax_conc_rac: f32,
) -> [f32; NVERTS] {
    let nvs = rho_acts.len();
    let mut kdgtps_rho = [0.0_f32; NVERTS];

    for i in 0..nvs {
        let mutual =
            hill_function3(halfmax_conc_rac, conc_rac_acts[i])
                * kdgtp_rac_on_rho;
        kdgtps_rho[i] = kdgtp_rho_base + mutual;
    }

    kdgtps_rho
}

#[derive(Copy, Clone, Deserialize, Serialize, PartialEq)]
pub struct RacRandState {
    pub enabled: bool,
    /// When does the next update occur?
    pub next_update: u32,
    /// Rac1 randomization factors per vertex.
    pub x_rands: [f32; NVERTS],
    distrib: NormalDistrib,
}

impl Debug for RacRandState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RacRandState {{ enabled: {}, next_update: {}, x_rands: [{}], distrib: <skipped> }}", self.enabled, self.next_update, self.x_rands.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(", "))
    }
}

impl RacRandState {
    pub fn gen_rand_factors(
        rng: &mut Pcg32,
        num_rand_verts: usize,
        rand_mag: f32,
    ) -> [f32; NVERTS] {
        let vs = (0..NVERTS).collect::<Vec<usize>>();
        let mut r = [0.0; NVERTS];
        vs.choose_multiple(rng, num_rand_verts)
            .for_each(|&v| r[v] = rand_mag);
        r
    }

    pub fn new(
        rng: &mut Pcg32,
        parameters: &Parameters,
    ) -> RacRandState {
        let ut = Uniform::from(0.0..parameters.rand_avg_t);
        RacRandState {
            enabled: true,
            next_update: ut.sample(rng).floor() as u32,
            x_rands: Self::gen_rand_factors(
                rng,
                parameters.num_rand_vs as usize,
                parameters.rand_mag,
            ),
            distrib: NormalDistrib::new(
                parameters.rand_avg_t,
                parameters.rand_std_t,
            ),
        }
    }

    pub fn update(
        &self,
        tstep: u32,
        rng: &mut Pcg32,
        parameters: &Parameters,
    ) -> RacRandState {
        if tstep == self.next_update && self.enabled {
            let next_update =
                tstep + self.distrib.sample(rng).floor() as u32;
            // println!("random update from {} to {}", tstep, next_update);
            let x_rands = Self::gen_rand_factors(
                rng,
                parameters.num_rand_vs as usize,
                parameters.rand_mag,
            );
            // println!("{:?}", x_rands);
            RacRandState {
                enabled: self.enabled,
                next_update,
                x_rands,
                distrib: self.distrib,
            }
        } else {
            *self
        }
    }
}

impl Display for RacRandState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_var_arr(f, "rfs", &self.x_rands)?;
        writeln!(f, "next_update: {}", self.next_update)
    }
}

impl Default for RacRandState {
    fn default() -> Self {
        RacRandState {
            enabled: false,
            next_update: 0,
            x_rands: [0.0; NVERTS],
            distrib: NormalDistrib::new(0.0, 1.0),
        }
    }
}
