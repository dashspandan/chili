#!/usr/bin/env python3
"""
run_chili_magmavol.py

CHILI intercomparison runner for the MAGMAVOL static model.
Produces a per-planet compo file (temperature grid) and CHILI-formatted
static output CSVs for TRAPPIST-1e, TRAPPIST-1b, TRAPPIST-1alpha, Earth,
and Venus at specified characteristic ages (tau3, tau4, tau6).

TRAPPIST-1e/b/alpha share Earth-homogenised mass/radius per CHILI protocol
and therefore share a single compo file. Venus has its own mass/radius
(Table 2) and therefore requires its own compo file and grid run. Earth
and Venus share identical initial volatile inventories with the exoplanet
cases, but their hot/cold evolutionary model assignments are independent
and to be filled in later.

Usage: run from anywhere — all paths are absolute.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import glob
import logging
import shutil
from scipy.interpolate import UnivariateSpline

# -------------------------------------------------------------------------
# Add MAGMAVOL to path so we can import it
# -------------------------------------------------------------------------
MAGMAVOL_DIR = '/home/spandan/Desktop/magmavol/ThermoEngine/vaporock/notebooks/MAGMAVOL'
sys.path.insert(0, MAGMAVOL_DIR)
import MAGMAVOL_2 as mag

# -------------------------------------------------------------------------
# CHILI intercomparison paths
# -------------------------------------------------------------------------
CHILI_DIR    = '/home/spandan/Desktop/chili initiative/chili/intercomparison'
EVOL_DIR     = os.path.join(CHILI_DIR, 'outputs')

# OUTPUTS_DIR and INPUTS_DIR are set after FO2_CALCULATION is defined below,
# since the subfolder name (magmavol_massbalance / magmavol_meltfug) depends on it.

# -------------------------------------------------------------------------
# MAGMAVOL reservoir — use absolute MAIN_DIR + relative reservoir name
# -------------------------------------------------------------------------
mag.MAIN_DIR = MAGMAVOL_DIR
mag.DIR_FOR_RESERVOIR_OF_ATMO_SOLUTIONS = 'RESERVOIR_SOLUTIONS_MO_ATMO_EQUIL'

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
R_earth = 6.371e6    # m
M_earth = 5.972e24   # kg
G       = 6.674e-11  # m^3 kg^-1 s^-2

# -------------------------------------------------------------------------
# Per-planet mass/radius — CHILI Table 2 (Earth, Venus) and Table 4
# (exoplanets, homogenised to Earth-like mass/radius).
# -------------------------------------------------------------------------
PLANET_PARAMS = {
	'earth'         : {'Mp': 1.0   * M_earth, 'Rp': 1.0  * R_earth},
	'venus'         : {'Mp': 0.815 * M_earth, 'Rp': 0.95 * R_earth},
	'trappist1e'    : {'Mp': 1.0   * M_earth, 'Rp': 1.0  * R_earth},
	'trappist1b'    : {'Mp': 1.0   * M_earth, 'Rp': 1.0  * R_earth},
	'trappist1alpha': {'Mp': 1.0   * M_earth, 'Rp': 1.0  * R_earth},
}

for _planet, _params in PLANET_PARAMS.items():
	_params['g'] = G * _params['Mp'] / _params['Rp']**2

# -------------------------------------------------------------------------
# Fixed initial volatile inventories — CHILI Table 2 / Table 4
# (identical across Earth, Venus, and the exoplanet cases)
# -------------------------------------------------------------------------
massH_kg = 4.7e20    # kg
massC_kg = 2.73e20   # kg

def mass_to_pressure_bar(mass_kg, g, Rp):
	"""Convert atmospheric elemental mass (kg) to surface pressure (bar)."""
	return (mass_kg * g / (4 * np.pi * Rp**2)) / 1e5


def get_volatile_pressures(planet):
	"""Return (pH_bar, pC_bar) for a given planet using its own g/Rp.

	Used by run_full_grid() / get_compo_file_path() / interpolate_compo_file(),
	which build the original Table-2/4-driven per-planet compo table.
	NOT used by run_chili_single(), which seeds H/C/O/fO2 from a specific
	evolutionary model's row at a specific tau instead (see
	extract_seed_from_evolution() below).
	"""
	params = PLANET_PARAMS[planet]
	pH_bar = mass_to_pressure_bar(massH_kg, params['g'], params['Rp'])
	pC_bar = mass_to_pressure_bar(massC_kg, params['g'], params['Rp'])
	return pH_bar, pC_bar


# -------------------------------------------------------------------------
# Elemental partial pressures from the evolution model outputs
#
# mass_to_pressure_bar above returns the WEIGHT PER UNIT AREA of an
# element, M_X g / A, which is NOT its partial pressure. MAGMAVOL's
# targets are monatomic-equivalent partial pressures, sum_i nu(X,i)*p_i,
# and for a well-mixed hydrostatic atmosphere
#
#     P_X = p_surf * (M_X / sum M) * (mmw / mu_X)
#
# The two differ by mu_bar/mu_X, which is a different factor for each
# element, so elemental RATIOS built the weight-per-area way are mass
# ratios rather than molar ratios. Across the CHILI models that made
# hydrogen 18 to 42 times too low relative to carbon and oxygen.
#
# Two routes are available and both are computed:
#
#   species  sum the reported molecular partial pressures weighted by
#            stoichiometry. Touches no masses, so a wrong massO_atm
#            cannot reach it. Preferred whenever the reported species
#            account for p_surf.
#   mass     the formula above. Normalises by the sum of all three
#            masses, so a wrong oxygen mass inflates hydrogen and
#            carbon too: halving massO_atm leaves oxygen 16% low but
#            hydrogen and carbon 68% high. Used only as a fallback.
#
# On the CHILI outputs the species route is available for every model at
# every tau, and neongooey, gooey and moai have oxygen masses that are
# wrong by four to six orders of magnitude, so the distinction matters.
# -------------------------------------------------------------------------
ATOMIC_MASS_KG_PER_MOL = {'H': 1.008e-3, 'C': 12.011e-3, 'O': 15.999e-3}

REPORTED_SPECIES = {
	'p_H2O(bar)' : {'H': 2, 'O': 1},
	'p_CO2(bar)' : {'C': 1, 'O': 2},
	'p_CO(bar)'  : {'C': 1, 'O': 1},
	'p_H2(bar)'  : {'H': 2},
	'p_CH4(bar)' : {'C': 1, 'H': 4},
	'p_O2(bar)'  : {'O': 2},
}

# how far the reported molecular partials may fall short of p_surf
# before the species route is treated as incomplete
SPECIES_COVERAGE_TOL = 0.02


def elemental_pressures_from_species(row):
	"""
	Monatomic-equivalent pressures (bar) from the reported molecular
	partial pressures: P_X = sum_i nu(X, i) * p_i

	Returns (dict, reported_total_bar, n_species_present). An absent or
	NaN column counts as zero; the reported total lets a genuine zero be
	told from an unreported species by comparison with p_surf.
	"""
	totals = {'H': 0., 'C': 0., 'O': 0.}
	reported_total = 0.
	n_present = 0

	for col, stoich in REPORTED_SPECIES.items():
		if col not in row.index:
			continue
		val = row[col]
		if val is None or np.isnan(val):
			continue
		n_present += 1
		reported_total += val
		for el, nu in stoich.items():
			totals[el] += nu * val

	return totals, reported_total, n_present


def partial_pressure_from_mass(mass_kg, mass_total_kg, p_surf_bar,
	mmw_kg_mol, element):
	"""
	P_X = p_surf * (M_X / sum M) * (mmw / mu_X), in bar.

	Uses only reported quantities, so unlike the g/A form it cannot
	inherit an error from assumed planetary parameters. NaN if any
	ingredient is missing.
	"""
	mu_X = ATOMIC_MASS_KG_PER_MOL[element]
	if mass_kg is None or np.isnan(mass_kg):
		return 0.
	for v in (mass_total_kg, p_surf_bar, mmw_kg_mol):
		if v is None or np.isnan(v):
			return np.nan
	if mass_total_kg <= 0. or mmw_kg_mol <= 0.:
		return np.nan
	return p_surf_bar * (mass_kg / mass_total_kg) * (mmw_kg_mol / mu_X)


# -------------------------------------------------------------------------
# IW buffer deviation (mirrors compute_Delta_IW in MAGMAVOLlite_construct.py
# and check_chili.py)
# -------------------------------------------------------------------------
FO2_LOG_DEV_FALLBACK = 4.0   # used only if neither fO2_melt nor p_O2 is available

def compute_Delta_IW(T_K, PO2_bar, Ptot_bar):
	"""
	Deviation from the IW buffer (Delta_IW, log10 units), given fO2
	(PO2_bar, in bar), temperature, and total pressure.
	Wolf et al. (2023), footnote, VAPOROCK paper.
	Returns NaN if PO2_bar is NaN (cannot compute without it).
	"""
	if np.isnan(PO2_bar):
		return np.nan
	log10fO2IW = -28776.8/T_K + 14.057 + 0.055*(Ptot_bar-1)/T_K - 0.8853*np.log(T_K)
	Delta_IW = np.log10(PO2_bar) - log10fO2IW
	return Delta_IW


def extract_seed_from_evolution(model_name, planet, tau_yr):
	"""
	Read the evolutionary model's own output at the time step closest to
	tau_yr and return the seed for MAGMAVOL.

	Elemental pressures are monatomic-equivalent partial pressures in
	bar, matching what MAGMAVOL's targets are compared against. They come
	from the reported molecular partial pressures where those account for
	p_surf, and otherwise from the masses with p_surf and mmw. See the
	note above the helpers for why the earlier weight-per-area values
	were wrong and by how much.

	Returns T_surf, H, C and O in bar, the FO2_LOG_DEV target, the time
	actually used, and which routes supplied the fugacity and the
	pressures.
	"""
	csv_file = os.path.join(EVOL_DIR, model_name,
		f'evolution-{model_name}-{planet}-data.csv')
	if not os.path.exists(csv_file):
		raise FileNotFoundError(f"Evolution CSV not found: {csv_file}")

	df  = pd.read_csv(csv_file)
	idx = np.argmin(np.abs(df['t(yr)'].values - tau_yr))
	row = df.iloc[idx]

	def get_col(colname):
		"""Missing column or NaN value -> NaN (resolved to 0. for H/C/O below)."""
		return row[colname] if colname in row.index else np.nan

	t_used     = row['t(yr)']
	T_surf     = get_col('T_surf(K)')
	massC_atm  = get_col('massC_atm(kg)')
	massH_atm  = get_col('massH_atm(kg)')
	massO_atm  = get_col('massO_atm(kg)')
	p_surf     = get_col('p_surf(bar)')
	fO2_melt   = get_col('fO2_melt(bar)')
	p_O2_gas   = get_col('p_O2(bar)')

	mmw        = get_col('mmw(kg/mol)')

	params = PLANET_PARAMS[planet]
	g, Rp  = params['g'], params['Rp']

	# --- elemental partial pressures, by the two routes ---
	p_species, p_reported_total, n_species = elemental_pressures_from_species(row)

	masses = {'C': massC_atm, 'H': massH_atm, 'O': massO_atm}
	mass_total = sum(v for v in masses.values()
		if v is not None and not np.isnan(v))
	p_mass = {el: partial_pressure_from_mass(masses[el], mass_total, p_surf,
		mmw, el) for el in ('H', 'C', 'O')}

	# fraction of p_surf the reported molecular partials fail to cover.
	# Molecular partials must sum to p_surf by Dalton's law, so any
	# shortfall means species are missing from the report. (The
	# monatomic pressures do NOT sum to p_surf and cannot be used for
	# this test: for an H2O/CO2 atmosphere they total about 3*p_surf.)
	if p_surf is not None and not np.isnan(p_surf) and p_surf > 0.:
		unaccounted = 1. - p_reported_total / p_surf
	else:
		unaccounted = np.nan

	if (n_species > 0 and not np.isnan(unaccounted)
			and abs(unaccounted) <= SPECIES_COVERAGE_TOL
			and sum(p_species.values()) > 0.):
		pH_bar, pC_bar, pO_bar = p_species['H'], p_species['C'], p_species['O']
		p_source = 'species'
	else:
		pH_bar = p_mass['H'] if not np.isnan(p_mass['H']) else 0.
		pC_bar = p_mass['C'] if not np.isnan(p_mass['C']) else 0.
		pO_bar = p_mass['O'] if not np.isnan(p_mass['O']) else 0.
		p_source = 'mass'

	# The old weight-per-area values, kept only so the change is visible
	# in the log rather than silent.
	pH_old = mass_to_pressure_bar(massH_atm, g, Rp) if not np.isnan(massH_atm) else 0.
	pC_old = mass_to_pressure_bar(massC_atm, g, Rp) if not np.isnan(massC_atm) else 0.
	pO_old = mass_to_pressure_bar(massO_atm, g, Rp) if not np.isnan(massO_atm) else 0.

	log.info(f"  elemental pressures from '{p_source}' route "
		f"(reported species cover {1-unaccounted:.1%} of p_surf)")
	for el, new_val, old_val in [('H', pH_bar, pH_old), ('C', pC_bar, pC_old),
			('O', pO_bar, pO_old)]:
		factor = new_val/old_val if old_val > 0. else float('nan')
		log.info(f"    p{el} = {new_val:10.4f} bar "
			f"(was {old_val:9.4f} by weight per area, factor {factor:6.2f})")

	# Disagreement between the routes for oxygen flags an inconsistent
	# massO_atm. massO_atm counts all atmospheric oxygen and the species
	# sum only the reported volatiles, so the mass route should be the
	# larger of the two.
	if p_species['O'] > 0. and not np.isnan(p_mass['O']):
		o_ratio = p_mass['O'] / p_species['O']
		if o_ratio < 0.5 or o_ratio > 2.0:
			log.info(f"    WARNING: oxygen from masses is {o_ratio:.3g} times "
				f"that from the reported species; massO_atm for "
				f"'{model_name}' is inconsistent with its own partial "
				f"pressures")

	# FO2_LOG_DEV cascade: melt -> gas -> fixed fallback
	if not np.isnan(fO2_melt):
		fo2_log_dev = round(compute_Delta_IW(T_surf, fO2_melt, p_surf), 2)
		fo2_source  = 'melt'
	elif not np.isnan(p_O2_gas):
		fo2_log_dev = round(compute_Delta_IW(T_surf, p_O2_gas, p_surf), 2)
		fo2_source  = 'gas'
	else:
		fo2_log_dev = FO2_LOG_DEV_FALLBACK
		fo2_source  = 'fallback'

	return {
		'T_surf'      : T_surf,
		'H'           : round(pH_bar, 4),
		'C'           : round(pC_bar, 4),
		'O'           : round(pO_bar, 4),
		'FO2_LOG_DEV' : fo2_log_dev,
		't_used'      : t_used,
		'fo2_source'  : fo2_source,
		'p_source'    : p_source,
		'unaccounted' : unaccounted,
	}


# -------------------------------------------------------------------------
# MAGMAVOL run parameters (shared FO2/dissolution settings; H/C are
# planet-dependent and computed per-call via get_volatile_pressures)
# -------------------------------------------------------------------------
O    = 0.
N    = 0.
He   = 0.
P    = 0.
S    = 0.

FO2_CALCULATION = 'MASS_BALANCE'
FO2_BUFFER      = 'IW'
FO2_LOG_DEV     = 4.0   # IW+4 as per CHILI protocol

# Subfolder and model-name string used in output filenames/paths —
# derived from FO2_CALCULATION so both cases are clearly separated.
# Change FO2_CALCULATION above (or override in a variant script) and
# OUTPUTS_DIR/INPUTS_DIR/MODEL_NAME_STR update automatically.
def _fo2_subfolder(fo2_calc):
	return 'magmavol_massbalance' if fo2_calc == 'MASS_BALANCE' else 'magmavol_meltfug'

MODEL_NAME_STR = _fo2_subfolder(FO2_CALCULATION)
OUTPUTS_DIR    = os.path.join(CHILI_DIR, 'outputs', MODEL_NAME_STR)
INPUTS_DIR     = os.path.join(CHILI_DIR, 'inputs',  MODEL_NAME_STR)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR,  exist_ok=True)

magma_ocean_compo  = 'BSE'
H2O_dissolution    = False
CO2_dissolution    = False
SO2_dissolution    = False

# -------------------------------------------------------------------------
# Temperature grid — 1500 to 3800 K in 25 K steps, run high to low
# -------------------------------------------------------------------------
T_MIN    = 1500.
T_MAX    = 3800.
T_STEP   = 25.
t_arr    = np.arange(T_MIN, T_MAX + T_STEP, T_STEP)   # low to high (for saving)
t_arr_run = np.flip(t_arr)                              # high to low (for running)

# -------------------------------------------------------------------------
# Characteristic times and hot/cold model assignments per planet.
# Hot/cold models are chosen from the evolutionary ensemble as the
# highest/lowest T_surf at each tau.
# All planets share the same tau_yr values; only the models differ.
# For trappist1alpha tau6, no distinct cold model exists —
# both hot and cold are set to neongooey for consistency.
# earth/venus hot/cold model assignments are placeholders — fill in
# once the evolutionary model ensemble has been run for these cases.
# -------------------------------------------------------------------------
TAU_CONFIG = {
	'earth': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'planatmo',  'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'proteus',   'cold_model': 'pacman'},
		'tau5': {'tau_yr': 1e5, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
	},
	'venus': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'planatmo',  'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'proteus',   'cold_model': 'pacman'},
		'tau5': {'tau_yr': 1e5, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
	},
	'trappist1e': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'planatmo',  'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'proteus',   'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'moai'},
	},
	'trappist1b': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'planatmo'},
	},
	'trappist1alpha': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'gooey',     'cold_model': 'proteus'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'gooey',     'cold_model': 'proteus'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'gooey',     'cold_model': 'gooey'},
	},
}

VALID_PLANETS = list(TAU_CONFIG.keys())

# -------------------------------------------------------------------------
# CHILI static output columns
# -------------------------------------------------------------------------
CHILI_COLUMNS = ['z(m)', 'p_tot(bar)', 'T(K)',
	'p_H2O(bar)', 'p_CO2(bar)', 'p_CO(bar)',
	'p_H2(bar)', 'p_CH4(bar)', 'p_O2(bar)']

# Species name mapping: MAGMAVOL/FastChem name -> CHILI name
SPECIES_MAP = {
	'H2O' : 'p_H2O(bar)',
	'H2O1': 'p_H2O(bar)',
	'CO2' : 'p_CO2(bar)',
	'CO2'  : 'p_CO2(bar)',
	'C1O2': 'p_CO2(bar)',
	'CO'  : 'p_CO(bar)',
	'C1O1': 'p_CO(bar)',
	'H2'  : 'p_H2(bar)',
	'H2'  : 'p_H2(bar)',
	'CH4' : 'p_CH4(bar)',
	'C1H4': 'p_CH4(bar)',
	'O2'  : 'p_O2(bar)',
	'O2'  : 'p_O2(bar)',
}

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
log_path = os.path.join(OUTPUTS_DIR, 'run_chili_magmavol.log')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s  %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S',
	handlers=[
		logging.FileHandler(log_path, mode='a'),
		logging.StreamHandler()
	]
)
log = logging.getLogger()

# -------------------------------------------------------------------------
# Helper: read T_surf from evolution CSV at closest time to tau_yr
# -------------------------------------------------------------------------
def get_T_surf_at_tau(model_name, planet, tau_yr):
	"""Read T_surf from evolution CSV at closest time to tau_yr."""
	csv_file = os.path.join(EVOL_DIR, model_name,
		f'evolution-{model_name}-{planet}-data.csv')
	if not os.path.exists(csv_file):
		raise FileNotFoundError(f"Evolution CSV not found: {csv_file}")
	df  = pd.read_csv(csv_file)
	idx = np.argmin(np.abs(df['t(yr)'].values - tau_yr))
	row = df.iloc[idx]
	log.info(f"  {model_name}: closest t = {row['t(yr)']:.2f} yr, "
		f"T_surf = {row['T_surf(K)']:.2f} K")
	return float(row['T_surf(K)'])


# -------------------------------------------------------------------------
# Helper: build pkl filename (mirrors generatecompofile.py convention)
# -------------------------------------------------------------------------
def build_pkl_path(Tsurf, H, C, O=None, N=None, He=None, P=None, S=None,
	fO2_calculation=None, fO2_buffer=None, fO2_log_dev=None):
	"""
	Build the expected pkl file path matching MAGMAVOL_2's actual saved
	filename convention exactly.

	All parameters default to None, resolving to the corresponding
	module-level constant at call time (None-sentinel pattern).

	In TARGET_FO2 mode: appends _{fO2_calculation}_{fO2_buffer}_{fO2_log_dev}
	to the filename, matching MAGMAVOL_2's pkl save logic.
	In MASS_BALANCE mode: omits that suffix entirely, also matching MAGMAVOL_2.
	fO2_buffer/fO2_log_dev are silently ignored in MASS_BALANCE mode.
	"""
	if O               is None: O               = globals()['O']
	if N               is None: N               = globals()['N']
	if He              is None: He              = globals()['He']
	if P               is None: P               = globals()['P']
	if S               is None: S               = globals()['S']
	if fO2_calculation is None: fO2_calculation = FO2_CALCULATION
	if fO2_buffer      is None: fO2_buffer      = FO2_BUFFER
	if fO2_log_dev     is None: fO2_log_dev     = FO2_LOG_DEV

	initial_atmo_dict = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S}
	target_atmo_dict  = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S}

	file = f'atmo_sol_T={Tsurf}K_target_atm='
	for k, v in target_atmo_dict.items():
		file += f'_{k}={v}'
	file += '_init_atm='
	for k, v in initial_atmo_dict.items():
		file += f'_{k}={v}'
	file += f'_magma_ocean_compo={magma_ocean_compo}'
	file += f'_H2O_dissolve={H2O_dissolution}'
	file += f'_CO2_dissolve={CO2_dissolution}'

	if fO2_calculation == 'TARGET_FO2':
		file += f'_{fO2_calculation}_{fO2_buffer}_{fO2_log_dev}'

	return os.path.join(
		mag.MAIN_DIR,
		mag.DIR_FOR_RESERVOIR_OF_ATMO_SOLUTIONS,
		file + '.pkl'
	)


def get_compo_file_path(planet):
	"""Build the compo .dat filename for a given planet's volatile pressures.

	The filename is keyed on pH/pC (not the planet name directly), since
	pH/pC alone already disambiguate planets with different g/Rp. Planets
	sharing identical g/Rp (trappist1e/b/alpha) therefore share a file.
	"""
	pH_bar, pC_bar = get_volatile_pressures(planet)
	return os.path.join(MAGMAVOL_DIR,
		f'compo_chili_pH={pH_bar:.2f}_pC={pC_bar:.2f}_fug={FO2_LOG_DEV}_new.dat')


# -------------------------------------------------------------------------
# Helper: extract CHILI partial pressures from MAGMAVOL output
# -------------------------------------------------------------------------
def extract_chili_pressures(mol_pressures):
	"""
	Extract CHILI-required partial pressures from MAGMAVOL
	molecular_composition_pressures dict.
	Returns dict with CHILI column names as keys.
	"""
	out = {col: np.nan for col in CHILI_COLUMNS}
	p_tot = 0.

	for sp, p in mol_pressures.items():
		chili_key = SPECIES_MAP.get(sp, None)
		if chili_key is not None:
			# take the max if there are duplicate mappings
			if np.isnan(out[chili_key]) or p > out[chili_key]:
				out[chili_key] = p
		p_tot += p

	out['p_tot(bar)'] = p_tot
	return out


# -------------------------------------------------------------------------
# STEP 1: Run MAGMAVOL at a single test temperature (hottest at tau3)
#         and print output to check molecular_composition_pressures keys
# -------------------------------------------------------------------------
def run_chili_single(planet='trappist1e', tau_key='tau3', hot_or_cold='hot',
	use_seeded_O=True, fO2_calculation=None):
	"""
	Run MAGMAVOL at the surface temperature of the hot or cold evolutionary
	model for the given planet and tau, seeding H/C/O (bar) and the FO2_LOG_DEV
	target from that SAME model's own row at that tau — NOT from the fixed
	Table 2/4 massH_kg/massC_kg values get_volatile_pressures() uses.

	This represents the atmosphere AS ALREADY ESTABLISHED at that tau (no
	further exsolution): initial_atmo_dict['H']==target_atmo_dict['H'],
	likewise for C; O appears only in initial_atmo_dict (fO2 is governed
	separately via FO2_CALCULATION/FO2_BUFFER/FO2_LOG_DEV, not a target-O
	mass-balance parameter). N/He/P/S remain fixed at 0, as in the
	original Table 2/4 convention.

	If MAGMAVOL does not converge (data['success'] is False), this logs
	the residual variance and does NOT save a CSV — a non-converged
	result is not a usable CHILI data point.

	planet         : one of 'earth', 'venus', 'trappist1e', 'trappist1b', 'trappist1alpha'
	tau_key        : a key in TAU_CONFIG[planet], e.g. 'tau3', 'tau4', 'tau5', 'tau6'
	hot_or_cold    : 'hot' or 'cold'
	use_seeded_O   : if True (default), O is seeded from the evolutionary
	                 model's massO_atm(kg). If False, O is forced to 0.
	fO2_calculation: 'TARGET_FO2' or 'MASS_BALANCE'. Defaults to None,
	                 resolving to the module-level FO2_CALCULATION at call time.
	use_seeded_O   : if True (default), O is seeded from the evolutionary
	                 model's massO_atm(kg). If False, O is forced to 0.
	"""
	if fO2_calculation is None:
		fO2_calculation = FO2_CALCULATION
	if planet not in VALID_PLANETS:
		raise ValueError(f"Unknown planet '{planet}'. Choose from: {VALID_PLANETS}")

	cfg        = TAU_CONFIG[planet][tau_key]
	tau_yr     = cfg['tau_yr']
	model_name = cfg['hot_model'] if hot_or_cold == 'hot' else cfg['cold_model']

	if model_name is None:
		raise ValueError(
			f"No {hot_or_cold}_model defined yet for planet='{planet}', "
			f"tau_key='{tau_key}'. Fill in TAU_CONFIG['{planet}']['{tau_key}'] first."
		)

	log.info(f"=== CHILI single run: {planet} {tau_key} ({hot_or_cold}) "
		f"(tau = {tau_yr:.0e} yr), seeded from model '{model_name}' ===")

	seed   = extract_seed_from_evolution(model_name, planet, tau_yr)
	H, C, O = seed['H'], seed['C'], seed['O']
	T_surf  = seed['T_surf']
	fo2_dev = seed['FO2_LOG_DEV']

	if not use_seeded_O:
		log.info(f"  use_seeded_O=False: overriding seeded O = {O} bar with O = 0.")
		O = 0.

	log.info(f"  Seed row: t_used = {seed['t_used']:.2f} yr "
		f"(target tau = {tau_yr:.0e} yr)")
	log.info(f"  Seeded volatile inputs: H = {H} bar, C = {C} bar, O = {O} bar")
	log.info(f"  Seeded FO2_LOG_DEV = {fo2_dev}  (source: {seed['fo2_source']})")
	log.info(f"  T_surf ({hot_or_cold}) = {T_surf:.2f} K")

	# run MAGMAVOL at T_surf, with H/C/O/FO2_LOG_DEV seeded from the
	# selected model's row rather than the fixed global table values
	log.info(f"Running MAGMAVOL at T = {T_surf:.2f} K ...")
	data = mag.atmo_compo_magma_ocean_with_dissolution_AND_initial_atmosphere_ATMO_Target_atmosphere(
		Tsurf              = T_surf,
		initial_atmo_dict  = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
		target_atmo_dict   = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
		init_sol           = -1,
		H2O_dissolution    = H2O_dissolution,
		CO2_dissolution    = CO2_dissolution,
		SO2_dissolution    = SO2_dissolution,
		chemistry_module   = 'FASTCHEM',
		magma_ocean_compo  = magma_ocean_compo,
		magma_evaporation_model = 'VAPOROCK',
		save_data_in_solutions_reservoir = True,
		FO2_CALCULATION    = fO2_calculation,
		FO2_BUFFER         = FO2_BUFFER,
		FO2_LOG_DEV        = fo2_dev,
	)

	# check convergence before doing anything with the result — a
	# non-converged solution is not a usable CHILI data point
	if not data.get('success', False):
		res_fun = data.get('result_minimize', None)
		res_fun_val = res_fun.fun if res_fun is not None else float('nan')
		log.info(f"NOT CONVERGED for {planet} {tau_key} ({hot_or_cold}): "
			f"residual variance = {res_fun_val:.6e}. CSV NOT saved.")
		return data

	# print molecular_composition_pressures to check keys
	log.info("molecular_composition_pressures keys and values:")
	mol_pressures = data['molecular_composition_pressures']
	for sp, p in sorted(mol_pressures.items(), key=lambda x: -x[1]):
		log.info(f"  {sp:20s} = {p:.6e} bar")

	# extract CHILI pressures
	chili_vals         = extract_chili_pressures(mol_pressures)
	chili_vals['z(m)'] = 0.
	chili_vals['T(K)'] = T_surf

	# save single-row CHILI CSV — filename and folder reflect the fO2 mode
	model_str = _fo2_subfolder(fO2_calculation)
	out_dir   = os.path.join(CHILI_DIR, 'outputs', model_str)
	in_dir    = os.path.join(CHILI_DIR, 'inputs',  model_str)
	os.makedirs(out_dir, exist_ok=True)
	os.makedirs(in_dir,  exist_ok=True)

	out_csv = os.path.join(out_dir,
		f'static-{model_str}-{planet}-{tau_key}-{hot_or_cold}-data.csv')
	df_out  = pd.DataFrame([chili_vals])[CHILI_COLUMNS]
	df_out.to_csv(out_csv, index=False, na_rep='NaN')
	log.info(f"Saved CHILI static CSV to: {out_csv}")
	print(df_out.to_string())

	# copy this runner script to inputs folder
	runner_dst = os.path.join(in_dir,
		f'static-{model_str}-{planet}-{tau_key}-{hot_or_cold}-config.py')
	shutil.copy(__file__, runner_dst)
	log.info(f"Copied runner script to inputs: {runner_dst}")

	return data


# -------------------------------------------------------------------------
# STEP 2a: Interpolate missing temperatures in an existing compo .dat file
#          Can be called standalone on a pre-generated file, or automatically
#          from run_full_grid() after writing the initial compo file.
# -------------------------------------------------------------------------
def interpolate_compo_file(compo_file=None, planet=None):
	"""
	Interpolate missing temperature rows in a compo .dat file and overwrite it.

	Two modes:
	  - Called with compo_file path: load the file, detect temperatures missing
	    from the expected t_arr grid, interpolate from present rows using a
	    cubic UnivariateSpline, and overwrite.
	  - Called with planet (and no compo_file): derives the default compo
	    filename for that planet via get_compo_file_path().

	Skips interpolation if fewer than half the expected temperatures are present.
	"""
	header  = ['# T', 'Ptot', 'H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si',
		'Ar', 'Ti', 'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr',
		'Li', 'Cs', 'Rb', 'F', 'P']

	if compo_file is None:
		if planet is None:
			raise ValueError("Provide either compo_file or planet.")
		compo_file = get_compo_file_path(planet)

	if not os.path.exists(compo_file):
		log.info(f"interpolate_compo_file: file not found: {compo_file}")
		return

	# load existing file — skip the header row (comments='')
	existing = np.loadtxt(compo_file, comments='#')
	if existing.ndim == 1:
		existing = existing[np.newaxis, :]

	T_present = existing[:, 0]
	n_cols    = existing.shape[1]

	# find which temperatures from the expected grid are missing
	missing_temps = [T for T in t_arr if not np.any(np.isclose(T_present, T))]
	n_present     = len(T_present)
	n_expected    = len(t_arr)

	log.info(f"interpolate_compo_file: {n_present}/{n_expected} temperatures present "
		f"in {os.path.basename(compo_file)}")

	if not missing_temps:
		log.info("interpolate_compo_file: no missing temperatures — nothing to do.")
		return

	# threshold: need at least n/2 present to interpolate reliably
	if n_present < n_expected / 2:
		log.info(f"interpolate_compo_file: too few rows ({n_present}/{n_expected}) "
			f"to interpolate reliably. Skipping.")
		return

	log.info(f"interpolate_compo_file: interpolating {len(missing_temps)} missing "
		f"temperature(s) using cubic UnivariateSpline over {n_present} present points.")

	# fit a spline per column (Ptot + each species) over present temperatures
	T_sorted = np.array(sorted(T_present))
	splines  = {}
	for col_idx in range(1, n_cols):   # skip col 0 (T itself)
		y = existing[np.argsort(T_present), col_idx]
		splines[col_idx] = UnivariateSpline(
			T_sorted, y, k=min(3, n_present - 1), s=0
		)

	# build interpolated rows for missing temperatures
	interp_rows = []
	for T in missing_temps:
		row    = np.zeros(n_cols)
		row[0] = T
		for col_idx in range(1, n_cols):
			row[col_idx] = float(splines[col_idx](T))
		interp_rows.append(row)
		log.info(f"  Interpolated T={T:.0f} K")

	# merge existing and interpolated rows, sort by temperature
	all_rows = np.vstack([existing, np.array(interp_rows)])
	all_rows = all_rows[np.argsort(all_rows[:, 0])]

	np.savetxt(compo_file, all_rows, delimiter=' ',
		header=' '.join(header), comments='')
	log.info(f"interpolate_compo_file: overwrote {compo_file} "
		f"({len(all_rows)} rows total).")


# -------------------------------------------------------------------------
# STEP 1b: Run full temperature grid seeded from evolutionary model inputs
# -------------------------------------------------------------------------
def run_chili_grid(planet, tau_key, hot_or_cold,
	fO2_calculation=None, use_seeded_O=True):
	"""
	Run MAGMAVOL across the full temperature grid (t_arr_run) for a
	specific (planet, tau_key, hot_or_cold) combination, seeding H/C/O
	(bar) and FO2_LOG_DEV from the selected evolutionary model's row at
	that tau — exactly as run_chili_single does, but sweeping over the
	full temperature grid rather than a single T_surf.

	On each temperature point:
	  - If a pkl already exists in the reservoir, skip.
	  - If MAGMAVOL does not converge (pkl not saved), log a warning.
	  - MAGMAVOL_2 handles convergence/reservoir saving internally.

	After the full sweep, writes a compo .dat file and calls
	interpolate_compo_file() automatically to fill any missing rows.

	planet         : one of 'earth', 'venus', 'trappist1e', 'trappist1b', 'trappist1alpha'
	tau_key        : a key in TAU_CONFIG[planet], e.g. 'tau3', 'tau4', 'tau5', 'tau6'
	hot_or_cold    : 'hot' or 'cold'
	use_seeded_O   : if True (default), O is seeded from the evolutionary
	                 model's massO_atm(kg). If False, O is forced to 0.
	fO2_calculation: 'TARGET_FO2' or 'MASS_BALANCE'. Defaults to None,
	                 resolving to the module-level FO2_CALCULATION at call time.
	"""
	if fO2_calculation is None:
		fO2_calculation = FO2_CALCULATION
	if planet not in VALID_PLANETS:
		raise ValueError(f"Unknown planet '{planet}'. Choose from: {VALID_PLANETS}")

	cfg_tau    = TAU_CONFIG[planet][tau_key]
	tau_yr     = cfg_tau['tau_yr']
	model_name = cfg_tau['hot_model'] if hot_or_cold == 'hot' else cfg_tau['cold_model']

	if model_name is None:
		raise ValueError(
			f"No {hot_or_cold}_model defined yet for planet='{planet}', "
			f"tau_key='{tau_key}'. Fill in TAU_CONFIG['{planet}']['{tau_key}'] first."
		)

	seed    = extract_seed_from_evolution(model_name, planet, tau_yr)
	H, C, O = seed['H'], seed['C'], seed['O']
	fo2_dev  = seed['FO2_LOG_DEV']

	if not use_seeded_O:
		log.info(f"  use_seeded_O=False: overriding seeded O = {O} bar with O = 0.")
		O = 0.

	log.info(f"=== CHILI grid run: {planet} {tau_key} ({hot_or_cold}), "
		f"seeded from model '{model_name}' ===")
	log.info(f"  Seed row: t_used = {seed['t_used']:.2f} yr "
		f"(target tau = {tau_yr:.0e} yr)")
	log.info(f"  Seeded volatile inputs: H = {H} bar, C = {C} bar, O = {O} bar")
	log.info(f"  Seeded FO2_LOG_DEV = {fo2_dev}  (source: {seed['fo2_source']})")
	log.info(f"  FO2_CALCULATION = {fO2_calculation}")
	log.info(f"  T range: {T_MIN} to {T_MAX} K in {T_STEP} K steps "
		f"({len(t_arr)} points)")
	log.info("  Running high to low temperature")

	for Tsurf in t_arr_run:
		pkl_path = build_pkl_path(Tsurf, H, C, O=O, N=N, He=He, P=P, S=S,
			fO2_calculation=fO2_calculation, fO2_buffer=FO2_BUFFER,
			fO2_log_dev=fo2_dev)
		if os.path.exists(pkl_path):
			log.info(f"  T={Tsurf:.0f} K: pkl already exists, skipping.")
			continue

		log.info(f"  T={Tsurf:.0f} K: running MAGMAVOL ...")
		mag.atmo_compo_magma_ocean_with_dissolution_AND_initial_atmosphere_ATMO_Target_atmosphere(
			Tsurf              = Tsurf,
			initial_atmo_dict  = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			target_atmo_dict   = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			init_sol           = -1,
			H2O_dissolution    = H2O_dissolution,
			CO2_dissolution    = CO2_dissolution,
			SO2_dissolution    = SO2_dissolution,
			chemistry_module   = 'FASTCHEM',
			magma_ocean_compo  = magma_ocean_compo,
			magma_evaporation_model = 'VAPOROCK',
			save_data_in_solutions_reservoir = True,
			FO2_CALCULATION    = fO2_calculation,
			FO2_BUFFER         = FO2_BUFFER,
			FO2_LOG_DEV        = fo2_dev,
		)

		if not os.path.exists(pkl_path):
			log.info(f"NOT CONVERGED at T={Tsurf:.0f} K")

	# --- write compo file ---
	log.info("Writing compo file ...")
	header  = ['# T', 'Ptot', 'H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si',
		'Ar', 'Ti', 'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr',
		'Li', 'Cs', 'Rb', 'F', 'P']
	species = ['H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si', 'Ar', 'Ti',
		'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr', 'Li', 'Cs',
		'Rb', 'F', 'P']

	arr_list      = []
	ptot          = []
	present_temps = []

	for Tsurf in t_arr:
		pkl_path = build_pkl_path(Tsurf, H, C, O=O, N=N, He=He, P=P, S=S,
			fO2_calculation=fO2_calculation, fO2_buffer=FO2_BUFFER,
			fO2_log_dev=fo2_dev)
		if not os.path.exists(pkl_path):
			log.info(f"  WARNING: pkl missing for T={Tsurf:.0f} K, skipping.")
			continue
		with open(pkl_path, 'rb') as f:
			data = pickle.load(f)
			arr_list.append(data['atoms_molar_fractions'])
			ptot.append(data['Ptot_gas_bar'])
			present_temps.append(Tsurf)

	if not arr_list:
		log.info("ERROR: no pkl files found. Run the grid first.")
		return

	values  = np.zeros((len(present_temps), len(header)))
	sp_list = list(data['atoms_molar_fractions'].keys())

	for num, (T, p, mf) in enumerate(zip(present_temps, ptot, arr_list)):
		sp_temp = np.zeros(len(species))
		for sp_num, sp in enumerate(species):
			check = [i for i, s in enumerate(sp_list) if s == sp]
			val_temp = list(mf.values())
			if check:
				sp_temp[sp_num] = val_temp[check[0]]
		values[num, 2:] = sp_temp
		values[num, 1]  = p
		values[num, 0]  = T

	model_str = _fo2_subfolder(fO2_calculation)
	if fO2_calculation == 'MASS_BALANCE':
		compo_file = os.path.join(
			CHILI_DIR,
			f'compo_chili_pH={H}_pC={C}_pO={O}'
			f'_{model_name}_{tau_key}_{hot_or_cold}_mass_balance_new.dat'
		)
	else:
		compo_file = os.path.join(
			CHILI_DIR,
			f'compo_chili_pH={H}_pC={C}_pO={O}_fug={fo2_dev}'
			f'_{model_name}_{tau_key}_{hot_or_cold}_new.dat'
		)

	np.savetxt(compo_file, values, delimiter=' ',
		header=' '.join(header), comments='')
	log.info(f"Saved compo file to: {compo_file}")

	interpolate_compo_file(compo_file)


# -------------------------------------------------------------------------
# STEP 2: Run full temperature grid and write compo file for a given planet
#         (call after verifying single test output)
# -------------------------------------------------------------------------
def run_full_grid(planet):
	"""
	Run MAGMAVOL across the full temperature grid for the given planet's
	g/Rp and H/C, and write a per-planet compo .dat file.

	planet: one of 'earth', 'venus', 'trappist1e', 'trappist1b', 'trappist1alpha'.
	        trappist1e/b/alpha share identical g/Rp (Earth-homogenised) and
	        therefore produce an identical compo file; venus and earth have
	        their own distinct g/Rp and therefore their own compo file.
	"""
	if planet not in VALID_PLANETS:
		raise ValueError(f"Unknown planet '{planet}'. Choose from: {VALID_PLANETS}")

	pH_bar, pC_bar = get_volatile_pressures(planet)
	H = round(pH_bar, 2)
	C = round(pC_bar, 2)

	log.info(f"=== Running full temperature grid for planet='{planet}' ===")
	log.info(f"  Volatile inputs: pH = {pH_bar:.4f} bar, pC = {pC_bar:.4f} bar")
	log.info(f"  T range: {T_MIN} to {T_MAX} K in {T_STEP} K steps "
		f"({len(t_arr)} points)")
	log.info(f"  Running high to low temperature")

	for Tsurf in t_arr_run:
		pkl_path = build_pkl_path(Tsurf, H, C, O=O, N=N, He=He, P=P, S=S,
			fO2_calculation=FO2_CALCULATION, fO2_buffer=FO2_BUFFER, fO2_log_dev=FO2_LOG_DEV)
		if os.path.exists(pkl_path):
			log.info(f"  T={Tsurf:.0f} K: pkl already exists, skipping.")
			continue
		log.info(f"  T={Tsurf:.0f} K: running MAGMAVOL ...")
		mag.atmo_compo_magma_ocean_with_dissolution_AND_initial_atmosphere_ATMO_Target_atmosphere(
			Tsurf              = Tsurf,
			initial_atmo_dict  = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			target_atmo_dict   = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			init_sol           = -1,
			H2O_dissolution    = H2O_dissolution,
			CO2_dissolution    = CO2_dissolution,
			SO2_dissolution    = SO2_dissolution,
			chemistry_module   = 'FASTCHEM',
			magma_ocean_compo  = magma_ocean_compo,
			magma_evaporation_model = 'VAPOROCK',
			save_data_in_solutions_reservoir = True,
			FO2_CALCULATION    = FO2_CALCULATION,
			FO2_BUFFER         = FO2_BUFFER,
			FO2_LOG_DEV        = FO2_LOG_DEV,
		)

	# --- write compo file ---
	log.info("Writing compo file ...")
	header  = ['# T', 'Ptot', 'H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si',
		'Ar', 'Ti', 'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr',
		'Li', 'Cs', 'Rb', 'F', 'P']
	species = ['H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si', 'Ar', 'Ti',
		'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr', 'Li', 'Cs',
		'Rb', 'F', 'P']

	arr_list      = []
	ptot          = []
	present_temps = []

	for Tsurf in t_arr:
		pkl_path = build_pkl_path(Tsurf, H, C, O=O, N=N, He=He, P=P, S=S,
			fO2_calculation=FO2_CALCULATION, fO2_buffer=FO2_BUFFER, fO2_log_dev=FO2_LOG_DEV)
		if not os.path.exists(pkl_path):
			log.info(f"  WARNING: pkl missing for T={Tsurf:.0f} K, skipping.")
			continue
		with open(pkl_path, 'rb') as f:
			data = pickle.load(f)
			arr_list.append(data['atoms_molar_fractions'])
			ptot.append(data['Ptot_gas_bar'])
			present_temps.append(Tsurf)

	if not arr_list:
		log.info("ERROR: no pkl files found. Run the grid first.")
		return

	index    = np.array(list(arr_list[0].keys()))
	sp_temp  = np.zeros(len(species))
	values   = np.zeros((len(present_temps), len(header)))

	for num, (T, molar, p) in enumerate(zip(present_temps, arr_list, ptot)):
		val_temp = np.array(list(molar.values()))
		for sp_num, sp in enumerate(species):
			check = np.where(index == sp)[0]
			if check.size == 1:
				sp_temp[sp_num] = val_temp[check[0]]
		values[num, 2:] = sp_temp
		values[num, 1]  = p
		values[num, 0]  = T

	compo_file = get_compo_file_path(planet)
	np.savetxt(compo_file, values, delimiter=' ',
		header=' '.join(header), comments='')
	log.info(f"Saved compo file to: {compo_file}")

	# interpolate any missing temperatures and overwrite
	interpolate_compo_file(compo_file)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
	# -------------------------------------------------------------------------
	# Step 1: run single-temperature z=0 cases.
	# Uncomment the planet/tau/hot_or_cold combination you want to run.
	# -------------------------------------------------------------------------
	# --- trappist1e ---
	data = run_chili_single(planet='trappist1e', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1e', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1e', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1e', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1e', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1e', tau_key='tau6', hot_or_cold='cold', use_seeded_O=True)

	# --- trappist1b ---
	data = run_chili_single(planet='trappist1b', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1b', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1b', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1b', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1b', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1b', tau_key='tau6', hot_or_cold='cold', use_seeded_O=True)

	# --- trappist1alpha ---
	data = run_chili_single(planet='trappist1alpha', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1alpha', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1alpha', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='trappist1alpha', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='trappist1alpha', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)

	# --- earth ---
	data = run_chili_single(planet='earth', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='earth', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='earth', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='earth', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='earth', tau_key='tau5', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='earth', tau_key='tau5', hot_or_cold='cold', use_seeded_O=True)

	# --- venus ---
	data = run_chili_single(planet='venus', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='venus', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='venus', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='venus', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	data = run_chili_single(planet='venus', tau_key='tau5', hot_or_cold='hot', use_seeded_O=True)
	data = run_chili_single(planet='venus', tau_key='tau5', hot_or_cold='cold', use_seeded_O=True)

	# -------------------------------------------------------------------------
	# Step 1b: run seeded temperature grid and write a compo file.
	# Each (planet, tau_key, hot_or_cold) produces its own compo file.
	# Interpolation of missing temperatures runs automatically at the end.
	# -------------------------------------------------------------------------
	# --- trappist1e ---
	# run_chili_grid(planet='trappist1e', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1e', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1e', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1e', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1e', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1e', tau_key='tau6', hot_or_cold='cold', use_seeded_O=True)

	# --- trappist1b ---
	# run_chili_grid(planet='trappist1b', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1b', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1b', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1b', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1b', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1b', tau_key='tau6', hot_or_cold='cold', use_seeded_O=True)

	# --- trappist1alpha ---
	# run_chili_grid(planet='trappist1alpha', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1alpha', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1alpha', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='trappist1alpha', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='trappist1alpha', tau_key='tau6', hot_or_cold='hot', use_seeded_O=True)

	# --- earth ---
	# run_chili_grid(planet='earth', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='earth', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='earth', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='earth', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='earth', tau_key='tau5', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='earth', tau_key='tau5', hot_or_cold='cold', use_seeded_O=True)

	# --- venus ---
	# run_chili_grid(planet='venus', tau_key='tau3', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='venus', tau_key='tau3', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='venus', tau_key='tau4', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='venus', tau_key='tau4', hot_or_cold='cold', use_seeded_O=True)
	# run_chili_grid(planet='venus', tau_key='tau5', hot_or_cold='hot', use_seeded_O=True)
	# run_chili_grid(planet='venus', tau_key='tau5', hot_or_cold='cold', use_seeded_O=True)

	# -------------------------------------------------------------------------
	# Step 2: run fixed-input temperature grid (Table 2/4 H/C, planet-only).
	# -------------------------------------------------------------------------
	# run_full_grid(planet='trappist1e')
	# run_full_grid(planet='venus')
	# run_full_grid(planet='earth')
	pass
