# ============================================================
# EV Bus Charging Scheduling (Weekly) - Pyomo Model
# Delta t = 15 minutes; 2 terminals: A (Habana), B (Pinar del Rio)
# Rule: V1 is always A->B, V2 always B->A
# When bus is not in route, it is at A, except between V1 arrival and V2 departure (at B)
# After finishing V2, it is at A (for the rest of the time)
# ============================================================

from __future__ import annotations

import math
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional

import pyomo.environ as pyo
from pyomo.environ import value
from pyomo_windows.solvers import SolverManager
from pyomo.core.expr.visitor import identify_variables

from utils import save_solution_json


# -----------------------------
# Helpers: time parsing & mapping
# -----------------------------
DT_MIN = 15
SLOTS_PER_DAY = 24 * 60 // DT_MIN  # 96

NDAYS = 14  # 2 semanas
DAYS = list(range(NDAYS))  # 0..13

SLOTS_PER_HORIZON = NDAYS * SLOTS_PER_DAY  # 1344
WEEK_SLOTS = 7 * SLOTS_PER_DAY  # 672 (para wrap semanal)

BASE_FREQ = {
    "D": {0,1,2,3,4,5,6},  # daily
    "MV": {0,3},          # Tue & Fri
    "JD": {2,5},          # Thu & Sun
}

def expand_freq_days(freq: str, NDAYS: int = 14):
    base = BASE_FREQ[freq]
    out = set()
    n_weeks = NDAYS // 7
    for w in range(n_weeks):
        out |= {d + 7*w for d in base}
    return out

def parse_hhmm(x) -> Tuple[int, int]:
    """
    Parse a time cell that might be:
      - pandas Timestamp / datetime.time
      - string "HH:MM"
      - csv float (fraction of a day)
    Returns (hour, minute).
    """
    if pd.isna(x):
        raise ValueError("Time cell is NaN")

    # Timestamp
    if isinstance(x, pd.Timestamp):
        return int(x.hour), int(x.minute)

    # datetime.time
    if hasattr(x, "hour") and hasattr(x, "minute") and not isinstance(x, str):
        return int(x.hour), int(x.minute)

    # csv float (fraction of day)
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        # csv time fraction: 0.5 = 12:00
        minutes = int(round(float(x) * 24 * 60))
        minutes %= (24 * 60)
        return minutes // 60, minutes % 60

    # String
    s = str(x).strip()
    # allow "HH:MM" or "H:MM"
    parts = s.split(":")
    if len(parts) >= 2:
        hh = int(parts[0])
        mm = int(parts[1])
        return hh, mm

    raise ValueError(f"Unrecognized time format: {x!r}")


def minutes_from_midnight(x) -> int:
    hh, mm = parse_hhmm(x)
    return 60 * hh + mm


def abs_minutes(day_index: int, hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    return day_index * 24 * 60 + hh * 60 + mm


def abs_arrival_minutes(day_index: int, dep_hhmm: str, arr_hhmm: str) -> int:
    """Arrival absolute minutes; handles crossing midnight."""
    dep = abs_minutes(day_index, dep_hhmm)
    arr = abs_minutes(day_index, arr_hhmm)
    if arr < dep:  # cruza medianoche
        arr += 24 * 60
    return arr


def slot_index_from_abs_minute(abs_min: int) -> int:
    """Map absolute minute to slot index h (0..1334)."""
    return abs_min // DT_MIN

def slot_interval_abs_minutes(h: int):
    a0 = h * DT_MIN
    a1 = (h + 1) * DT_MIN
    return a0, a1

def slot_index_from_abs_minute(abs_minute: int) -> int:
    """Return (start_min, end_min) absolute minutes for slot h."""
    return int(abs_minute // DT_MIN)


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Return True if [a0,a1) overlaps [b0,b1)."""
    return max(a0, b0) < min(a1, b1)


# -----------------------------
# Data structure for each bus schedule
# -----------------------------
@dataclass(frozen=True)
class BusSchedule:
    bus_id: int
    freq: str
    dep_v1: any
    arr_v1: any
    dep_v2: any
    arr_v2: any


def load_schedules_from_csv(path: str, sheet_name: Optional[str] = None) -> List[BusSchedule]:
    df = pd.read_csv(path)

    # Try to guess column names (adjust here if your sheet differs)
    # Expected: ri, Salida_V1, Llegada_V1, Salida_V2, Llegada_V2, Frecuencia
    col_map_candidates = {
        "bus_id": ["bus_id"],
        "dep_v1": ["dep_v1"],
        "arr_v1": ["arr_v1"],
        "dep_v2": ["dep_v2"],
        "arr_v2": ["arr_v2"],
        "freq":   ["freq"],
    }

    def pick_col(name: str) -> str:
        for c in col_map_candidates[name]:
            if c in df.columns:
                return c
        raise KeyError(f"Could not find a column for {name}. Available columns: {list(df.columns)}")

    c_bus = pick_col("bus_id")
    c_d1 = pick_col("dep_v1")
    c_a1 = pick_col("arr_v1")
    c_d2 = pick_col("dep_v2")
    c_a2 = pick_col("arr_v2")
    c_fr = pick_col("freq")

    schedules = []
    for _, row in df.iterrows():
        bus_id = int(row[c_bus])
        freq = str(row[c_fr]).strip()
        if freq not in BASE_FREQ:
            raise ValueError(f"Unknown Frecuencia {freq!r} for bus {bus_id}. Expected one of {list(BASE_FREQ.keys())}")
        schedules.append(
            BusSchedule(
                bus_id=bus_id,
                freq=freq,
                dep_v1=row[c_d1],
                arr_v1=row[c_a1],
                dep_v2=row[c_d2],
                arr_v2=row[c_a2],
            )
        )
    return schedules

def load_consumption_from_csv(path: str) -> Dict[Tuple[int, int, str], float]:
    df = pd.read_csv(path)
    Etrip_sm = {}
    for _, row in df.iterrows():
        bus_id = int(row["id"])
        for m in row.index:
            if m == "id" or m == "km":
                continue
            energy = float(row[m])
            Etrip_sm[(bus_id, 1, m)] = energy  # assuming leg 1 for all entries; adjust as needed
            Etrip_sm[(bus_id, 2, m)] = energy  # assuming same energy for leg 2; adjust as needed
    return Etrip_sm


# -----------------------------
# Build weekly parameters: alpha, L, consumption profile
# -----------------------------
def build_horizon_parameters(
    schedules: List[BusSchedule],
    charger_types: List[str],
    P_kw: Dict[str, float],
    eta: Dict[str, float],
    SoC_min_m: Dict[str, float],
    battery_configs_kwh: Dict[str, float],
    # Energy per trip (kWh) for each bus leg; fill these when you have them
    Etrip_sm: Optional[Dict[Tuple[int,int,str], float]] = None,
) -> Dict:
    """
    Returns a dict with:
      K: list of bus ids
      T: ["A","B"]
      C: charger types
      H: list(range(672))
      alpha[(k, day)] = 1/0 if that bus operates on that day (i.e., its pair V1/V2 occurs)
      L[(k, t, h)] = 1/0 availability to charge at terminal t in slot h
      cons[(k,h)] = kWh consumed in slot h (weekly absolute time)
      first_dep_slot[(k, day)] = slot index for V1 departure on that day (if operates)
      w_config list M, B[m]
      plus P, eta, dt
    """
    K = sorted({s.bus_id for s in schedules})
    T = ["A", "B"]
    C = list(charger_types)
    H = list(range(SLOTS_PER_HORIZON))
    M = list(battery_configs_kwh.keys())

    # alpha: bus operates on day?
    alpha = {(k, day): 0 for k in K for day in DAYS}
    sched_by_k = {s.bus_id: s for s in schedules}
    for k in K:
        freq = sched_by_k[k].freq
        for day in expand_freq_days(freq, NDAYS=NDAYS):
            alpha[(k, day)] = 1

    # Build route intervals per (k, day): absolute minutes for V1 and V2
    v1_dep_abs = {}
    v1_arr_abs = {}
    v2_dep_abs = {}
    v2_arr_abs = {}
    first_dep_slot = {}  # (k, day) -> slot index h where V1 dep occurs
    for k in K:
        sch = sched_by_k[k]
        for day in DAYS:
            if alpha[(k, day)] == 0:
                continue
            dep1 = abs_minutes(day, sch.dep_v1)
            arr1 = abs_arrival_minutes(day, sch.dep_v1, sch.arr_v1)

            # V2 departure: "attach" it to the same cycle-day as V1,
            # possibly rolling to the next day until it is not before arr1
            dep2 = abs_minutes(day, sch.dep_v2)
            while dep2 < arr1:
                dep2 += 1440  # move to next day

            # V2 arrival: compute from dep2 (not from day!)
            arr2 = dep2 - (dep2 % 1440) + minutes_from_midnight(sch.arr_v2)
            # if arrival clock is earlier than departure clock, it crosses midnight
            if (arr2 % 1440) < (dep2 % 1440):
                arr2 += 1440

            v1_dep_abs[(k, day)] = dep1
            v1_arr_abs[(k, day)] = arr1
            v2_dep_abs[(k, day)] = dep2
            v2_arr_abs[(k, day)] = arr2

            # first_dep_slot[(k, day)] = slot_index_from_abs_minute(dep1)
            first_dep_slot[(k, day)] = int(math.ceil(dep1 / DT_MIN))

    # ------------------------------------------------------------
    # Consumption profile cons_kmh[(k,m,h)]  [kWh per slot]
    # depends on battery config m through Etrip_sm[(k,leg,m)]
    # leg = 1 for V1, leg = 2 for V2
    # ------------------------------------------------------------
    Etrip_sm = Etrip_sm or {}

    cons_kmh = {(k, m, h): 0.0 for k in K for m in M for h in H}

    for k in K:
        sch = sched_by_k[k]
        for day in DAYS:
            if alpha[(k, day)] == 0:
                continue

            # V1 interval
            dep1 = v1_dep_abs[(k, day)]
            arr1 = v1_arr_abs[(k, day)]
            h_start_1 = slot_index_from_abs_minute(dep1)
            h_end_1 = slot_index_from_abs_minute(arr1 - 1)
            slots_v1 = list(range(h_start_1, min(h_end_1 + 1, SLOTS_PER_HORIZON)))
            n1 = len(slots_v1)

            # V2 interval
            dep2 = v2_dep_abs[(k, day)]
            arr2 = v2_arr_abs[(k, day)]
            h_start_2 = slot_index_from_abs_minute(dep2)
            h_end_2 = slot_index_from_abs_minute(arr2 - 1)
            slots_v2 = list(range(h_start_2, min(h_end_2 + 1, SLOTS_PER_HORIZON)))
            n2 = len(slots_v2)

            for m in M:
                # total trip energies for this battery config
                E1 = float(Etrip_sm.get((k, 1, m), 0.0))  # V1
                E2 = float(Etrip_sm.get((k, 2, m), 0.0))  # V2

                if n1 > 0 and E1 > 0:
                    per_slot_1 = E1 / n1
                    for h in slots_v1:
                        cons_kmh[(k, m, h)] += per_slot_1

                if n2 > 0 and E2 > 0:
                    per_slot_2 = E2 / n2
                    for h in slots_v2:
                        cons_kmh[(k, m, h)] += per_slot_2

    # Location availability L[k,t,h]
    # Rule:
    # - In route => L=0 for both terminals
    # - Otherwise => at A, except between V1 arrival and V2 departure (at B)
    L = {(k, t, h): 0 for k in K for t in T for h in H}

    # Precompute "in route" flags and "between V1arr and V2dep" flags by bus
    in_route = {(k, h): False for k in K for h in H}
    at_B_window = {(k, h): False for k in K for h in H}

    for k in K:
        # default: if not in route and not in B-window, at A.
        for day in DAYS:
            if alpha[(k, day)] == 0:
                continue

            dep1, arr1 = v1_dep_abs[(k, day)], v1_arr_abs[(k, day)]
            dep2, arr2 = v2_dep_abs[(k, day)], v2_arr_abs[(k, day)]

            for h in H:
                a0, a1 = slot_interval_abs_minutes(h)

                # route windows
                if overlaps(a0, a1, dep1, arr1) or overlaps(a0, a1, dep2, arr2):
                    in_route[(k, h)] = True

                # between V1 arrival and V2 departure -> at B
                if overlaps(a0, a1, arr1, dep2):
                    at_B_window[(k, h)] = True

        for h in H:
            if in_route[(k, h)]:
                # cannot charge
                L[(k, "A", h)] = 0
                L[(k, "B", h)] = 0
            else:
                if at_B_window[(k, h)]:
                    L[(k, "A", h)] = 0
                    L[(k, "B", h)] = 1
                else:
                    # when bus not operating (or after finishing) it is at A by your rule
                    L[(k, "A", h)] = 1
                    L[(k, "B", h)] = 0

    anchor_slot = {}
    for k in K:
        found = None
        for h in range(WEEK_SLOTS):  # solo semana 1
            if L[(k, "A", h)] == 1:
                found = h
                break
        anchor_slot[k] = 0 if found is None else int(found)


    return {
        "K": K, "T": T, "C": C, "H": H, "DAYS": DAYS, "M": M,
        "alpha": alpha,
        "L": L,
        "cons_kmh": cons_kmh,
        "first_dep_slot": first_dep_slot,
        "P_kw": P_kw,
        "eta": eta,
        "SoC_min_m": SoC_min_m,
        "B": battery_configs_kwh,
        "dt_hours": DT_MIN / 60.0,
        "anchor_slot": anchor_slot,
    }

def add_switching_terms(m, lambda_sw=1.0):
    """
    Adds ON/OFF switching penalty terms:
      u[k,h] = 1 if bus charges in slot h
      sw[k,h] >= |u[k,h] - u[k,h-1]|
      SW = sum sw
    """

    # If already added, do nothing
    if hasattr(m, "SW"):
        return

    # Hsw = 1..H_last (needs RangeSet)
    H_last = int(m.H.last())
    m.Hsw = pyo.RangeSet(1, H_last)

    # u and sw binaries
    m.u = pyo.Var(m.K, m.H, domain=pyo.Binary)
    m.sw = pyo.Var(m.K, m.Hsw, domain=pyo.Binary)

    # u == sum z (since OneChargerPerBus enforces <=1, this becomes 0/1)
    def u_link_rule(_m, k, h):
        return _m.u[k, h] == sum(_m.z[k, t, c, h] for t in _m.T for c in _m.C)
    m.Ulink = pyo.Constraint(m.K, m.H, rule=u_link_rule)

    # sw >= |u_h - u_{h-1}|
    def sw_pos_rule(_m, k, h):
        return _m.sw[k, h] >= _m.u[k, h] - _m.u[k, h-1]
    def sw_neg_rule(_m, k, h):
        return _m.sw[k, h] >= _m.u[k, h-1] - _m.u[k, h]
    m.SWpos = pyo.Constraint(m.K, m.Hsw, rule=sw_pos_rule)
    m.SWneg = pyo.Constraint(m.K, m.Hsw, rule=sw_neg_rule)

    # Total switches
    m.SW = pyo.Expression(expr=sum(m.sw[k, h] for k in m.K for h in m.Hsw))

    # Penalty term (Expression)
    m.SwitchPenalty = pyo.Expression(expr=float(lambda_sw) * m.SW)



# -----------------------------
# Pyomo model builder
# -----------------------------
def build_pyomo_model(
    params: dict,
    # Cost inputs (data): you can pass placeholders now and update later
    charger_cost: dict,   # Cinv[c]  e.g. {"slow": 30000, "fast": 90000}
    battery_cost: dict,   # Cbat[m]  e.g. {"B200": 120000, "B250": 150000, "B300": 185000}
    minimize_only_chargers: bool = False,  # if True: ignore battery/energy costs
):
    """
    Builds a weekly EV bus charging MILP.

    Requires params produced by build_horizon_parameters with at least:
      - K, T, C, H, D (days), M
      - L[(k,t,h)] ∈ {0,1}
      - cons_kmh[(k,m,h)] ≥ 0  (kWh consumed per slot if battery config m is chosen)
      - first_dep_slot[(k,day)]  slot index of V1 departure for that bus/day (when alpha=1)
      - alpha[(k,day)] ∈ {0,1}
      - B[m] (battery capacity kWh)
      - P_kw[c], eta[c], dt_hours, SoC_min

    Decision variables:
      - SoC[k,h] continuous
      - z[k,t,c,h] binary (bus uses one charger of type c at terminal t in slot h)
      - N[t,c] integer number of chargers of type c installed at terminal t
      - w[k,m] binary battery config selection

    Objective:
      Min CAPEX(chargers) + CAPEX(batteries)
    """
    K = params["K"]
    T = params["T"]
    C = params["C"]
    H = params["H"]   # expects range(672) or list of ints
    D = params["DAYS"]
    M = params["M"]

    m = pyo.ConcreteModel("EV_bus_weekly")

    # -----------------------
    # Sets
    # -----------------------
    m.K = pyo.Set(initialize=K)
    m.T = pyo.Set(initialize=T)
    m.C = pyo.Set(initialize=C)
    m.H = pyo.RangeSet(0, len(H) - 1)
    m.D = pyo.Set(initialize=D)
    m.M = pyo.Set(initialize=M)

    # -----------------------
    # Scalars
    # -----------------------
    m.dt = pyo.Param(initialize=float(params["dt_hours"]))  # hours per slot (0.25 for 15min)

    # -----------------------
    # Parameters: tech
    # -----------------------
    def B_init(_m, mm):
        return float(params["B"][mm])
    m.B = pyo.Param(m.M, initialize=B_init)  # kWh

    def SoC_min_init(_m, mm):
        return float(params["SoC_min_m"][mm])
    m.SoC_min = pyo.Param(m.M, initialize=SoC_min_init)  # kWh

    def P_init(_m, cc):
        return float(params["P_kw"][cc])
    m.P = pyo.Param(m.C, initialize=P_init)  # kW

    def eta_init(_m, cc):
        return float(params["eta"][cc])
    m.eta = pyo.Param(m.C, initialize=eta_init)

    # Location feasibility L[k,t,h]
    def L_init(_m, k, t, h):
        return int(params["L"][(k, t, int(h))])
    m.L = pyo.Param(m.K, m.T, m.H, initialize=L_init, within=pyo.Binary)

    # Consumption cons_kmh[k,m,h]
    # kWh consumed by bus k in slot h IF it uses battery config m
    def cons_kmh_init(_m, k, mm, h):
        return float(params["cons_kmh"][(k, mm, int(h))])
    m.cons_kmh = pyo.Param(m.K, m.M, m.H, initialize=cons_kmh_init, within=pyo.NonNegativeReals)

    # Alpha and first departure slot
    def alpha_init(_m, k, d):
        return int(params["alpha"][(k, d)])
    m.alpha = pyo.Param(m.K, m.D, initialize=alpha_init, within=pyo.Binary)

    first_dep_slot = {(k, d): params["first_dep_slot"].get((k, d), -1) for k in K for d in D}

    def first_dep_init(_m, k, d):
        return int(first_dep_slot[(k, d)])
    m.first_dep = pyo.Param(m.K, m.D, initialize=first_dep_init, within=pyo.Integers)

    # -----------------------
    # Parameters: costs (DATA, not variables)
    # -----------------------
    def Cinv_init(_m, cc):
        if cc not in charger_cost:
            raise KeyError(f"Missing charger_cost for charger type {cc!r}")
        return float(charger_cost[cc])
    m.Cinv = pyo.Param(m.C, initialize=Cinv_init)  # $ per charger

    def Cbat_init(_m, mm):
        if mm not in battery_cost:
            raise KeyError(f"Missing battery_cost for battery config {mm!r}")
        return float(battery_cost[mm])
    m.Cbat = pyo.Param(m.M, initialize=Cbat_init)  # $ per bus battery choice

    # -----------------------
    # Decision variables
    # -----------------------
    m.SoC = pyo.Var(m.K, m.H, domain=pyo.NonNegativeReals)

    # Charger usage (binary): one bus occupies one charger
    m.z = pyo.Var(m.K, m.T, m.C, m.H, domain=pyo.Binary)

    # Charging power (continuous)
    m.p = pyo.Var(m.K, m.T, m.C, m.H, domain=pyo.NonNegativeReals)  # kW

    # Charger counts (design)
    m.N = pyo.Var(m.T, m.C, domain=pyo.NonNegativeIntegers)

    # Battery selection
    m.w = pyo.Var(m.K, m.M, domain=pyo.Binary)

    # Additional helper variables
    m.u = pyo.Var(m.K, m.H, domain=pyo.Binary)  # charging on/off
    m.sw = pyo.Var(m.K, pyo.RangeSet(1, len(H)-1), domain=pyo.Binary)  # switch indicator

    # Helper: capacity of bus k from chosen battery
    def CapExpr(_m, k):
        return sum(_m.B[mm] * _m.w[k, mm] for mm in _m.M)
    
    def CapMinExpr(_m, k):
        return sum(_m.SoC_min[mm] * _m.w[k, mm] for mm in _m.M)

    # -----------------------
    # Constraints
    # -----------------------

    # (1) Choose exactly one battery config per bus
    def battery_choice_rule(_m, k):
        return sum(_m.w[k, mm] for mm in _m.M) == 1
    m.BatteryChoice = pyo.Constraint(m.K, rule=battery_choice_rule)

    # ✅ (1.5) Initial condition: at start of week (h=0) all buses fully charged
    # def soc_initial_rule(_m, k):
    #     return _m.SoC[k, 0] == CapExpr(_m, k) * 0.80
    # m.SoCInitial = pyo.Constraint(m.K, rule=soc_initial_rule)

    # (2) SoC bounds
    # --- Lower bound: SoC >= SoCmin (depends on battery choice)
    def soc_lb_rule(_m, k, h):
        return _m.SoC[k, h] >= CapMinExpr(_m, k)
    m.SoC_LB = pyo.Constraint(m.K, m.H, rule=soc_lb_rule)

    # --- Upper bound: SoC <= Capacity (depends on battery choice)
    def soc_ub_rule(_m, k, h):
        return _m.SoC[k, h] <= CapExpr(_m, k)
    m.SoC_UB = pyo.Constraint(m.K, m.H, rule=soc_ub_rule)

    # (3) Charging only where the bus is
    def location_rule(_m, k, t, c, h):
        return _m.z[k, t, c, h] <= _m.L[k, t, h]
    m.Location = pyo.Constraint(m.K, m.T, m.C, m.H, rule=location_rule)

    # (4) One charger at most per bus per time slot (no double-plugging)
    def one_charger_per_bus_rule(_m, k, h):
        return sum(_m.z[k, t, c, h] for t in _m.T for c in _m.C) <= 1
    m.OneChargerPerBus = pyo.Constraint(m.K, m.H, rule=one_charger_per_bus_rule)

    # (5) Charger capacity per terminal/type/time
    def charger_capacity_rule(_m, t, c, h):
        return sum(_m.z[k, t, c, h] for k in _m.K) <= _m.N[t, c]
    m.ChargerCapacity = pyo.Constraint(m.T, m.C, m.H, rule=charger_capacity_rule)

    def p_limit_rule(_m, k, t, c, h):
        return _m.p[k, t, c, h] <= _m.P[c] * _m.z[k, t, c, h]
    m.PLimit = pyo.Constraint(m.K, m.T, m.C, m.H, rule=p_limit_rule)

    # (6) SoC balance (weekly)
    # SoC[k,h+1] = SoC[k,h] + sum_{t,c} eta[c]*P[c]*dt*z - sum_m cons_kmh[k,m,h]*w[k,m]
    def soc_balance_rule(_m, k, h):
        if int(h) == len(H) - 1:
            return pyo.Constraint.Skip

        charge_gain = sum(
            _m.eta[c] * _m.p[k, t, c, h] * _m.dt
            for t in _m.T for c in _m.C
        )

        consumption = sum(
            _m.cons_kmh[k, mm, h] * _m.w[k, mm]
            for mm in _m.M
        )

        return _m.SoC[k, h + 1] == _m.SoC[k, h] + charge_gain - consumption

    m.SoCBalance = pyo.Constraint(m.K, m.H, rule=soc_balance_rule)

    # (7) Full battery at first departure of each operating day
    def full_before_first_service_rule(_m, k, d):
        if int(_m.alpha[k, d]) == 0:
            return pyo.Constraint.Skip
        h0 = int(_m.first_dep[k, d])
        if h0 <= 0:
            return pyo.Constraint.Skip
        return _m.SoC[k, h0 - 1] >= CapExpr(_m, k) * 0.80

    m.FullBeforeFirst = pyo.Constraint(m.K, m.D, rule=full_before_first_service_rule)

    # (8) Switching indicator
    def u_link_rule(_m, k, h):
        return _m.u[k,h] == sum(_m.z[k,t,c,h] for t in _m.T for c in _m.C)
    m.Ulink = pyo.Constraint(m.K, m.H, rule=u_link_rule)

    def sw_pos_rule(_m, k, h):
        return _m.sw[k,h] >= _m.u[k,h] - _m.u[k,h-1]
    def sw_neg_rule(_m, k, h):
        return _m.sw[k,h] >= _m.u[k,h-1] - _m.u[k,h]
    m.SWpos = pyo.Constraint(m.K, pyo.RangeSet(1, len(H)-1), rule=sw_pos_rule)
    m.SWneg = pyo.Constraint(m.K, pyo.RangeSet(1, len(H)-1), rule=sw_neg_rule)

    # (9) Wrap-around constraint: SoC after one week >= SoC at anchor slot
    W = int(WEEK_SLOTS)   # 672
    H_last = int(m.H.last())

    def anchor_init(_m, k):
        return int(params["anchor_slot"][k])
    m.anchor = pyo.Param(m.K, initialize=anchor_init, within=pyo.Integers)

    def wrap_anchor_rule(_m, k):
        a = int(_m.anchor[k])
        # necesitamos que exista a+W dentro del horizonte de 2 semanas
        if a < 0 or a + W > H_last:
            return pyo.Constraint.Skip
        return _m.SoC[k, a + W] >= _m.SoC[k, a]  # o igualdad si quieres ciclo estricto

    m.WrapAnchor = pyo.Constraint(m.K, rule=wrap_anchor_rule)
    

    # -----------------------
    # Objective
    # -----------------------
    # CAPEX chargers: sum_{t,c} Cinv[c] * N[t,c]
    m.capex_chargers = pyo.Expression(expr=sum(m.Cinv[c] * m.N[t, c] for t in m.T for c in m.C))

    if minimize_only_chargers:
        # Useful when you want "minimum chargers needed", ignoring battery costs.
        m.OBJ = pyo.Objective(expr=m.capex_chargers, sense=pyo.minimize)
        return m

    # CAPEX batteries: sum_{k,m} Cbat[m] * w[k,m]
    m.capex_batteries = pyo.Expression(expr=sum(m.Cbat[mm] * m.w[k, mm] for k in m.K for mm in m.M))
    
    m.CAPEX = pyo.Expression(expr=m.capex_chargers + m.capex_batteries)

    m.OBJ = pyo.Objective(
        expr=m.CAPEX,
        sense=pyo.minimize
    )

    return m

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    csv_path = "src/data/schedule.csv"  # put your path here

    schedules = load_schedules_from_csv(csv_path)

    # Define charger types (placeholders)
    charger_types = ["slow", "fast"]
    P_kw = {"slow": 42.0, "fast": 300.0}     # you can change later
    eta = {"slow": 1, "fast": 1}       # you can change later

    # Battery configurations (placeholders)
    battery_configs_kwh = {
        "BYD": 560.0,
        "IVECO-BUS": 415.0,
        "SOLARIS-URBINO": 400.0,
    }

    SoC_min_m = {n: 0.2 * c for n, c in battery_configs_kwh.items()}  # 20% min SoC

    # Energy per trip (kWh) using battery config m placeholders:
    # Fill these once you estimate energy use per leg per bus
    # Example: Etrip_sm[(bus_id, leg, battery_config)] = energy_kwh
    Etrip_sm = load_consumption_from_csv("src/data/trip_consumption.csv")

    params = build_horizon_parameters(
        schedules=schedules,
        charger_types=charger_types,
        P_kw=P_kw,
        eta=eta,
        SoC_min_m=SoC_min_m,
        battery_configs_kwh=battery_configs_kwh,
        Etrip_sm=Etrip_sm,
    )

    charger_cost = {"slow": 8000, "fast": 100000}
    battery_cost = {"BYD": 1000000, "IVECO-BUS": 900000, "SOLARIS-URBINO": 700000}

    # Build model
    model = build_pyomo_model(
        params,
        charger_cost=charger_cost,
        battery_cost=battery_cost,
    )

    model.write("debug.lp", io_options={"symbolic_solver_labels": True})

    # -------------------
    # FASE 1: minimizar CAPEX
    # -------------------

    # Solve
    solver_manager = SolverManager()
    solver = solver_manager.get_solver("cbc")
    res = solver.solve(model, tee=True, options={
        "findiIS": "on",
        # "seconds": 300,   # 5 minutos
        # "ratio": 0.02     # 2% gap
    })

    save_solution_json(model)
    
    print("\n=== Solver report ===")
    print("status:", res.solver.status)
    print("termination:", res.solver.termination_condition)

    # Si NO hay solución factible/óptima, no intentes leer variables
    term = str(res.solver.termination_condition).lower()
    if term not in ("optimal", "feasible"):
        print("No hay solución cargada (o el problema no terminó en feasible/optimal).")
        # guarda log para inspección
        # res = solver.solve(model, tee=True, logfile="highs_run.log", load_solutions=True)
        raise SystemExit(0)

    print("\n=== Charger counts ===")
    for t in model.T:
        for c in model.C:
            print(t, c, pyo.value(model.N[t, c]))

    print("\n=== Battery selected by bus ===")
    for k in model.K:
        for m in model.M:
            if pyo.value(model.w[k, m]) >= 0.9:
                print(f"Bus {k}: Battery {m}")

    # capex_star = pyo.value(model.CAPEX)
    # print("CAPEX* =", capex_star)

    # # -------------------
    # # FASE 2: minimizar switches con CAPEX congelado
    # # -------------------
    # add_switching_terms(model, lambda_sw=1.0)

    # # congelar CAPEX con un margen pequeño
    # eps = 0.001  # 0.1%
    # model.CAPEX_Freeze = pyo.Constraint(expr=model.CAPEX <= capex_star * (1 + eps))

    # # cambiar objetivo
    # model.OBJ.deactivate()
    # model.OBJ2 = pyo.Objective(expr=model.SW, sense=pyo.minimize)

    # res2 = solver.solve(model, tee=True, options={
    #     # "seconds": 300, "ratio": 0.01
    # }, load_solutions=True)
    # print("termination2:", res2.solver.termination_condition)

    # print("SW =", pyo.value(model.SW))
    # print("CAPEX final =", pyo.value(model.CAPEX))
