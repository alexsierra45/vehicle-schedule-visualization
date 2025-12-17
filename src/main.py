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


# -----------------------------
# Helpers: time parsing & mapping
# -----------------------------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_INDEX = {d: i for i, d in enumerate(DAYS)}

FREQ_MAP = {
    "D": set(DAYS),          # daily
    "MV": {"Tue", "Fri"},    # martes y viernes
    "JD": {"Thu", "Sun"},    # jueves y domingo
}

DT_MIN = 15
SLOTS_PER_DAY = 24 * 60 // DT_MIN  # 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY  # 672


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


def abs_minutes(day: str, hhmm_cell) -> int:
    """Absolute minutes from week start (Mon 00:00)."""
    return 1440 * DAY_INDEX[day] + minutes_from_midnight(hhmm_cell)


def abs_arrival_minutes(day: str, dep_cell, arr_cell) -> int:
    """Arrival absolute minutes; handles crossing midnight."""
    dep = abs_minutes(day, dep_cell)
    arr_raw = abs_minutes(day, arr_cell)
    if arr_raw < dep:
        arr_raw += 1440
    return arr_raw


def slot_index_from_abs_minute(abs_minute: int) -> int:
    """Map absolute minute to slot index h (0..671)."""
    return int(abs_minute // DT_MIN)


def slot_interval_abs_minutes(h: int) -> Tuple[int, int]:
    """Return (start_min, end_min) absolute minutes for slot h."""
    start = h * DT_MIN
    end = (h + 1) * DT_MIN
    return start, end


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
        if freq not in FREQ_MAP:
            raise ValueError(f"Unknown Frecuencia {freq!r} for bus {bus_id}. Expected one of {list(FREQ_MAP.keys())}")
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


# -----------------------------
# Build weekly parameters: alpha, L, consumption profile
# -----------------------------
def build_week_parameters(
    schedules: List[BusSchedule],
    charger_types: List[str],
    P_kw: Dict[str, float],
    eta: Dict[str, float],
    SoC_min_kwh: float,
    battery_configs_kwh: Dict[str, float],
    # Energy per trip (kWh) for each bus leg; fill these when you have them
    Etrip_v1: Optional[Dict[int, float]] = None,
    Etrip_v2: Optional[Dict[int, float]] = None,
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
    H = list(range(SLOTS_PER_WEEK))
    M = list(battery_configs_kwh.keys())

    # alpha: bus operates on day?
    alpha = {(k, day): 0 for k in K for day in DAYS}
    sched_by_k = {s.bus_id: s for s in schedules}
    for k in K:
        freq = sched_by_k[k].freq
        for day in FREQ_MAP[freq]:
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
            dep2 = abs_minutes(day, sch.dep_v2)
            arr2 = abs_arrival_minutes(day, sch.dep_v2, sch.arr_v2)

            v1_dep_abs[(k, day)] = dep1
            v1_arr_abs[(k, day)] = arr1
            v2_dep_abs[(k, day)] = dep2
            v2_arr_abs[(k, day)] = arr2

            first_dep_slot[(k, day)] = slot_index_from_abs_minute(dep1)

    # Consumption profile cons[k,h]
    cons = {(k, h): 0.0 for k in K for h in H}
    Etrip_v1 = Etrip_v1 or {}
    Etrip_v2 = Etrip_v2 or {}

    for k in K:
        for day in DAYS:
            if alpha[(k, day)] == 0:
                continue

            # V1 slots
            dep1 = v1_dep_abs[(k, day)]
            arr1 = v1_arr_abs[(k, day)]
            h_start_1 = slot_index_from_abs_minute(dep1)
            h_end_1 = slot_index_from_abs_minute(arr1 - 1)  # last slot touching [dep,arr)
            slots_v1 = [h for h in range(h_start_1, min(h_end_1 + 1, SLOTS_PER_WEEK))]

            # V2 slots
            dep2 = v2_dep_abs[(k, day)]
            arr2 = v2_arr_abs[(k, day)]
            h_start_2 = slot_index_from_abs_minute(dep2)
            h_end_2 = slot_index_from_abs_minute(arr2 - 1)
            slots_v2 = [h for h in range(h_start_2, min(h_end_2 + 1, SLOTS_PER_WEEK))]

            # Distribute energy uniformly across slots (if provided; else 0)
            if k in Etrip_v1 and len(slots_v1) > 0:
                per_slot = float(Etrip_v1[k]) / len(slots_v1)
                for h in slots_v1:
                    cons[(k, h)] += per_slot
            if k in Etrip_v2 and len(slots_v2) > 0:
                per_slot = float(Etrip_v2[k]) / len(slots_v2)
                for h in slots_v2:
                    cons[(k, h)] += per_slot

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

    return {
        "K": K, "T": T, "C": C, "H": H, "DAYS": DAYS, "M": M,
        "alpha": alpha,
        "L": L,
        "cons": cons,
        "first_dep_slot": first_dep_slot,
        "P_kw": P_kw,
        "eta": eta,
        "SoC_min": SoC_min_kwh,
        "B": battery_configs_kwh,
        "dt_hours": DT_MIN / 60.0,
    }


# -----------------------------
# Pyomo model builder
# -----------------------------
def build_pyomo_model(params: Dict, optimize_chargers: bool = True, choose_battery: bool = False):
    """
    If optimize_chargers=True: N[t,c] are variables and objective minimizes sum N.
    If optimize_chargers=False: N fixed via Param (set after building, or modify code).
    If choose_battery=True: w[k,m] binary decision. Else: fix a single battery config for all buses (set in params).
    """
    K = params["K"]; T = params["T"]; C = params["C"]; H = params["H"]; DAYS = params["DAYS"]; M = params["M"]
    dt = params["dt_hours"]

    m = pyo.ConcreteModel("EV_bus_weekly")

    # Sets
    m.K = pyo.Set(initialize=K)
    m.T = pyo.Set(initialize=T)
    m.C = pyo.Set(initialize=C)
    m.H = pyo.RangeSet(0, len(H) - 1)  # 0..671
    m.D = pyo.Set(initialize=DAYS)
    m.M = pyo.Set(initialize=M)

    # Params
    m.dt = pyo.Param(initialize=dt)
    m.SoC_min = pyo.Param(initialize=float(params["SoC_min"]))

    def B_init(_m, mm):
        return float(params["B"][mm])
    m.B = pyo.Param(m.M, initialize=B_init)

    def P_init(_m, cc):
        return float(params["P_kw"][cc])
    m.P = pyo.Param(m.C, initialize=P_init)

    def eta_init(_m, cc):
        return float(params["eta"][cc])
    m.eta = pyo.Param(m.C, initialize=eta_init)

    # Weekly location feasibility
    def L_init(_m, k, t, h):
        return int(params["L"][(k, t, int(h))])
    m.L = pyo.Param(m.K, m.T, m.H, initialize=L_init, within=pyo.Binary)

    # Weekly consumption
    def cons_init(_m, k, h):
        return float(params["cons"][(k, int(h))])
    m.cons = pyo.Param(m.K, m.H, initialize=cons_init, within=pyo.NonNegativeReals)

    # Alpha per day (bus operates on that day?) and first departure slot
    def alpha_init(_m, k, d):
        return int(params["alpha"][(k, d)])
    m.alpha = pyo.Param(m.K, m.D, initialize=alpha_init, within=pyo.Binary)

    # First departure slot for V1 on each active day; store as Param with default -1 if inactive
    first_dep_slot = {(k, d): params["first_dep_slot"].get((k, d), -1) for k in K for d in DAYS}

    def first_dep_init(_m, k, d):
        return int(first_dep_slot[(k, d)])
    m.first_dep = pyo.Param(m.K, m.D, initialize=first_dep_init, within=pyo.Integers)

    # Decision variables
    m.SoC = pyo.Var(m.K, m.H, domain=pyo.NonNegativeReals)

    # Charger usage: binary by default (you can relax to [0,1] by changing domain)
    m.z = pyo.Var(m.K, m.T, m.C, m.H, domain=pyo.Binary)

    # Number of chargers
    if optimize_chargers:
        m.N = pyo.Var(m.T, m.C, domain=pyo.NonNegativeIntegers)
    else:
        # If you want fixed N, set via params dict and make Param here
        raise NotImplementedError("Set optimize_chargers=True or modify to use fixed N as Param.")

    # Battery choice
    if choose_battery:
        m.w = pyo.Var(m.K, m.M, domain=pyo.Binary)

        def battery_choice_rule(_m, k):
            return sum(_m.w[k, mm] for mm in _m.M) == 1
        m.BatteryChoice = pyo.Constraint(m.K, rule=battery_choice_rule)

        def cap_k(_m, k):
            return sum(_m.B[mm] * _m.w[k, mm] for mm in _m.M)
    else:
        # Fix a single battery config for all buses: pick first M or specify in code
        default_m = list(M)[0]
        m.default_m = pyo.Param(initialize=default_m, mutable=False)
        # capacity is constant
        def cap_k(_m, k):
            return _m.B[_m.default_m]

    # Constraint: SoC bounds
    def soc_bounds_rule(_m, k, h):
        return pyo.inequality(_m.SoC_min, _m.SoC[k, h], cap_k(_m, k))
    m.SoCBounds = pyo.Constraint(m.K, m.H, rule=soc_bounds_rule)

    # Constraint: charging only where the bus is (location feasibility)
    def location_rule(_m, k, t, c, h):
        return _m.z[k, t, c, h] <= _m.L[k, t, h]
    m.Location = pyo.Constraint(m.K, m.T, m.C, m.H, rule=location_rule)

    # Constraint: at most one charger (one terminal/type) used per bus per time slot
    def one_charger_per_bus_rule(_m, k, h):
        return sum(_m.z[k, t, c, h] for t in _m.T for c in _m.C) <= 1
    m.OneChargerPerBus = pyo.Constraint(m.K, m.H, rule=one_charger_per_bus_rule)

    # Constraint: charger capacity at each terminal/type/time
    def charger_capacity_rule(_m, t, c, h):
        return sum(_m.z[k, t, c, h] for k in _m.K) <= _m.N[t, c]
    m.ChargerCapacity = pyo.Constraint(m.T, m.C, m.H, rule=charger_capacity_rule)

    # SoC balance across the week
    # SoC[k,h+1] = SoC[k,h] + sum_{t,c} eta[c]*P[c]*dt*z - cons[k,h]
    def soc_balance_rule(_m, k, h):
        if int(h) == len(H) - 1:
            return pyo.Constraint.Skip
        charge_gain = sum(_m.eta[c] * _m.P[c] * _m.dt * _m.z[k, t, c, h] for t in _m.T for c in _m.C)
        return _m.SoC[k, h + 1] == _m.SoC[k, h] + charge_gain - _m.cons[k, h]
    m.SoCBalance = pyo.Constraint(m.K, m.H, rule=soc_balance_rule)

    # Full battery at first service departure each operating day
    # SoC[k, first_dep(k,d)] == capacity(k) for days where alpha=1
    def full_before_first_service_rule(_m, k, d):
        if _m.alpha[k, d] == 0:
            return pyo.Constraint.Skip
        h0 = int(_m.first_dep[k, d])
        if h0 < 0:
            return pyo.Constraint.Skip
        return _m.SoC[k, h0] == cap_k(_m, k)
    m.FullBeforeFirst = pyo.Constraint(m.K, m.D, rule=full_before_first_service_rule)

    # Objective: minimize total number of chargers (dimensioning)
    if optimize_chargers:
        m.OBJ = pyo.Objective(expr=sum(m.N[t, c] for t in m.T for c in m.C), sense=pyo.minimize)
    else:
        m.OBJ = pyo.Objective(expr=0.0, sense=pyo.minimize)

    return m


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    csv_path = "src/data/trips.csv"  # put your path here

    schedules = load_schedules_from_csv(csv_path)

    # Define charger types (placeholders)
    charger_types = ["slow", "fast"]
    P_kw = {"slow": 50.0, "fast": 150.0}     # you can change later
    eta = {"slow": 0.92, "fast": 0.90}       # you can change later

    # Battery configurations (placeholders)
    battery_configs_kwh = {
        "B200": 200.0,
        "B250": 250.0,
        "B300": 300.0,
    }

    SoC_min_kwh = 20.0  # placeholder safety threshold

    # Energy per trip (kWh) placeholders:
    # Fill these once you estimate energy use per leg per bus
    # Example: Etrip_v1 = {1: 60.0, 2: 58.0, ...}, Etrip_v2 = {1: 62.0, ...}
    Etrip_v1 = {}  # empty -> consumption will be 0, so model will push chargers to 0
    Etrip_v2 = {}

    params = build_week_parameters(
        schedules=schedules,
        charger_types=charger_types,
        P_kw=P_kw,
        eta=eta,
        SoC_min_kwh=SoC_min_kwh,
        battery_configs_kwh=battery_configs_kwh,
        Etrip_v1=Etrip_v1,
        Etrip_v2=Etrip_v2,
    )

    # Build model
    model = build_pyomo_model(params, optimize_chargers=True, choose_battery=False)
    # choose_battery=True will allow the model to pick among B200/B250/B300 for each bus,
    # but you should then also add a battery cost in objective if you want meaningful results.

    # Solve (choose one solver you have installed)
    solver = pyo.SolverFactory("highs")  # or "cbc", "glpk", "gurobi"
    res = solver.solve(model, tee=True)

    # Print results: total chargers per terminal/type
    print("\n=== Charger counts ===")
    for t in model.T:
        for c in model.C:
            print(t, c, pyo.value(model.N[t, c]))

    # Optional: check SoC minimum by bus
    print("\n=== Min SoC by bus ===")
    for k in model.K:
        soc_min = min(pyo.value(model.SoC[k, h]) for h in model.H)
        print(k, soc_min)
