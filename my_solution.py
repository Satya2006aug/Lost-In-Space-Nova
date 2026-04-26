"""
plan_imaging_v5_3.py -- Basilisk-oriented case-adaptive mosaic planner

Fixes over v5.2:
  * Uses one inertial frame consistently.  SGP4 TEME spacecraft states and
    ground target vectors are both converted to the same J2000-like frame before
    any pointing quaternion is computed.
  * Builds q_BN as body-to-J2000 with scalar-last [qx, qy, qz, qw], matching the
    original harness contract used by the supplied planner.
  * Keeps the planner frame-maxing at 49/49.  Wheel checks only reject hard peak
    saturation cases; they do not cap total momentum throughput.
"""

from typing import Any, Dict, List, Tuple


def plan_imaging(
    tle_line1: str,
    tle_line2: str,
    aoi_polygon_llh: List[Tuple[float, float]],
    pass_start_utc: str,
    pass_end_utc: str,
    sc_params: Dict[str, Any],
) -> Dict[str, Any]:

    import math
    from datetime import datetime, timezone

    try:
        from sgp4.api import Satrec
    except Exception:
        Satrec = None

    MU = 398600.4418
    WGS84_A = 6378.137
    WGS84_F = 1.0 / 298.257223563

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def mul(s, a):
        return [s * a[0], s * a[1], s * a[2]]

    def cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def norm(a):
        return math.sqrt(max(0.0, dot(a, a)))

    def unit(a):
        n = norm(a)
        return [0.0, 0.0, 0.0] if n < 1e-12 else [a[0] / n, a[1] / n, a[2] / n]

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def look_angle_rad(u, v):
        return math.acos(clamp(dot(u, v), -1.0, 1.0))

    def look_angle_deg(u, v):
        return math.degrees(look_angle_rad(u, v))

    def parse_utc(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

    def julian_date(dt: datetime) -> float:
        y, m = dt.year, dt.month
        d = dt.day
        hr = dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond * 1e-6) / 3600.0
        if m <= 2:
            y -= 1
            m += 12
        a = y // 100
        b = 2 - a + a // 4
        return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + hr / 24.0

    def gmst_rad(dt: datetime) -> float:
        jd = julian_date(dt)
        t = (jd - 2451545.0) / 36525.0
        deg = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t
            - t * t * t / 38710000.0
        )
        return math.radians(deg % 360.0)

    def rz(theta, v):
        c, s = math.cos(theta), math.sin(theta)
        return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]

    def rx(theta, v):
        c, s = math.cos(theta), math.sin(theta)
        return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]

    def ry(theta, v):
        c, s = math.cos(theta), math.sin(theta)
        return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]

    def llh_to_ecef(lat_deg, lon_deg, h_km=0.0):
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        e2 = WGS84_F * (2.0 - WGS84_F)
        sl = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - e2 * sl * sl)
        return [
            (n + h_km) * math.cos(lat) * math.cos(lon),
            (n + h_km) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - e2) + h_km) * sl,
        ]

    def teme_to_j2000(r_teme, v_teme, jd):
        """Approximate TEME/date vector to J2000 using eqeq + IAU 1976 precession."""
        T = (jd - 2451545.0) / 36525.0

        omega_rad = math.radians(125.04455 - 1934.1362608 * T)
        l_sun_rad = math.radians(280.4665 + 36000.7698 * T)
        dpsi_arcsec = -17.1996 * math.sin(omega_rad) - 1.3187 * math.sin(2.0 * l_sun_rad)
        eps_deg = 23.4392911 - 0.0130042 * T
        eq_eq = math.radians(dpsi_arcsec / 3600.0) * math.cos(math.radians(eps_deg))

        r_tod = rz(-eq_eq, r_teme)
        v_tod = rz(-eq_eq, v_teme)

        zeta_A = math.radians((2306.2181 * T + 0.30188 * T * T + 0.017998 * T * T * T) / 3600.0)
        z_A = math.radians((2306.2181 * T + 1.09468 * T * T + 0.018203 * T * T * T) / 3600.0)
        theta_A = math.radians((2004.3109 * T - 0.42665 * T * T - 0.041775 * T * T * T) / 3600.0)

        r_j2k = rz(zeta_A, ry(-theta_A, rz(z_A, r_tod)))
        v_j2k = rz(zeta_A, ry(-theta_A, rz(z_A, v_tod)))
        return r_j2k, v_j2k

    def llh_to_j2000(lat_deg, lon_deg, dt):
        # GMST gives an ECEF -> date-equator inertial vector. Convert it through
        # the same date->J2000 path used for spacecraft states so look vectors do
        # not mix frames.
        jd = julian_date(dt)
        r_date = rz(gmst_rad(dt), llh_to_ecef(lat_deg, lon_deg))
        r_j2k, _ = teme_to_j2000(r_date, [0.0, 0.0, 0.0], jd)
        return r_j2k

    _S2 = math.sqrt(2.0) / 2.0
    _S4 = math.sqrt(2.0) / 4.0
    _APINV = [
        [_S2, 0.0, _S4],
        [0.0, _S2, _S4],
        [-_S2, 0.0, _S4],
        [0.0, -_S2, _S4],
    ]

    def inertia_diag():
        I = sc_params.get("inertia_kgm2")
        if isinstance(I, list) and len(I) >= 3:
            try:
                return [float(I[0][0]), float(I[1][1]), float(I[2][2])]
            except Exception:
                pass
        return [0.12, 0.12, 0.08]

    _I_BODY = inertia_diag()
    H_WHEEL_MAX = float(sc_params.get("wheel_Hmax_Nms", 0.030))
    H_DH_BUDGET = float(sc_params.get("wheel_dh_budget_Nms", 0.200))
    _SAT_MARGIN = 0.90

    def wheel_peak_deltas(slew_axis_unit, omega_peak_rad_s):
        dH_body = [_I_BODY[k] * omega_peak_rad_s * slew_axis_unit[k] for k in range(3)]
        return [-sum(_APINV[i][k] * dH_body[k] for k in range(3)) for i in range(4)]

    def body_dh_norm(slew_axis_unit, omega_peak_rad_s):
        return norm([_I_BODY[k] * omega_peak_rad_s * slew_axis_unit[k] for k in range(3)])

    class SimpleProp:
        def __init__(self, line1, line2):
            f = line2.split()
            self.inc = math.radians(float(f[2]))
            self.raan = math.radians(float(f[3]))
            self.e = float("0." + f[4].strip())
            self.argp = math.radians(float(f[5]))
            self.m0 = math.radians(float(f[6]))
            self.n = float(f[7]) * 2.0 * math.pi / 86400.0
            self.a = (MU / (self.n * self.n)) ** (1.0 / 3.0)
            epoch = line1.split()[3]
            yy = int(epoch[:2])
            year = 2000 + yy if yy < 57 else 1900 + yy
            doy = float(epoch[2:])
            self.epoch_jd = julian_date(datetime(year, 1, 1, tzinfo=timezone.utc)) + doy - 1.0

        def sgp4(self, jd_i, jd_f):
            dt = ((jd_i + jd_f) - self.epoch_jd) * 86400.0
            m = (self.m0 + self.n * dt) % (2.0 * math.pi)
            E = m
            for _ in range(8):
                E -= (E - self.e * math.sin(E) - m) / (1.0 - self.e * math.cos(E))
            ce, se = math.cos(E), math.sin(E)
            fac = math.sqrt(max(0.0, 1.0 - self.e * self.e))
            den = 1.0 - self.e * ce
            r = [self.a * (ce - self.e), self.a * fac * se, 0.0]
            v = [-self.a * self.n * se / den, self.a * self.n * fac * ce / den, 0.0]
            r = rz(self.raan, rx(self.inc, rz(self.argp, r)))
            v = rz(self.raan, rx(self.inc, rz(self.argp, v)))
            return 0, r, v

    def sat_state(t_sec):
        cur = datetime.fromtimestamp(start_dt.timestamp() + t_sec, tz=timezone.utc)
        jd = julian_date(cur)
        jd_i = math.floor(jd)
        err, r_teme, v_teme = sat.sgp4(jd_i, jd - jd_i)
        if err != 0:
            return None
        r_j2k, v_j2k = teme_to_j2000([float(x) for x in r_teme], [float(x) for x in v_teme], jd)
        return r_j2k, v_j2k, cur

    def off_nadir_deg(r_sat, r_tgt):
        look = unit(sub(r_tgt, r_sat))
        nadir = unit(mul(-1.0, r_sat))
        return math.degrees(math.acos(clamp(dot(look, nadir), -1.0, 1.0)))

    def quat_from_cols(x, y, z):
        m00, m01, m02 = x[0], y[0], z[0]
        m10, m11, m12 = x[1], y[1], z[1]
        m20, m21, m22 = x[2], y[2], z[2]
        tr = m00 + m11 + m22
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
        n = math.sqrt(sum(a * a for a in q))
        q = [a / n for a in q]
        return [-a for a in q] if q[3] < 0.0 else q

    def pointing_quat(r_sat, v_sat, r_tgt):
        z = unit(sub(r_tgt, r_sat))
        xh = sub(v_sat, mul(dot(v_sat, z), z))
        if norm(xh) < 1e-9:
            xh = cross([0.0, 0.0, 1.0], z)
        x = unit(xh)
        y = unit(cross(z, x))
        x = unit(cross(y, z))
        return quat_from_cols(x, y, z)

    start_dt = parse_utc(pass_start_utc)
    end_dt = parse_utc(pass_end_utc)
    pass_len = max(0.0, (end_dt - start_dt).total_seconds())
    sat = Satrec.twoline2rv(tle_line1, tle_line2) if Satrec is not None else SimpleProp(tle_line1, tle_line2)

    verts = aoi_polygon_llh[:]
    if len(verts) > 1 and tuple(verts[0]) == tuple(verts[-1]):
        verts = verts[:-1]
    lat_min = min(p[0] for p in verts)
    lat_max = max(p[0] for p in verts)
    lon_min = min(p[1] for p in verts)
    lon_max = max(p[1] for p in verts)
    center_lat = 0.5 * (lat_min + lat_max)
    center_lon = 0.5 * (lon_min + lon_max)

    targets = []
    lat_pad = 0.04 * (lat_max - lat_min)
    lon_pad = 0.04 * (lon_max - lon_min)
    for i in range(7):
        lat = lat_min + lat_pad + (lat_max - lat_min - 2.0 * lat_pad) * i / 6.0
        for j in range(7):
            lon = lon_min + lon_pad + (lon_max - lon_min - 2.0 * lon_pad) * j / 6.0
            targets.append((lat, lon))

    states = []
    t = 0.0
    while t <= pass_len + 1e-9:
        st = sat_state(t)
        if st:
            states.append((t, st[0], st[1], st[2]))
        t += 2.0

    center_best = 180.0
    for _, r, _, cur in states:
        center_best = min(center_best, off_nadir_deg(r, llh_to_j2000(center_lat, center_lon, cur)))

    official_limit = float(sc_params.get("off_nadir_max_deg", 60.0))

    if center_best < 20.0:
        case_label = "near_nadir_full_sweep"
        planning_limit = min(54.0, official_limit)
        max_frames = 49
        min_gap = 1.0
        sample_dt_fine = 2.0
    elif center_best < 45.0:
        case_label = "mid_off_nadir_full_sweep"
        planning_limit = min(55.0, official_limit)
        max_frames = 49
        min_gap = 1.0
        sample_dt_fine = 2.0
    else:
        case_label = "edge_visibility_greedy"
        planning_limit = min(59.98, official_limit - 0.02)
        max_frames = 49
        min_gap = 0.9
        sample_dt_fine = 1.0

    if sample_dt_fine < 2.0:
        states_fine = []
        t = 0.0
        while t <= pass_len + 1e-9:
            st = sat_state(t)
            if st:
                states_fine.append((t, st[0], st[1], st[2]))
            t += sample_dt_fine
    else:
        states_fine = states

    candidates = []
    for lat, lon in targets:
        best = None
        t_first = None
        t_last = None
        for t2, r, v, cur in states_fine:
            r_tgt = llh_to_j2000(lat, lon, cur)
            off = off_nadir_deg(r, r_tgt)
            if off <= planning_limit:
                if t_first is None:
                    t_first = t2
                t_last = t2
                if best is None or off < best["off"]:
                    best = {
                        "lat": lat,
                        "lon": lon,
                        "t": t2,
                        "off": off,
                        "r": r,
                        "v": v,
                        "r_tgt": r_tgt,
                        "look": unit(sub(r_tgt, r)),
                    }
        if best:
            best["t_first"] = t_first
            best["t_last"] = t_last
            candidates.append(best)

    def strip_snake_sort(cands, lat_mn, lat_mx, n_strips):
        if not cands:
            return cands
        lat_span = max(lat_mx - lat_mn, 1e-9)

        def strip_idx(c):
            return min(n_strips - 1, int(n_strips * (c["lat"] - lat_mn) / lat_span))

        buckets = [[] for _ in range(n_strips)]
        for c in cands:
            buckets[strip_idx(c)].append(c)
        result = []
        for s_idx, bucket in enumerate(buckets):
            bucket.sort(key=lambda c: c["lon"], reverse=(s_idx % 2 == 1))
            result.extend(bucket)
        return result

    if case_label in ("near_nadir_full_sweep", "mid_off_nadir_full_sweep"):
        candidates = strip_snake_sort(candidates, lat_min, lat_max, 7)
    else:
        candidates.sort(key=lambda c: (c["t"], c["off"]))

    time_slack = {
        "near_nadir_full_sweep": 10.0,
        "mid_off_nadir_full_sweep": 8.0,
        "edge_visibility_greedy": 2.0,
    }[case_label]

    def order_slew_score(order):
        total = 0.0
        for i in range(1, len(order)):
            total += look_angle_deg(order[i - 1]["look"], order[i]["look"])
        return total

    def two_opt_reorder(cands, max_passes=6):
        """Local route cleanup that preserves all tiles and only accepts shorter look paths."""
        if len(cands) <= 4:
            return cands
        order = list(cands)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, len(order) - 2):
                for k in range(i + 1, len(order) - 1):
                    before = (
                        look_angle_deg(order[i - 1]["look"], order[i]["look"])
                        + look_angle_deg(order[k]["look"], order[k + 1]["look"])
                    )
                    after = (
                        look_angle_deg(order[i - 1]["look"], order[k]["look"])
                        + look_angle_deg(order[i]["look"], order[k + 1]["look"])
                    )
                    if after + 1e-9 < before:
                        order[i : k + 1] = reversed(order[i : k + 1])
                        improved = True
        return order

    def nn_reorder(cands, slack):
        if len(cands) <= 1:
            return cands
        remaining = list(cands)
        ordered = [remaining.pop(0)]
        while remaining:
            cur_look = ordered[-1]["look"]
            cur_t = ordered[-1]["t"]
            best_idx = None
            best_cost = 1e99
            for idx, c in enumerate(remaining):
                if c["t"] < cur_t - slack:
                    continue
                angle = look_angle_deg(cur_look, c["look"])
                time_penalty = 0.04 * max(0.0, c["t"] - cur_t)
                off_penalty = 0.02 * c["off"]
                cost = angle + time_penalty + off_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx is None:
                remaining.sort(key=lambda c: (c["t"], c["off"]))
                ordered.extend(remaining)
                break
            ordered.append(remaining.pop(best_idx))
        return ordered

    candidates = nn_reorder(candidates, time_slack)
    for rank, c in enumerate(candidates):
        c["_rank"] = rank

    shutter_duration = float(sc_params.get("integration_s", 0.120))
    hold_pad = 0.020
    selected = []
    last_t = -999.0
    last_look = None
    cumul_dH = 0.0
    max_peak_wheel = 0.0

    # Targeted soft efficiency tuning:
    #   * Only reject extreme zig-zags above 45 deg as a safety bound.
    #   * Otherwise, angular slew is a soft score penalty.  This preserves high
    #     coverage while naturally preferring smoother transitions.
    HARD_SLEW_LIMIT_DEG = 45.0
    ANGLE_SCORE_PENALTY = 0.25
    DH_SCORE_PENALTY = 1.25
    FRAME_SCORE_REWARD = 60.0
    BEAM_WIDTH = 7
    GREEDY_LOOKAHEAD = 10
    MIN_EFFICIENT_FRAMES = 30
    MOMENTUM_PROXY_BUDGET_DEG = 180.0
    MOMENTUM_PROXY_PER_DEG = 2.25

    # Case-specific coverage floors for balanced coverage/efficiency behavior.
    if case_label == "near_nadir_full_sweep":
        target_floor = 42
    elif case_label == "mid_off_nadir_full_sweep":
        target_floor = 38
    else:
        target_floor = 30

    def evaluate_candidate(c, state):
        """Run the existing timing/constraint/wheel checks for one candidate."""
        sel, prev_t, prev_look, prev_dh, prev_peak, prev_momentum_est = state
        t_img = max(c["t"], prev_t + min_gap)
        if t_img + shutter_duration + hold_pad > pass_len:
            return None

        st = sat_state(t_img)
        if not st:
            return None
        r, v, cur = st
        r_tgt = llh_to_j2000(c["lat"], c["lon"], cur)
        off = off_nadir_deg(r, r_tgt)

        if off > official_limit - 0.015:
            return None
        if off > planning_limit + 0.45:
            return None

        current_look = unit(sub(r_tgt, r))
        maneuver_dh = 0.0
        angle_deg = 0.0

        if prev_look is not None:
            angle_deg = look_angle_deg(prev_look, current_look)
            if angle_deg > HARD_SLEW_LIMIT_DEG:
                return None

            # Global momentum proxy budget.  Coverage is prioritized until the
            # accumulated slew-angle proxy approaches the 200 mNms eta_E budget;
            # 180 leaves margin for model mismatch in the Basilisk run.
            candidate_momentum_est = prev_momentum_est + angle_deg * MOMENTUM_PROXY_PER_DEG
            if candidate_momentum_est > MOMENTUM_PROXY_BUDGET_DEG:
                return None

            slew_cross = cross(prev_look, current_look)
            slew_cross_norm = norm(slew_cross)
            if slew_cross_norm > 1e-9:
                slew_axis = unit(slew_cross)
                slew_rad = look_angle_rad(prev_look, current_look)
                t_gap = max(0.020, t_img - prev_t - shutter_duration - 2.0 * hold_pad)
                omega_peak = 2.0 * slew_rad / t_gap

                dh_w = wheel_peak_deltas(slew_axis, omega_peak)
                peak = max(abs(x) for x in dh_w)
                if peak > H_WHEEL_MAX * _SAT_MARGIN:
                    return None

                maneuver_dh = 2.0 * body_dh_norm(slew_axis, omega_peak)
                prev_peak = max(prev_peak, peak)
        else:
            candidate_momentum_est = prev_momentum_est

        c2 = dict(c)
        c2["t_img"] = t_img
        c2["r"] = r
        c2["v"] = v
        c2["r_tgt"] = r_tgt
        c2["off"] = off
        c2["q"] = pointing_quat(r, v, r_tgt)

        new_state = (
            sel + [c2],
            t_img,
            current_look,
            prev_dh + maneuver_dh,
            prev_peak,
            candidate_momentum_est,
        )
        return new_state, angle_deg, off, maneuver_dh

    if case_label in ("mid_off_nadir_full_sweep", "edge_visibility_greedy"):
        # Soft optimization for Case 2 and Case 3.  Coverage is still rewarded
        # strongly, while larger angular slews are discouraged instead of being
        # rejected unless they exceed the 45 deg safety bound.
        case_frame_reward = FRAME_SCORE_REWARD
        beams = [(([], -999.0, None, 0.0, 0.0, 0.0), 0.0)]
        for c in candidates:
            next_beams = list(beams)
            for state, score in beams:
                if len(state[0]) >= max_frames:
                    next_beams.append((state, score))
                    continue
                result = evaluate_candidate(c, state)
                if result is None:
                    continue
                new_state, angle_deg, off, maneuver_dh = result
                new_score = score
                new_score += case_frame_reward
                new_score -= angle_deg * ANGLE_SCORE_PENALTY
                new_score -= maneuver_dh * 1000.0 * DH_SCORE_PENALTY
                new_score -= off * 0.1
                next_beams.append((new_state, new_score))
            next_beams.sort(
                key=lambda item: (min(len(item[0][0]), target_floor), item[1]),
                reverse=True,
            )
            beams = next_beams[:BEAM_WIDTH]

        viable_beams = [item for item in beams if len(item[0][0]) >= target_floor]
        best_pool = viable_beams if viable_beams else beams
        best_state, _ = max(best_pool, key=lambda item: item[1])
        selected, last_t, last_look, cumul_dH, max_peak_wheel, total_momentum_est = best_state
    else:
        # Case 1 stays greedy, but uses a small local lookahead so the soft
        # angular penalty can prefer smoother nearby choices without rewriting
        # candidate generation or timing logic.
        state = (selected, last_t, last_look, cumul_dH, max_peak_wheel, 0.0)
        remaining = list(candidates)
        while remaining and len(state[0]) < max_frames:
            window = remaining[:GREEDY_LOOKAHEAD]
            best = None
            for idx, c in enumerate(window):
                result = evaluate_candidate(c, state)
                if result is None:
                    continue
                candidate_state, angle_deg, off, maneuver_dh = result
                greedy_score = FRAME_SCORE_REWARD
                greedy_score -= angle_deg * ANGLE_SCORE_PENALTY
                greedy_score -= maneuver_dh * 1000.0 * DH_SCORE_PENALTY
                greedy_score -= off * 0.1
                greedy_score -= idx * 0.05
                if best is None or greedy_score > best[0]:
                    best = (greedy_score, idx, candidate_state)

            if best is None:
                remaining.pop(0)
                continue

            best_score, chosen_idx, candidate_state = best
            if len(state[0]) >= max(MIN_EFFICIENT_FRAMES, target_floor) and best_score < 0.0:
                remaining.pop(0)
                continue

            state = candidate_state
            remaining.pop(chosen_idx)
        selected, last_t, last_look, cumul_dH, max_peak_wheel, total_momentum_est = state

    if not selected:
        return {
            "objective": "custom:no_reachable_targets_v5_3",
            "attitude": [
                {"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]},
                {"t": round(pass_len, 3), "q_BN": [0.0, 0.0, 0.0, 1.0]},
            ],
            "shutter": [],
            "notes": "No reachable sweep targets found in consistent J2000 geometry.",
            "target_hints_llh": [],
        }

    events = [(0.0, selected[0]["q"])]
    for s in selected:
        t0 = max(0.0, s["t_img"] - hold_pad)
        t1 = s["t_img"]
        t2 = s["t_img"] + shutter_duration
        t3 = min(pass_len, t2 + hold_pad)
        q = s["q"]
        for tt in (t0, t1, t2, t3):
            events.append((float(tt), q))
    events.append((max(pass_len, selected[-1]["t_img"] + shutter_duration), selected[-1]["q"]))
    events.sort(key=lambda x: x[0])

    compact = []
    for t, q in events:
        if compact and abs(t - compact[-1][0]) < 1e-6:
            compact[-1] = (t, q)
        elif compact and t <= compact[-1][0]:
            compact.append((compact[-1][0] + 0.021, q))
        else:
            compact.append((t, q))

    total_slew = 0.0
    max_slew_rate_dps = 0.0
    min_shutter_gap = 1e99
    for i in range(1, len(selected)):
        la = unit(sub(selected[i - 1]["r_tgt"], selected[i - 1]["r"]))
        lb = unit(sub(selected[i]["r_tgt"], selected[i]["r"]))
        slew_deg = look_angle_deg(la, lb)
        total_slew += slew_deg
        dt_slew = max(
            1e-6,
            selected[i]["t_img"] - selected[i - 1]["t_img"] - shutter_duration - 2.0 * hold_pad,
        )
        min_shutter_gap = min(min_shutter_gap, selected[i]["t_img"] - selected[i - 1]["t_img"])
        max_slew_rate_dps = max(max_slew_rate_dps, 2.0 * slew_deg / dt_slew)
    mean_slew = total_slew / max(1, len(selected) - 1)
    max_off_selected = max(s["off"] for s in selected)
    eta_E_est = max(0.0, min(1.0, 1.0 - cumul_dH / H_DH_BUDGET))

    attitude = [{"t": round(t, 3), "q_BN": [float(x) for x in q]} for t, q in compact]
    shutter = [
        {"t_start": round(s["t_img"], 3), "duration": round(shutter_duration, 3)}
        for s in selected
    ]
    hints = [
        {"lat_deg": round(s["lat"], 6), "lon_deg": round(s["lon"], 6)}
        for s in selected
    ]

    return {
        "objective": "custom:case_adaptive_greedy_v5_3_j2000_consistent",
        "attitude": attitude,
        "shutter": shutter,
        "notes": (
            f"{case_label}; center_best_off_nadir={center_best:.1f} deg; "
            f"{len(shutter)} frames from 49 grid tiles; "
            f"planning_limit={planning_limit:.2f} deg; min_gap={min_gap}s; "
            f"mean_inter_tile_slew={mean_slew:.2f} deg (total={total_slew:.1f} deg); "
            f"peak_slew_rate={max_slew_rate_dps:.2f} deg/s; "
            f"min_shutter_gap={min_shutter_gap:.2f}s; "
            f"max_selected_off_nadir={max_off_selected:.2f} deg; "
            f"cumul_dH={cumul_dH * 1000:.1f} mNms; "
            f"momentum_proxy={total_momentum_est:.1f}/180.0; "
            f"eta_E_est={eta_E_est:.3f}; "
            f"max_peak_wheel={max_peak_wheel * 1000:.1f} mNms; "
            f"frame_ref=J2000 consistent spacecraft+targets."
        ),
        "target_hints_llh": hints,
    }


if __name__ == "__main__":
    AOI = [
        (44.55, 9.37),
        (44.55, 10.63),
        (45.45, 10.63),
        (45.45, 9.37),
        (44.55, 9.37),
    ]
    SC = {
        "inertia_kgm2": [[0.12, 0, 0], [0, 0.12, 0], [0, 0, 0.08]],
        "wheel_layout": "pyramid_45deg",
        "wheel_Hmax_Nms": 0.030,
        "n_wheels": 4,
        "integration_s": 0.120,
        "fov_deg": [2.0, 2.0],
        "imager_boresight_B": [0.0, 0.0, 1.0],
        "smear_rate_limit_dps": 0.05,
        "off_nadir_max_deg": 60.0,
        "earth_model": "WGS84",
        "eci_frame": "J2000",
    }
    CASES = [
        (
            "Case 1 / Direct overpass",
            "1 99991U 26001A   26113.50000000  .00000000  00000-0  00000-0 0     7",
            "2 99991  97.4000 296.7000 0001000  90.0000 230.0000 15.21920000    08",
        ),
        (
            "Case 2 / 30 deg offset",
            "1 99992U 26001B   26113.50000000  .00000000  00000-0  00000-0 0     8",
            "2 99992  97.4000 292.9000 0001000  90.0000 230.0000 15.21920000    07",
        ),
        (
            "Case 3 / 60 deg offset",
            "1 99993U 26001C   26113.50000000  .00000000  00000-0  00000-0 0     9",
            "2 99993  97.4000 283.9000 0001000  90.0000 230.0000 15.21920000    08",
        ),
    ]
    print("plan_imaging v5.3 self-test")
    for name, tle1, tle2 in CASES:
        sched = plan_imaging(tle1, tle2, AOI, "2026-04-23T17:24:00Z", "2026-04-23T17:36:00Z", SC)
        print(f"\n{name}")
        print(f"  frames: {len(sched['shutter'])} / 49")
        print(f"  notes : {sched['notes']}")
        att = sched["attitude"]
        assert att[0]["t"] == 0.0
        assert att == sorted(att, key=lambda x: x["t"])
        for i in range(1, len(att)):
            dt = att[i]["t"] - att[i - 1]["t"]
            assert dt >= 0.019, f"Sub-20ms attitude gap at {i}: {dt}"
        sh = sched["shutter"]
        for i in range(1, len(sh)):
            assert sh[i]["t_start"] >= sh[i - 1]["t_start"] + sh[i - 1]["duration"]
        print("  checks: monotonic attitude, no shutter overlap")
