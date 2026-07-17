"""
model/fit_adjustment.py

Explicit net-rating adjustment for roster *composition* pathologies that
almost never appear in real NBA training rows (see MODELING_DECISIONS.md:
composition features are underpowered because historical fives are already
somewhat role-balanced).

Applied on top of Stage 1's value-driven prediction:

  adjusted_net = raw_net + fit_net_rating_adjustment(features)

Design targets (fan-toughness, five-man only):
  - Bird MVP + Kawhi + Aguirre iso + role Vince + scrub English  → ~54 wins
    (raw Stage 1 was ~64; too seduced by ΣBPM)
  - Peak piles where *everyone* is a plus player with real defense stay elite
  - One MVP + four scrubs gets a mild star-carry bump (raw was too harsh)

Taxes stack only when the roster looks pathological (3+ creators, no spacing
with creator pileup, offense without defense). If every player is clearly
positive and team defense is fine, mashup taxes are waived — five elite
two-ways with usage overlap still win a ton of games.
"""

from __future__ import annotations


# --- tax knobs (net-rating points; ~2.2 NR ≈ ~5 wins near .500) -------------
CREATOR_EXCESS_TAX = 1.55          # per creator beyond 2
USG_OVERFLOW_TAX_PER_POINT = 0.05  # only when creator_excess > 0; capped
USG_OVERFLOW_TAX_CAP = 20.0
SPACING_VOID_WITH_COLLISION_TAX = 0.90
THIN_DEF_THRESHOLD = 3.0           # sum_def_value below this + high off → tax
THIN_DEF_TAX_PER_POINT = 0.40
THIN_DEF_TAX_CAP = 1.50
ROLE_REDUNDANCY_WITH_COLLISION_TAX = 0.40

# Waive collision taxes when the five is uniformly strong + not defense-thin.
ELITE_FLOOR_MIN_OVERALL = 2.0
ELITE_FLOOR_MIN_DEF = 5.0

# --- star-carry knobs ------------------------------------------------------
STAR_MAX_THRESHOLD = 7.0           # normalized overall value
STAR_SUPPORT_WEAK = 5.0            # support_value below this → carry
STAR_CARRY_PER_SUPPORT_GAP = 0.55
STAR_CARRY_CAP = 5.5
SUPERSTAR_MAX_THRESHOLD = 7.5
SUPERSTAR_FLOOR_BOOST = 1.5        # extra when min is clearly negative


def fit_net_rating_adjustment(features: dict) -> float:
    """
    Return a net-rating delta (negative = tax, positive = star carry).
    Safe when features are missing (returns 0).
    """
    return _star_carry(features) - _fit_tax(features)


def _fit_tax(features: dict) -> float:
    creators = features.get("n_primary_creators") or 0
    creator_excess = float(features.get("creator_excess") or max(0, int(creators) - 2))

    collision = 0.0
    collision += CREATOR_EXCESS_TAX * creator_excess

    usg_overflow = features.get("usg_overflow")
    if creator_excess > 0 and usg_overflow is not None:
        collision += USG_OVERFLOW_TAX_PER_POINT * min(float(usg_overflow), USG_OVERFLOW_TAX_CAP)

    spacing_void = features.get("spacing_void")
    sum_off = features.get("sum_off_value")
    # Only punish no-spacing when the offense is crowded or already high-octane.
    # Pre-3PT-era complementary teams without "spacers" should not get nuked alone.
    if spacing_void == 1.0 and (creator_excess > 0 or (sum_off is not None and sum_off > 12.0)):
        collision += SPACING_VOID_WITH_COLLISION_TAX

    role_red = features.get("role_redundancy")
    if creator_excess > 0 and role_red is not None and role_red >= 2:
        collision += ROLE_REDUNDANCY_WITH_COLLISION_TAX

    sum_def = features.get("sum_def_value")
    thin_def = 0.0
    if sum_off is not None and sum_def is not None and sum_off > 10.0 and sum_def < THIN_DEF_THRESHOLD:
        thin_def = min(THIN_DEF_TAX_CAP, (THIN_DEF_THRESHOLD - float(sum_def)) * THIN_DEF_TAX_PER_POINT)

    # Five plus-players with real team defense: usage overlap is not a death sentence.
    min_v = features.get("min_overall_value")
    if (
        min_v is not None and min_v >= ELITE_FLOOR_MIN_OVERALL
        and sum_def is not None and sum_def >= ELITE_FLOOR_MIN_DEF
    ):
        collision = 0.0

    # True two-alpha piles (e.g. peak LeBron+Kobe, Curry+Giannis): don't
    # let creator-count taxes erase them. Keep thin-defense tax so iso-heavy
    # Bird+Kawhi+Aguirre boards still get punished (their max is ~7.8).
    max_v = features.get("max_overall_value")
    top2 = features.get("top2_overall_value")
    if (
        max_v is not None and max_v >= 8.5
        and top2 is not None and top2 >= 15.0
    ):
        collision = 0.0

    return collision + thin_def


def _star_carry(features: dict) -> float:
    max_v = features.get("max_overall_value")
    if max_v is None or max_v < STAR_MAX_THRESHOLD:
        return 0.0

    support = features.get("support_value")
    if support is None:
        sum_v = features.get("sum_overall_value")
        if sum_v is None:
            return 0.0
        support = float(sum_v) - float(max_v)

    carry = 0.0
    if support < STAR_SUPPORT_WEAK:
        carry += min(STAR_CARRY_CAP, (STAR_SUPPORT_WEAK - float(support)) * STAR_CARRY_PER_SUPPORT_GAP)
        # Extra bump only when the supporting cast is actually weak — not
        # when a scrub floor sits under an otherwise fine co-star stack
        # (e.g. Bird+Kawhi+Aguirre with English as the min).
        min_v = features.get("min_overall_value")
        if max_v >= SUPERSTAR_MAX_THRESHOLD and min_v is not None and min_v < -1.0:
            carry += SUPERSTAR_FLOOR_BOOST

    return carry


def describe_adjustment(features: dict) -> dict:
    """Debug helper for tests / notebooks."""
    tax = _fit_tax(features)
    carry = _star_carry(features)
    return {
        "tax": round(tax, 3),
        "carry": round(carry, 3),
        "net_delta": round(carry - tax, 3),
        "creator_excess": features.get("creator_excess"),
        "spacing_void": features.get("spacing_void"),
        "off_def_gap": features.get("off_def_gap"),
        "support_value": features.get("support_value"),
        "top2_overall_value": features.get("top2_overall_value"),
    }
