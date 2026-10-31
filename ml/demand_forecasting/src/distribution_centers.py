"""Maps customer state to macro-region to distribution center (DC).

Order location attribution is unreliable in this dataset: `location_id` does
not necessarily identify the responsible DC. The proxy uses
`dim_customer.state`, an independent field populated across all 27 states.
Each state maps to an IBGE macro-region, and each macro-region is served by a
DC; regions without a dedicated DC are assigned to the Southern DC.
This is a regional coverage rule, not a geodesic distance calculation."""

from __future__ import annotations

from dataclasses import dataclass

# Distribution centers by location_id, according to dim_location.
DC_SOUTH = 2  # Southern distribution center (PR)
DC_NORTH = 3  # Northern distribution center (AP)
DC_NORTHEAST = 4  # Northeastern distribution center (PI)

STATE_TO_MACRO_REGION: dict[str, str] = {
    # North
    "AC": "North",
    "AP": "North",
    "AM": "North",
    "PA": "North",
    "RO": "North",
    "RR": "North",
    "TO": "North",
    # Northeast
    "AL": "Northeast",
    "BA": "Northeast",
    "CE": "Northeast",
    "MA": "Northeast",
    "PB": "Northeast",
    "PE": "Northeast",
    "PI": "Northeast",
    "RN": "Northeast",
    "SE": "Northeast",
    # Central-West
    "DF": "Central-West",
    "GO": "Central-West",
    "MT": "Central-West",
    "MS": "Central-West",
    # Southeast
    "ES": "Southeast",
    "MG": "Southeast",
    "RJ": "Southeast",
    "SP": "Southeast",
    # South
    "PR": "South",
    "RS": "South",
    "SC": "South",
}

# Regions without a dedicated distribution center (Central-West and Southeast)
# are covered by the Southern distribution center; see README section 9.3.
MACRO_REGION_TO_DISTRIBUTION_CENTER: dict[str, int] = {
    "North": DC_NORTH,
    "Northeast": DC_NORTHEAST,
    "Central-West": DC_SOUTH,
    "Southeast": DC_SOUTH,
    "South": DC_SOUTH,
}

# State where each DC is physically located (dim_location), used only to derive
# which macro-regions have their own DC. Do not confuse this with
# STATE_TO_MACRO_REGION, which maps the *customer's* state.
DC_HOME_STATE: dict[int, str] = {
    DC_SOUTH: "PR",
    DC_NORTH: "AP",
    DC_NORTHEAST: "PI",
}


def macro_regions_with_dedicated_dc() -> frozenset[str]:
    """Return macro-regions with a physically located distribution center."""
    return frozenset(STATE_TO_MACRO_REGION[state] for state in DC_HOME_STATE.values())


def macro_regions_without_dedicated_dc() -> tuple[str, ...]:
    """Return macro-regions covered by a distribution center in another region."""
    dedicated = macro_regions_with_dedicated_dc()
    all_regions = set(MACRO_REGION_TO_DISTRIBUTION_CENTER.keys())
    return tuple(sorted(all_regions - dedicated))


@dataclass(frozen=True)
class StateCoverage:
    """One state-to-region-to-DC mapping row, ready for persistence."""

    state: str
    macro_region: str
    distribution_center_location_id: int


def build_state_coverage() -> tuple[StateCoverage, ...]:
    """Build the complete mapping of all 27 states to their designated DC."""
    return tuple(
        StateCoverage(
            state=state,
            macro_region=macro_region,
            distribution_center_location_id=MACRO_REGION_TO_DISTRIBUTION_CENTER[
                macro_region
            ],
        )
        for state, macro_region in sorted(STATE_TO_MACRO_REGION.items())
    )


def distribution_center_for_state(state: str) -> int | None:
    """Return the `location_id` of the designated DC for a state.

    Return `None` for a null state or one outside the 27 valid states. Callers
    decide how to handle it (see README section 9.3: exclude it from regional
    aggregation rather than assigning a default DC).
    """
    macro_region = STATE_TO_MACRO_REGION.get(state)
    if macro_region is None:
        return None
    return MACRO_REGION_TO_DISTRIBUTION_CENTER[macro_region]
