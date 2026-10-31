"""Sanity checks for the state-to-region-to-distribution-center mapping.

These assertions protect the mapping documented in README section 9 from
silent drift. Changes to either source should be reflected in the other.
"""

from __future__ import annotations

from ml.demand_forecasting.src.distribution_centers import (
    DC_NORTH,
    DC_NORTHEAST,
    DC_SOUTH,
    MACRO_REGION_TO_DISTRIBUTION_CENTER,
    STATE_TO_MACRO_REGION,
    build_state_coverage,
    distribution_center_for_state,
    macro_regions_with_dedicated_dc,
    macro_regions_without_dedicated_dc,
)

ALL_27_STATES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
}  # fmt: skip


def test_covers_all_27_states():
    assert set(STATE_TO_MACRO_REGION) == ALL_27_STATES


def test_every_macro_region_has_a_designated_dc():
    expected_regions = {"North", "Northeast", "Central-West", "Southeast", "South"}
    assert set(STATE_TO_MACRO_REGION.values()) == expected_regions
    assert set(MACRO_REGION_TO_DISTRIBUTION_CENTER) == expected_regions


def test_dc_ids_match_readme():
    # location_id values for the three centers, per dim_location and README section 9.1.
    assert DC_SOUTH == 2
    assert DC_NORTH == 3
    assert DC_NORTHEAST == 4


def test_regions_without_own_dc_fall_back_to_south():
    # Central-West and Southeast are served by the Southern distribution center.
    assert MACRO_REGION_TO_DISTRIBUTION_CENTER["Central-West"] == DC_SOUTH
    assert MACRO_REGION_TO_DISTRIBUTION_CENTER["Southeast"] == DC_SOUTH


def test_build_state_coverage_has_one_row_per_state():
    coverage = build_state_coverage()
    assert len(coverage) == 27
    assert {row.state for row in coverage} == ALL_27_STATES
    assert all(row.distribution_center_location_id in (2, 3, 4) for row in coverage)


def test_distribution_center_for_known_state():
    assert distribution_center_for_state("PR") == DC_SOUTH
    assert distribution_center_for_state("AP") == DC_NORTH
    assert distribution_center_for_state("PI") == DC_NORTHEAST
    # AL has no distribution center in the state; it maps to the Northeast center in PI.
    assert distribution_center_for_state("AL") == DC_NORTHEAST


def test_distribution_center_for_null_or_unmapped_state_returns_none():
    assert distribution_center_for_state(None) is None
    assert distribution_center_for_state("") is None
    assert distribution_center_for_state("XX") is None


def test_macro_regions_with_dedicated_dc_are_the_three_home_regions():
    # The Southern center is in PR, the Northern center in AP, and the Northeastern center in PI.
    assert macro_regions_with_dedicated_dc() == {"South", "North", "Northeast"}


def test_macro_regions_without_dedicated_dc_are_candidates():
    # Candidate regions are served by a center located in another region.
    assert macro_regions_without_dedicated_dc() == ("Central-West", "Southeast")


def test_dedicated_and_candidate_regions_partition_all_five():
    dedicated = macro_regions_with_dedicated_dc()
    candidates = macro_regions_without_dedicated_dc()
    all_regions = set(STATE_TO_MACRO_REGION.values())
    assert dedicated | set(candidates) == all_regions
    assert not (dedicated & set(candidates))
