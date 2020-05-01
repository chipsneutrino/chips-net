import chipscvn.data as data
import pytest


def test_get_map():
    map_name = 't_nu_type'
    assert data.get_map(map_name).name == map_name
