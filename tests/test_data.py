import chipscvn.data as data
import pytest

def test_get_categories():
	pdgs = [12]
	types = [1]
	conf = {"12-1": 3}
	assert data.get_categories(pdgs, types, conf) == [3]