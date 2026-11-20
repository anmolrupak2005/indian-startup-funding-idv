"""
tests/test_clean.py
Unit tests for data cleaning, amount parsing, date fixing,
and entity standardization routines in src/clean.py.
"""

import sys
import os
import math
import numpy as np
import pytest

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from clean import (
    clean_amount,
    parse_single_date,
    clean_city_name,
    assign_city_group,
    standardize_sector,
    standardize_investment_type,
    clean_investors
)


# -----------------------------------------------------------------------------
# 1. AMOUNT PARSER TESTS
# -----------------------------------------------------------------------------
def test_clean_amount_standard_values():
    assert clean_amount("1000000") == 1000000.0
    assert clean_amount("1,000,000") == 1000000.0
    assert clean_amount("50000.50") == 50000.50
    assert clean_amount(2500000) == 2500000.0


def test_clean_amount_symbols_and_artifacts():
    assert clean_amount("$50,000") == 50000.0
    assert clean_amount("$ 12,500,000") == 12500000.0
    assert clean_amount("\\xc2\\xa0$1,000,000") == 1000000.0
    assert clean_amount("\xc2\xa0200000") == 200000.0
    assert clean_amount("1,000,000+") == 1000000.0


def test_clean_amount_undisclosed_and_missing():
    assert math.isnan(clean_amount("undisclosed"))
    assert math.isnan(clean_amount("Undisclosed"))
    assert math.isnan(clean_amount("unknown"))
    assert math.isnan(clean_amount("N/A"))
    assert math.isnan(clean_amount("na"))
    assert math.isnan(clean_amount("None"))
    assert math.isnan(clean_amount(None))
    assert math.isnan(clean_amount(np.nan))
    assert math.isnan(clean_amount(""))


# -----------------------------------------------------------------------------
# 2. DATE FIXER TESTS
# -----------------------------------------------------------------------------
def test_parse_single_date_standard():
    iso, y, m = parse_single_date("01/02/2016")
    assert iso == "2016-02-01"
    assert y == 2016
    assert m == 2


def test_parse_single_date_malformed_fixes():
    # Dot separator e.g. 12/05.2015
    iso, y, m = parse_single_date("12/05.2015")
    assert iso == "2015-05-12"
    assert y == 2015
    assert m == 5

    # Trailing space e.g. "13/04/2015 "
    iso, y, m = parse_single_date("13/04/2015 ")
    assert iso == "2015-04-13"
    assert y == 2015
    assert m == 4

    # Missing slash e.g. 05/072018
    iso, y, m = parse_single_date("05/072018")
    assert iso == "2018-07-05"
    assert y == 2018
    assert m == 7


def test_parse_single_date_unparseable():
    iso, y, m = parse_single_date("invalid-date")
    assert iso is None
    assert y is None
    assert m is None

    iso, y, m = parse_single_date("")
    assert iso is None
    assert y is None
    assert m is None

    iso, y, m = parse_single_date(None)
    assert iso is None


# -----------------------------------------------------------------------------
# 3. CITY & CITY GROUP MAPPING TESTS
# -----------------------------------------------------------------------------
def test_clean_city_name_unification():
    assert clean_city_name("Bangalore") == "Bengaluru"
    assert clean_city_name("Bengaluru") == "Bengaluru"
    assert clean_city_name("Bangalore / USA") == "Bengaluru"

    assert clean_city_name("Gurgaon") == "Gurugram"
    assert clean_city_name("Gurugram") == "Gurugram"
    assert clean_city_name("Gurgaon / New Delhi") == "Gurugram"

    assert clean_city_name("New Delhi") == "Delhi"
    assert clean_city_name("Delhi") == "Delhi"

    assert clean_city_name("Bombay") == "Mumbai"
    assert clean_city_name("Mumbai") == "Mumbai"
    assert clean_city_name("Mumbai / USA") == "Mumbai"

    assert clean_city_name("Hyderabad") == "Hyderabad"
    assert clean_city_name("Pune") == "Pune"


def test_assign_city_group():
    assert assign_city_group("Delhi") == "Delhi NCR"
    assert assign_city_group("Gurugram") == "Delhi NCR"
    assert assign_city_group("Noida") == "Delhi NCR"
    assert assign_city_group("Bengaluru") == "Bengaluru"
    assert assign_city_group("Mumbai") == "Mumbai"


# -----------------------------------------------------------------------------
# 4. SECTOR & STAGE MAPPING TESTS
# -----------------------------------------------------------------------------
def test_standardize_sector():
    assert standardize_sector("eCommerce") == "E-Commerce"
    assert standardize_sector("E-Commerce") == "E-Commerce"
    assert standardize_sector("ecommerce") == "E-Commerce"

    assert standardize_sector("Fin-Tech") == "FinTech"
    assert standardize_sector("FinTech") == "FinTech"
    assert standardize_sector("fintech") == "FinTech"

    assert standardize_sector("EdTech") == "EdTech"
    assert standardize_sector("education") == "EdTech"

    assert standardize_sector("HealthTech") == "HealthTech"
    assert standardize_sector("healthcare") == "HealthTech"


def test_standardize_investment_type():
    assert standardize_investment_type("Seed Funding") == "Seed"
    assert standardize_investment_type("Seed / Angel Funding") == "Seed"
    assert standardize_investment_type("Seed / Angle Funding") == "Seed"
    assert standardize_investment_type("Angel Funding") == "Angel"
    assert standardize_investment_type("Series A") == "Series A"
    assert standardize_investment_type("Series B") == "Series B"
    assert standardize_investment_type("Series C") == "Series C+"
    assert standardize_investment_type("Series D") == "Series C+"
    assert standardize_investment_type("Private Equity") == "Private Equity"
    assert standardize_investment_type("Debt Funding") == "Debt"


def test_clean_investors():
    invs, count, is_undisc = clean_investors("Sequoia Capital India, Accel Partners")
    assert count == 2
    assert "Sequoia Capital India" in invs
    assert "Accel Partners" in invs
    assert is_undisc is False

    invs, count, is_undisc = clean_investors("Undisclosed Investors")
    assert is_undisc is True
    assert count == 0
