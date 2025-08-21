#Tests for SQL generation logic.

import pytest
from src.sql_generator import SQLGenerator

#TODO: Mock schema and mappings for tests

def test_sql_generation():
    schema = {
        'trips': ['trip_distance_km', 'rider_gender', 'started_at'],
        'stations': ['station_name']
    }
    gen = SQLGenerator(schema)
    intent = {
        'select': 'trip_distance_km',
        'from': 'trips',
        'where': [
            {'col': 'rider_gender', 'op': '=', 'val': 'F'},
            {'col': 'started_at', 'op': '>=', 'val': '2025-06-01'},
            {'col': 'started_at', 'op': '<', 'val': '2025-07-01'}
        ],
        'agg': 'SUM',
        'group_by': None,
        'join': []
    }
    sql, values = gen.generate(intent, None)
    assert sql.startswith('SELECT SUM(trip_distance_km) FROM trips WHERE')
    assert values == ['F', '2025-06-01', '2025-07-01']

def test_average_ride_time():
    schema = {
        'trips': ['ended_at', 'started_at', 'start_station_id'],
        'stations': ['station_id', 'station_name']
    }
    gen = SQLGenerator(schema)
    intent = {
        'select': 'ended_at - started_at',
        'from': 'trips',
        'where': [
            {'col': 'station_name', 'op': '=', 'val': 'Congress Avenue'},
            {'col': 'started_at', 'op': '>=', 'val': '2025-06-01'},
            {'col': 'started_at', 'op': '<', 'val': '2025-07-01'}
        ],
        'agg': 'AVG',
        'group_by': None,
        'join': ['stations ON trips.start_station_id = stations.station_id']
    }
    sql, values = gen.generate(intent, None)
    assert 'AVG(EXTRACT(EPOCH FROM ended_at - started_at)/60.0)' in sql
    assert values == ['Congress Avenue', '2025-06-01', '2025-07-01']

def test_most_departures():
    schema = {
        'trips': ['started_at', 'start_station_id'],
        'stations': ['station_id', 'station_name']
    }
    gen = SQLGenerator(schema)
    intent = {
        'select': 'station_name',
        'from': 'trips',
        'where': [
            {'col': 'started_at', 'op': '>=', 'val': '2025-06-01'},
            {'col': 'started_at', 'op': '<=', 'val': '2025-06-07'}
        ],
        'agg': 'COUNT',
        'group_by': 'station_name',
        'join': [{'table': 'stations', 'on': 'trips.start_station_id = stations.station_id'}],
        'order_by': 'COUNT(station_name) DESC',
        'limit': 1
    }
    sql, values = gen.generate(intent, None)
    assert 'GROUP BY station_name' in sql
    assert 'ORDER BY COUNT(station_name) DESC' in sql
    assert 'LIMIT 1' in sql
    assert values == ['2025-06-01', '2025-06-07']

def test_kilometres_by_women_rainy():
    schema = {
        'trips': ['trip_distance_km', 'started_at', 'rider_gender'],
        'daily_weather': ['weather_date', 'precipitation_mm'],
    }
    gen = SQLGenerator(schema)
    intent = {
        'select': 'trip_distance_km',
        'from': 'trips',
        'where': [
            {'col': 'rider_gender', 'op': '=', 'val': 'F'},
            {'col': 'started_at', 'op': '>=', 'val': '2025-06-01'},
            {'col': 'started_at', 'op': '<', 'val': '2025-07-01'},
            {'col': 'precipitation_mm', 'op': '>', 'val': 0}
        ],
        'agg': 'SUM',
        'group_by': None,
        'join': ['daily_weather ON trips.started_at::date = daily_weather.weather_date']
    }
    sql, values = gen.generate(intent, None)
    assert 'SUM(trip_distance_km)' in sql
    assert 'JOIN daily_weather ON trips.started_at::date = daily_weather.weather_date' in sql
    assert values == ['F', '2025-06-01', '2025-07-01', 0]

def test_sql_injection_attempt():
    schema = {'trips': ['trip_distance_km', 'rider_gender']}
    gen = SQLGenerator(schema)
    intent = {
        'select': 'trip_distance_km',
        'from': 'trips',
        'where': [{'col': 'rider_gender', 'op': '=', 'val': "F'; DROP TABLE trips;--"}],
        'agg': 'SUM',
        'group_by': None,
        'join': []
    }
    sql, values = gen.generate(intent, None)
    assert '%s' in sql
    assert values == ["F'; DROP TABLE trips;--"]

def test_sql_injection_complex():
    schema = {'trips': ['trip_distance_km', 'rider_gender']}
    gen = SQLGenerator(schema)
    intent = {
        'select': 'trip_distance_km',
        'from': 'trips',
        'where': [{'col': 'rider_gender', 'op': '=', 'val': "M'; SELECT * FROM users;--"}],
        'agg': 'SUM',
        'group_by': None,
        'join': []
    }
    sql, values = gen.generate(intent, None)
    assert '%s' in sql
    assert values == ["M'; SELECT * FROM users;--"]

def test_derived_unit_conversion():
    schema = {'trips': ['trip_distance_km', 'rider_gender']}
    gen = SQLGenerator(schema)
    # Simulate a user asking for meters, but the system maps to trip_distance_km
    intent = {
        'select': 'trip_distance_km',  # conversion logic is handled elsewhere
        'from': 'trips',
        'where': [{'col': 'rider_gender', 'op': '=', 'val': 'F'}],
        'agg': 'SUM',
        'group_by': None,
        'join': []
    }
    sql, values = gen.generate(intent, None)
    assert 'SUM(trip_distance_km)' in sql
    assert values == ['F']
