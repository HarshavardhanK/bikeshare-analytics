import requests
import psycopg2
import os
import json
from decimal import Decimal

DB_PARAMS = dict(
    dbname='bike-share-assessment',
    user='attriassessment',
    password='(.aG0X>322Uk',
    host='agentify-assessment.postgres.database.azure.com',
    port=5432
)
API_URL = 'http://localhost:8000/query'

# Helper to run SQL
def run_query(sql, params=None):
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()

def almost_equal(a, b, tol=0.01):
    try:
        return abs(float(a) - float(b)) < tol
    except Exception:
        return a == b

def main():
    print("\n--- NL Query Accuracy Test ---\n")
    total = 0
    correct = 0
    results = []

    # 1. Total km ridden by women in June 2025
    nl1 = "How many kilometres were ridden by women in June 2025?"
    sql1 = """
        SELECT SUM(trip_distance_km)
        FROM trips
        WHERE LOWER(rider_gender) = 'female'
          AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
    """
    gt1 = run_query(sql1)[0][0]
    resp1 = requests.post(API_URL, json={"question": nl1}).json()
    api1 = resp1.get('result')
    api1_val = float(str(api1).replace(' km','')) if api1 else None
    ok1 = almost_equal(api1_val, gt1)
    results.append((nl1, api1, gt1, ok1))

    # 2. Average ride time for journeys that started at Congress Avenue in June 2025
    nl2 = "What was the average ride time for journeys that started at Congress Avenue in June 2025?"
    sql2 = """
        SELECT AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60.0)
        FROM trips t
        JOIN stations s ON t.start_station_id = s.station_id
        WHERE s.station_name = 'Congress Avenue'
          AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
    """
    gt2 = run_query(sql2)[0][0]
    resp2 = requests.post(API_URL, json={"question": nl2}).json()
    api2 = resp2.get('result')
    api2_val = float(str(api2).replace(' min','')) if api2 else None
    ok2 = almost_equal(api2_val, gt2)
    results.append((nl2, api2, gt2, ok2))

    # 3. Most popular end station in June 2025
    nl3 = "Which docking point saw the most arrivals in June 2025?"
    sql3 = """
        SELECT s.station_name, COUNT(*) as c
        FROM trips t
        JOIN stations s ON t.end_station_id = s.station_id
        WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
        GROUP BY s.station_name
        ORDER BY c DESC
        LIMIT 1;
    """
    gt3 = run_query(sql3)[0][0]
    resp3 = requests.post(API_URL, json={"question": nl3}).json()
    api3 = resp3.get('result')
    ok3 = (gt3 in str(api3)) if api3 else False
    results.append((nl3, api3, gt3, ok3))

    # 4. Total km by bike model in June 2025 (just check Classic)
    nl4 = "How many kilometres were ridden by Classic bikes in June 2025?"
    sql4 = """
        SELECT SUM(t.trip_distance_km)
        FROM trips t
        JOIN bikes b ON t.bike_id = b.bike_id
        WHERE b.bike_model = 'Classic'
          AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
    """
    gt4 = run_query(sql4)[0][0]
    resp4 = requests.post(API_URL, json={"question": nl4}).json()
    api4 = resp4.get('result')
    api4_val = float(str(api4).replace(' km','')) if api4 else None
    ok4 = almost_equal(api4_val, gt4)
    results.append((nl4, api4, gt4, ok4))

    # 5. Rides by non-binary riders on rainy days in June 2025
    nl5 = "How many rides were taken by non-binary riders on rainy days in June 2025?"
    sql5 = """
        SELECT COUNT(*)
        FROM trips t
        JOIN daily_weather w ON DATE(t.started_at) = w.weather_date
        WHERE LOWER(t.rider_gender) = 'non-binary'
          AND w.precipitation_mm > 0
          AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01';
    """
    gt5 = run_query(sql5)[0][0]
    resp5 = requests.post(API_URL, json={"question": nl5}).json()
    api5 = resp5.get('result')
    ok5 = (str(gt5) in str(api5)) if api5 else False
    results.append((nl5, api5, gt5, ok5))

    # 6. Youngest rider's trip in June 2025
    nl6 = "What is the trip_id and birth year of the youngest rider in June 2025?"
    sql6 = """
        SELECT trip_id, rider_birth_year
        FROM trips
        WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
        ORDER BY rider_birth_year DESC
        LIMIT 1;
    """
    gt6 = run_query(sql6)[0]
    resp6 = requests.post(API_URL, json={"question": nl6}).json()
    api6 = resp6.get('result')
    ok6 = (str(gt6[0]) in str(api6) and str(gt6[1]) in str(api6)) if api6 else False
    results.append((nl6, api6, gt6, ok6))

    # Print results and score
    print("\n--- Results ---\n")
    for i, (q, api, gt, ok) in enumerate(results, 1):
        print(f"Q{i}: {q}\n  API: {api}\n  GT: {gt}\n  {'✅' if ok else '❌'}\n")
        total += 1
        if ok:
            correct += 1
    print(f"Overall accuracy: {correct}/{total} = {round(100*correct/total,1)}%\n")

if __name__ == "__main__":
    main()
