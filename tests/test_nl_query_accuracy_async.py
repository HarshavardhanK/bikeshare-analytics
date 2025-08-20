import asyncio
import httpx
import psycopg2
import os
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

async def fetch(client, question):
    resp = await client.post(API_URL, json={"question": question})
    return resp.json()

async def main():
    print("\n--- NL Query Accuracy Test (Async) ---\n")
    total = 0
    correct = 0
    results = []

    # Prepare ground truth
    nl_sql_pairs = [
        ("How many kilometres were ridden by women in June 2025?",
         """
            SELECT SUM(trip_distance_km)
            FROM trips
            WHERE LOWER(rider_gender) = 'female'
              AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
         """),
        ("What was the average ride time for journeys that started at Congress Avenue in June 2025?",
         """
            SELECT AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60.0)
            FROM trips t
            JOIN stations s ON t.start_station_id = s.station_id
            WHERE s.station_name = 'Congress Avenue'
              AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
         """),
        ("Which docking point saw the most arrivals in June 2025?",
         """
            SELECT s.station_name, COUNT(*) as c
            FROM trips t
            JOIN stations s ON t.end_station_id = s.station_id
            WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
            GROUP BY s.station_name
            ORDER BY c DESC
            LIMIT 1;
         """),
        ("How many kilometres were ridden by Classic bikes in June 2025?",
         """
            SELECT SUM(t.trip_distance_km)
            FROM trips t
            JOIN bikes b ON t.bike_id = b.bike_id
            WHERE b.bike_model = 'Classic'
              AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
         """),
        ("How many rides were taken by non-binary riders on rainy days in June 2025?",
         """
            SELECT COUNT(*)
            FROM trips t
            JOIN daily_weather w ON DATE(t.started_at) = w.weather_date
            WHERE LOWER(t.rider_gender) = 'non-binary'
              AND w.precipitation_mm > 0
              AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01';
         """),
        ("What is the trip_id and birth year of the youngest rider in June 2025?",
         """
            SELECT trip_id, rider_birth_year
            FROM trips
            WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
            ORDER BY rider_birth_year DESC
            LIMIT 1;
         """),
    ]
    # Add 10 new queries
    nl_sql_pairs += [
        ("What is the total number of rides that started at Capitol Square in July 2025?",
         """
            SELECT COUNT(*) FROM trips WHERE start_station_id = (SELECT station_id FROM stations WHERE station_name = 'Capitol Square') AND started_at >= '2025-07-01' AND started_at < '2025-08-01';
         """),
        ("Which bike model had the longest single trip in June 2025?",
         """
            SELECT b.bike_model, MAX(t.trip_distance_km) FROM trips t JOIN bikes b ON t.bike_id = b.bike_id WHERE t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01' GROUP BY b.bike_model ORDER BY MAX(t.trip_distance_km) DESC LIMIT 1;
         """),
        ("What was the highest temperature recorded on a day with more than 5 rides in June 2025?",
         """
            SELECT MAX(w.high_temp_c) FROM daily_weather w JOIN trips t ON DATE(t.started_at) = w.weather_date WHERE t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01' GROUP BY w.weather_date HAVING COUNT(*) > 5 ORDER BY MAX(w.high_temp_c) DESC LIMIT 1;
         """),
        ("How many unique riders used E‑Bikes in June 2025?",
         """
            SELECT COUNT(DISTINCT t.rider_birth_year) FROM trips t JOIN bikes b ON t.bike_id = b.bike_id WHERE b.bike_model = 'E‑Bike' AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01';
         """),
        ("What is the average distance of trips ending at Congress Avenue in June 2025?",
         """
            SELECT AVG(trip_distance_km) FROM trips t JOIN stations s ON t.end_station_id = s.station_id WHERE s.station_name = 'Congress Avenue' AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01';
         """),
        ("Which station had the lowest capacity?",
         """
            SELECT station_name, capacity FROM stations ORDER BY capacity ASC LIMIT 1;
         """),
        ("How many trips started and ended at the same station in June 2025?",
         """
            SELECT COUNT(*) FROM trips WHERE start_station_id = end_station_id AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
         """),
        ("What is the total precipitation on days with at least one trip by a non-binary rider in June 2025?",
         """
            SELECT SUM(w.precipitation_mm) FROM daily_weather w WHERE EXISTS (SELECT 1 FROM trips t WHERE LOWER(t.rider_gender) = 'non-binary' AND DATE(t.started_at) = w.weather_date AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01');
         """),
        ("Which bike was acquired most recently?",
         """
            SELECT bike_id, bike_model, acquisition_date FROM bikes ORDER BY acquisition_date DESC LIMIT 1;
         """),
        ("What is the average age of riders (in 2025) for trips longer than 5 km in June 2025?",
         """
            SELECT AVG(2025 - rider_birth_year) FROM trips WHERE trip_distance_km > 5 AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
         """),
    ]

    # Get ground truth answers
    gt = []
    for _, sql in nl_sql_pairs:
        res = run_query(sql)
        if not res:
            gt.append(None)
        elif isinstance(res[0], (tuple, list)) and len(res[0]) > 1:
            gt.append(res[0])
        else:
            gt.append(res[0][0])

    questions = [q for q, _ in nl_sql_pairs]

    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks = [fetch(client, q) for q in questions]
        api_responses = await asyncio.gather(*tasks)

    # Evaluate
    for i, (q, api, gt_val) in enumerate(zip(questions, api_responses, gt), 1):
        api_result = api.get('result')
        ok = False
        # Existing logic for first 6
        if i == 1 or i == 4:  # km
            api_val = float(str(api_result).replace(' km','')) if api_result else None
            ok = almost_equal(api_val, gt_val)
        elif i == 2:  # min
            api_val = float(str(api_result).replace(' min','')) if api_result else None
            ok = almost_equal(api_val, gt_val)
        elif i == 3:  # station name
            ok = (str(gt_val[0]) in str(api_result)) if api_result else False
        elif i == 5:  # count
            ok = (str(gt_val) in str(api_result)) if api_result else False
        elif i == 6:  # trip_id and birth year
            ok = (str(gt_val[0]) in str(api_result) and str(gt_val[1]) in str(api_result)) if api_result else False
        # New queries
        elif i == 7:  # total rides at Capitol Square in July
            ok = (str(gt_val) in str(api_result)) if api_result is not None else False
        elif i == 8:  # bike model with longest trip
            ok = (str(gt_val[0]) in str(api_result) and str(gt_val[1]) in str(api_result)) if api_result else False
        elif i == 9:  # highest temp on busy day
            ok = (str(gt_val) in str(api_result)) if api_result is not None else gt_val is None
        elif i == 10:  # unique riders on E-Bikes
            ok = (str(gt_val) in str(api_result)) if api_result is not None else False
        elif i == 11:  # avg distance to Congress Avenue
            try:
                api_val = float(str(api_result)) if api_result else None
                ok = almost_equal(api_val, gt_val)
            except Exception:
                ok = False
        elif i == 12:  # station with lowest capacity
            ok = (str(gt_val[0]) in str(api_result) and str(gt_val[1]) in str(api_result)) if api_result else False
        elif i == 13:  # trips started and ended at same station
            ok = (str(gt_val) in str(api_result)) if api_result is not None else False
        elif i == 14:  # total precipitation on non-binary trip days
            try:
                api_val = float(str(api_result)) if api_result else None
                ok = almost_equal(api_val, gt_val)
            except Exception:
                ok = False
        elif i == 15:  # most recently acquired bike
            ok = (str(gt_val[0]) in str(api_result) and str(gt_val[1]) in str(api_result) and str(gt_val[2]) in str(api_result)) if api_result else False
        elif i == 16:  # avg age for >5km trips
            try:
                api_val = float(str(api_result)) if api_result else None
                ok = almost_equal(api_val, gt_val)
            except Exception:
                ok = False
        results.append((q, api_result, gt_val, ok))

    print("\n--- Results ---\n")
    total = len(results)
    correct = sum(1 for r in results if r[3])
    for i, (q, api, gt, ok) in enumerate(results, 1):
        print(f"Q{i}: {q}\n  API: {api}\n  GT: {gt}\n  {'✅' if ok else '❌'}\n")
    print(f"Overall accuracy: {correct}/{total} = {round(100*correct/total,1)}%\n")

if __name__ == "__main__":
    asyncio.run(main())
